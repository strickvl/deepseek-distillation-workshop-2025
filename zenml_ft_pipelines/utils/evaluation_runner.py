"""Dataset evaluation runner and results aggregation utilities.

This module centralizes dataset-wide inference execution using the inference
handlers abstraction and produces standardized evaluation metrics and records.

Exports:
- InferenceRecord: TypedDict representing a single evaluated sample.
- evaluate_dataset: Runs inference over a list of items, with optional concurrency,
  preserving input order and normalizing predictions.
- aggregate_results_from_records: Aggregates metrics (accuracy, non-NONE accuracy,
  precision/recall/F1, confusion matrix, classification report) into a dict shape
  expected by existing visualization and comparison utilities.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from utils.evaluation_utils import calculate_detailed_metrics
from utils.inference_handlers import BaseInferenceHandler

logger = logging.getLogger(__name__)


class InferenceRecord(TypedDict):
    """Per-sample inference result structure."""
    index: int
    item: Dict[str, Any]
    true_label: str
    pred_label: str  # "PARSING_ERROR" on parse/validation failure
    response: str    # truncated to 500 chars for storage/logging


def _truncate_response(text: str, limit: int = 500) -> str:
    """Truncate a response string to a maximum number of characters."""
    if text is None:
        return ""
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def evaluate_dataset(
    handler: BaseInferenceHandler,
    items: List[Dict[str, Any]],
    verbose: bool = False,
    max_workers: Optional[int] = None,
    log_every: int = 10,
) -> Tuple[List[InferenceRecord], Dict[str, Any]]:
    """Evaluate a dataset using the provided inference handler.

    Args:
        handler: Inference handler implementing run_single and normalization.
        items: List of dataset items (each must include a 'label' key).
        verbose: Enable verbose mode for handlers that support it.
        max_workers: Max concurrency for handlers that support it.
        log_every: Logging progress interval (in number of items).

    Returns:
        Tuple of (records, aggregated_results)
    """
    total = len(items)
    records: List[InferenceRecord] = [  # preallocate to preserve order
        {
            "index": i,
            "item": items[i],
            "true_label": items[i].get("label", ""),
            "pred_label": "PARSING_ERROR",
            "response": "",
        }
        for i in range(total)
    ]

    def build_record(idx: int, raw_pred: Optional[str], raw_resp: str) -> InferenceRecord:
        normalized = handler.normalize_label(raw_pred)
        pred_label = normalized if normalized is not None else "PARSING_ERROR"
        return {
            "index": idx,
            "item": items[idx],
            "true_label": items[idx].get("label", ""),
            "pred_label": pred_label,
            "response": _truncate_response(raw_resp, 500),
        }

    if handler.supports_concurrency and (max_workers or 0) > 1 and total > 0:
        logger.info(f"Evaluating with concurrency (max_workers={max_workers})...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(handler.run_single, items[i], verbose): i
                for i in range(total)
            }
            completed = 0
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    pred, resp = fut.result()
                except Exception as e:
                    pred, resp = None, f"Error during inference: {type(e).__name__}: {str(e)}"
                    logger.error(resp)

                records[idx] = build_record(idx, pred, resp)

                completed += 1
                if completed % log_every == 0 or completed == total:
                    logger.info(f"Progress: {completed}/{total} samples evaluated...")
    else:
        logger.info("Evaluating sequentially...")
        for i, item in enumerate(items):
            if i % log_every == 0 or i + 1 == total:
                logger.info(f"Processing example {i + 1}/{total}...")
            try:
                pred, resp = handler.run_single(item, verbose=verbose)
            except Exception as e:
                pred, resp = None, f"Error during inference: {type(e).__name__}: {str(e)}"
                logger.error(resp)

            records[i] = build_record(i, pred, resp)

    aggregated = aggregate_results_from_records(records, handler.label_schema)
    return records, aggregated


def aggregate_results_from_records(
    records: List[InferenceRecord], label_schema: List[str]
) -> Dict[str, Any]:
    """Aggregate evaluation metrics from per-sample records.

    Args:
        records: List of InferenceRecord objects in dataset order.
        label_schema: Complete list of valid labels for metrics computation.

    Returns:
        Dictionary shaped like EvaluationResults with all required keys.
    """
    total_count = len(records)
    correct_count = sum(1 for r in records if r["pred_label"] == r["true_label"])
    accuracy = (correct_count / total_count) if total_count > 0 else 0.0

    # Non-NONE metrics
    non_none_records = [r for r in records if r["true_label"] != "NONE"]
    non_none_total = len(non_none_records)
    non_none_correct_count = sum(
        1 for r in non_none_records if r["pred_label"] == r["true_label"]
    )
    non_none_accuracy = (
        (non_none_correct_count / non_none_total) if non_none_total > 0 else 0.0
    )

    # Errors: include mispredictions and parsing errors
    errors: List[Dict[str, Any]] = [
        {
            "item": r["item"],
            "true_label": r["true_label"],
            "pred_label": r["pred_label"],
            "response": r["response"],
        }
        for r in records
        if r["pred_label"] != r["true_label"]
    ]

    # Prepare lists for detailed metrics
    true_labels = [r["true_label"] for r in records]
    pred_labels = [r["pred_label"] for r in records]

    # Detailed metrics (precision/recall/F1, per-class, confusion matrix, report)
    detailed_metrics = calculate_detailed_metrics(true_labels, pred_labels, label_schema)

    results: Dict[str, Any] = {
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_count": total_count,
        "errors": errors,
        "non_none_accuracy": non_none_accuracy,
        "non_none_correct_count": non_none_correct_count,
        "non_none_total_count": non_none_total,
        # Detailed metrics
        "precision": detailed_metrics["overall_precision"],
        "recall": detailed_metrics["overall_recall"],
        "f1_score": detailed_metrics["overall_f1"],
        "per_class_precision": detailed_metrics["per_class_precision"],
        "per_class_recall": detailed_metrics["per_class_recall"],
        "per_class_f1": detailed_metrics["per_class_f1"],
        "confusion_matrix": detailed_metrics["confusion_matrix"],
        "classification_report": detailed_metrics["classification_report"],
    }

    return results