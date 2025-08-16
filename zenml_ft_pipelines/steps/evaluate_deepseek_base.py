import logging
from datetime import datetime
from typing import Annotated, Any, Dict, Tuple

import polars as pl
from constants import LEGAL_LABEL_SCHEMA, OPEN_ROUTER_EVALS_BASE_MODEL_NAME
from utils.inference_handlers import DeepSeekInferenceHandler
from utils.evaluation_runner import evaluate_dataset
from utils.visualization_utils import create_evaluation_visualization
from zenml import step
from zenml.types import HTMLString

# Reduce noisy logs from underlying HTTP clients if present
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)






@step(enable_cache=False)
def evaluate_deepseek_base(
    test_dataset: pl.DataFrame,
    max_workers: int = 5,
    verbose: bool = False,
) -> Tuple[
    Annotated[Dict[str, Any], "deepseek_evaluation_results"],
    Annotated[HTMLString, "deepseek_evaluation_viz"],
]:
    """Evaluate DeepSeek base model on test data.

    Args:
        test_dataset: Test dataset to evaluate on
        max_workers: Maximum number of concurrent API calls
        verbose: Whether to print detailed outputs

    Returns:
        Tuple of evaluation results and visualization
    """
    # Convert polars DataFrame to list of dicts for evaluation
    evaluation_items = test_dataset.to_dicts()
    total_items = len(evaluation_items)

    logger.info(f"Starting DeepSeek evaluation on {total_items} test examples")
    logger.info(f"Using model: {OPEN_ROUTER_EVALS_BASE_MODEL_NAME}")
    logger.info(f"Max concurrent workers: {max_workers}")

    # Set up the inference handler for DeepSeek/OpenRouter
    handler = DeepSeekInferenceHandler(
        model_name=OPEN_ROUTER_EVALS_BASE_MODEL_NAME,
        label_schema=LEGAL_LABEL_SCHEMA,
        include_none_in_prompt=False,
    )

    # Run evaluation using the unified runner (handles concurrency and metrics)
    records, results = evaluate_dataset(
        handler,
        evaluation_items,
        verbose=verbose,
        max_workers=max_workers,
        log_every=10,
    )

    # Add metadata (kept consistent with previous implementation)
    metadata = {
        "model_size": "DeepSeek Base",
        "model_path": OPEN_ROUTER_EVALS_BASE_MODEL_NAME,
        "model_display_name": "DeepSeek R1 (Base Model)",
        "base_model": "deepseek-r1-0528",
        "test_data_path": "Provided by pipeline",
        "use_local_model": False,
        "evaluation_timestamp": datetime.now().isoformat(),
        "max_workers": max_workers,
    }

    # Combine results with metadata
    full_results = {**results, "metadata": metadata}

    # Create visualization with metadata
    viz = create_evaluation_visualization(full_results)

    # Return as plain dict for ZenML compatibility
    return dict(full_results), viz
