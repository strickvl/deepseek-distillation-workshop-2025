import logging
from datetime import datetime
from typing import Annotated, Any, Dict, Tuple

import polars as pl
from constants import LEGAL_LABEL_SCHEMA, LOAD_IN_4BIT, LOAD_IN_8BIT, TEST_DATA_PATH
from peft import PeftModel
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from utils.custom_exceptions import ModelLoadError
from utils.visualization_utils import create_evaluation_visualization
from zenml import step
from zenml.types import HTMLString

# New unified inference/evaluation imports
from utils.inference_handlers import FinetunedInferenceHandler
from utils.evaluation_runner import evaluate_dataset

from steps.trainer import get_model_config

logger = logging.getLogger(__name__)






@step(enable_cache=False)
def evaluate_model(
    test_dataset: pl.DataFrame,
    model_size: str = "4b",
    verbose: bool = False,
    use_local_model: bool = True,
) -> Tuple[
    Annotated[Dict[str, Any], "evaluation_results"],
    Annotated[HTMLString, "evaluation_viz"],
]:
    """Evaluate a fine-tuned model on test data.

    Args:
        test_dataset: Test dataset to evaluate on
        model_size: Model size to use ('1b', '4b', or '12b')
        verbose: Whether to print detailed outputs
        use_local_model: If True, load from local path; if False, load from HF Hub

    Returns:
        Tuple of evaluation results and visualization
    """
    # Get model configuration
    config = get_model_config(model_size)

    # Determine model path
    if use_local_model:
        model_path = f"{config.output_dir}/final"
        logger.info(f"Loading LoRA adapters from local path: {model_path}")
    else:
        model_path = config.hf_repo
        logger.info(f"Loading LoRA adapters from HF Hub: {model_path}")

    logger.info("Model Configuration:")
    logger.info(f"  Base model: {config.base_model_id}")
    logger.info(f"  Model size: {model_size}")
    logger.info(f"  Max sequence length: {config.max_seq_length}")
    logger.info(f"  4-bit loading: {LOAD_IN_4BIT}")
    logger.info(f"  8-bit loading: {LOAD_IN_8BIT}")

    logger.info(f"Loading base model from {config.base_model_id}...")

    # Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.base_model_id,
        max_seq_length=config.max_seq_length,
        load_in_4bit=LOAD_IN_4BIT,
        load_in_8bit=LOAD_IN_8BIT,
    )

    logger.info("✓ Base model loaded successfully")

    # Enable faster inference
    FastLanguageModel.for_inference(model)
    logger.info("✓ Enabled faster inference mode")

    # Set up chat template
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
    logger.info("✓ Chat template configured")

    # Load adapters
    logger.info(f"Loading LoRA adapters from {model_path}...")
    try:
        model = PeftModel.from_pretrained(model, model_path)
        logger.info(f"✓ LoRA adapters loaded successfully from {model_path}")
    except Exception as e:
        logger.error(f"✗ Failed to load LoRA adapters: {str(e)}")
        raise ModelLoadError(
            f"Cannot load model adapters from {model_path}",
            model_path=model_path,
            model_size=model_size,
            is_training_required=use_local_model,
        ) from e

    # Convert polars DataFrame to list of dicts for evaluation
    evaluation_items = test_dataset.to_dicts()

    logger.info(f"Starting evaluation on {len(evaluation_items)} test examples")

    # Use unified inference handler and evaluation runner
    handler = FinetunedInferenceHandler(
        model=model,
        tokenizer=tokenizer,
        label_schema=LEGAL_LABEL_SCHEMA,
        include_none_in_prompt=False,
        generate_kwargs={
            "max_new_tokens": 1024,
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 64,
        },
    )

    records, results = evaluate_dataset(
        handler,
        evaluation_items,
        verbose=verbose,
        max_workers=None,
        log_every=10,
    )

    # Add metadata
    metadata = {
        "model_size": model_size,
        "model_path": model_path,
        "model_display_name": "Gemma 3 1B"
        if model_size == "1b"
        else ("Gemma 3 4B" if model_size == "4b" else "Gemma 3 12B"),
        "base_model": config.base_model_id,
        "test_data_path": str(TEST_DATA_PATH),
        "use_local_model": use_local_model,
        "evaluation_timestamp": datetime.now().isoformat(),
        "excluded_none": len(test_dataset)
        < len(evaluation_items) + (len(test_dataset) - len(evaluation_items)),
    }

    # Combine results with metadata
    full_results = {**results, "metadata": metadata}

    # Create visualization with metadata
    viz = create_evaluation_visualization(full_results)

    # Return as plain dict for ZenML compatibility
    return dict(full_results), viz
