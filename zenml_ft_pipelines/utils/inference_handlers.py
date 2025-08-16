"""Inference handlers abstraction for standardized evaluation logic.

This module defines:
- BaseInferenceHandler: Abstract base class providing common prompt building and
  label normalization functionality.
- FinetunedInferenceHandler: Concrete handler for local fine-tuned models using
  Hugging Face/Unsloth models.
- DeepSeekInferenceHandler: Concrete handler for DeepSeek model via LiteLLM/OpenRouter.

Design decisions:
- NONE label is excluded from prompts by default (configurable).
- The handler returns (Optional[str], str) = (predicted_label_or_None, raw_response).
  The evaluation runner will convert None/invalid to "PARSING_ERROR" and truncate
  responses uniformly for logging/metrics.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from transformers import TextStreamer
from utils.evaluation_utils import create_base_prompt_json, extract_prediction

try:
    # Import inside try to avoid hard failure if litellm is not installed where not needed
    import litellm
    from litellm import completion as litellm_completion
    # Best-effort: reduce noisy logs if library is present
    try:
        litellm.suppress_debug_info = True  # type: ignore[attr-defined]
    except Exception:
        pass
except Exception:  # pragma: no cover - litellm optional at import time for local eval
    litellm = None  # type: ignore[assignment]
    litellm_completion = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


class BaseInferenceHandler(ABC):
    """Abstract base class for single-item inference handlers."""

    def __init__(
        self,
        label_schema: List[str],
        include_none_in_prompt: bool = False,
    ) -> None:
        """
        Args:
            label_schema: Full list of valid labels for evaluation.
            include_none_in_prompt: If True, includes "NONE" in prompt valid labels.
        """
        self.label_schema = label_schema
        self.include_none_in_prompt = include_none_in_prompt

    @property
    def supports_concurrency(self) -> bool:
        """Whether this handler supports concurrent inference execution."""
        return False

    def build_prompt_json(self, item: Dict[str, Any]) -> str:
        """Build the prompt JSON using shared utility, filtering NONE by default."""
        valid_labels = (
            self.label_schema
            if self.include_none_in_prompt
            else [label for label in self.label_schema if label != "NONE"]
        )
        return create_base_prompt_json(item, valid_labels)

    def normalize_label(self, label: Optional[str]) -> Optional[str]:
        """Return label if within schema; otherwise None for downstream 'PARSING_ERROR'."""
        if label is None:
            return None
        return label if label in self.label_schema else None

    @abstractmethod
    def run_single(
        self, item: Dict[str, Any], verbose: bool = False, **kwargs: Any
    ) -> Tuple[Optional[str], str]:
        """Run inference for a single item.

        Args:
            item: Input example dict.
            verbose: Enable verbose generation (e.g., streaming).
            **kwargs: Extra backend-specific parameters.

        Returns:
            Tuple of (predicted_label_or_None, raw_response_text).
        """
        raise NotImplementedError("Subclasses must implement run_single().")


class FinetunedInferenceHandler(BaseInferenceHandler):
    """Inference handler for local fine-tuned models (HF/Unsloth)."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        label_schema: List[str],
        include_none_in_prompt: bool = False,
        generate_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            model: Loaded local model instance.
            tokenizer: Corresponding tokenizer (chat template should be configured by caller).
            label_schema: Full list of valid labels for evaluation.
            include_none_in_prompt: If True, includes "NONE" in prompt valid labels.
            generate_kwargs: Default generation kwargs for model.generate.
        """
        super().__init__(label_schema, include_none_in_prompt)
        self.model = model
        self.tokenizer = tokenizer
        self.generate_kwargs = generate_kwargs or {
            "max_new_tokens": 1024,
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 64,
        }

    @property
    def supports_concurrency(self) -> bool:
        """Local GPU-bound inference should default to sequential execution."""
        return False

    def _build_messages(self, prompt_json: str) -> List[Dict[str, Any]]:
        """Build Gemma-style chat messages from prompt JSON."""
        return [{"role": "user", "content": [{"type": "text", "text": prompt_json}]}]

    def run_single(
        self, item: Dict[str, Any], verbose: bool = False, **kwargs: Any
    ) -> Tuple[Optional[str], str]:
        """Run single-item inference using the local fine-tuned model."""
        try:
            prompt_json = self.build_prompt_json(item)
            messages = self._build_messages(prompt_json)

            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )

            # Tokenize and move to model device
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

            # Optional streaming for verbose mode
            streamer = TextStreamer(self.tokenizer, skip_prompt=True) if verbose else None

            # Merge default and per-call kwargs
            generate_params = dict(self.generate_kwargs)
            if kwargs:
                generate_params.update(kwargs)

            # Generate output
            outputs = self.model.generate(
                **inputs,
                streamer=streamer,
                **generate_params,
            )

            # Decode and extract generated portion
            decoded = self.tokenizer.batch_decode(outputs)[0]
            response_text = decoded[len(text) :].strip()

            # Parse prediction
            pred_label = extract_prediction(response_text, log_failures=False)
            return pred_label, response_text

        except Exception as e:
            error_msg = f"Error during local inference: {type(e).__name__}: {str(e)}"
            logger.error(error_msg)
            return None, error_msg


class DeepSeekInferenceHandler(BaseInferenceHandler):
    """Inference handler for DeepSeek via LiteLLM/OpenRouter."""

    def __init__(
        self,
        model_name: str,
        label_schema: List[str],
        include_none_in_prompt: bool = False,
        default_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            model_name: Fully qualified model name for LiteLLM/OpenRouter.
            label_schema: Full list of valid labels for evaluation.
            include_none_in_prompt: If True, includes "NONE" in prompt valid labels.
            default_kwargs: Default kwargs passed to litellm.completion (e.g., temperature).
        """
        super().__init__(label_schema, include_none_in_prompt)
        self.model_name = model_name
        self.default_kwargs = default_kwargs or {
            "temperature": 0.1,
            "max_tokens": 1024,
            "response_format": {"type": "json_object"},
        }

    @property
    def supports_concurrency(self) -> bool:
        """API-bound inference supports concurrency by default."""
        return True

    def run_single(
        self, item: Dict[str, Any], verbose: bool = False, **kwargs: Any
    ) -> Tuple[Optional[str], str]:
        """Run single-item inference using LiteLLM/OpenRouter API."""
        if litellm_completion is None:
            error_msg = (
                "Error during API inference: litellm is not installed or failed to import."
            )
            logger.error(error_msg)
            return None, error_msg

        prompt_json = self.build_prompt_json(item)

        # Build request messages: simple string content for API
        messages = [{"role": "user", "content": prompt_json}]

        # Merge default and per-call kwargs
        completion_kwargs: Dict[str, Any] = dict(self.default_kwargs)
        if kwargs:
            completion_kwargs.update(kwargs)

        try:
            response = litellm_completion(
                model=self.model_name,
                messages=messages,
                **completion_kwargs,
            )

            # Guard against missing fields
            try:
                response_text = response.choices[0].message.content or ""
            except Exception:
                response_text = ""

            pred_label = extract_prediction(response_text, log_failures=False)
            return pred_label, response_text

        except Exception as e:
            error_msg = f"Error during API inference: {type(e).__name__}: {str(e)}"
            logger.error(error_msg)
            return None, error_msg