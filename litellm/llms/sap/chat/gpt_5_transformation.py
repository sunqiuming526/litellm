"""Support for SAP AI Core gpt-5 model family."""

from .transformation import SAPChatConfig


class SAPGPT5Config(SAPChatConfig):
    """Configuration for gpt-5 models in SAP AI Core.

    Handles SAP AI Core and OpenAI API quirks for the gpt-5 series like:

    - Mapping ``max_tokens`` -> ``max_completion_tokens``.
    - Dropping unsupported ``temperature`` values when requested.
    - SAP-specific parameter handling.
    """
    
    @classmethod
    def is_model_gpt_5_model(cls, model: str) -> bool:
        return "gpt-5" in model
    
    def get_supported_openai_params(self, model: str) -> list:
        base_sap_params = super().get_supported_openai_params(model=model)
        gpt_5_only_params = ["reasoning_effort",  "reasoning"]
        base_sap_params.extend(gpt_5_only_params)
        return base_sap_params

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool = False,
    ) -> dict:
        # Set default reasoning configuration for GPT-5 models
        if self.is_model_gpt_5_model(model):
            reasoning_config = {
                "effort": "minimal",
                "summary": "auto"
            }
            
            # Check if user provided reasoning_effort, use it instead of default
            if "reasoning" in non_default_params:
                reasoning_config["effort"] = non_default_params.pop("reasoning")
            
            optional_params["reasoning"] = reasoning_config

        # Call parent method for other transformations
        return super().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
        )