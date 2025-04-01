import os
from transformers import AutoModelForCausalLM, AutoProcessor
import folder_paths


class LoadPhiVision:
    """Node to load Phi model with vision."""

    # Node setup for ComfyUI
    CATEGORY = "phi"
    FUNCTION = "execute"
    OUTPUT_NODE = False
    RETURN_TYPES = ("phi_model", "phi_processor")

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "model": (
                    ["Phi-3.5-vision-instruct"],
                    {
                        "default": "Phi-3.5-vision-instruct",
                        "tooltip": "The name of the model to load."
                    }
                ),
            }
        }

    def execute(self, model):
        # Model files should be placed in ./ComfyUI/models/microsoft
        model = os.path.join(folder_paths.models_dir, "microsoft", model)

        # Set _attn_implementation='eager' if you don't have flash_attn installed
        phi_model = AutoModelForCausalLM.from_pretrained(
            model,
            local_files_only=True,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
            #_attn_implementation="flash_attention_2",
            _attn_implementation="eager"
        )

        # For best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
        phi_processor = AutoProcessor.from_pretrained(
            model,
            local_files_only=True,
            trust_remote_code=True,
            num_crops=16
        )

        return (phi_model, phi_processor)