import os
from transformers import AutoModelForCausalLM, AutoProcessor

# Modules from ComfyUI
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
            },
            "optional": {
                "dtype": (
                    ["auto", "float16", "bfloat16"],
                    {
                        "default": "auto",
                        "tooltip": "The model dtype. bfloat16 provides better speed but less precision."
                    }
                ),
                "attention": (
                    ["eager", "flash_attention_2"],
                    {
                        "default": "eager",
                        "tooltip": "Attention mecanism. Flash Attention should be faster."
                    }
                )
            }
        }

    def execute(self, model, dtype="auto", attention="eager"):
        # Model files should be placed in ./ComfyUI/models/microsoft
        microsoft_folder = folder_paths.get_folder_paths("microsoft")[0]
        model_path = os.path.join(microsoft_folder, model)

        phi_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            device_map="cuda",
            torch_dtype=dtype,
            trust_remote_code=True,
            _attn_implementation=attention
        )

        # For best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
        phi_processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            num_crops=16
        )

        return (phi_model, phi_processor)