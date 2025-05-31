import os
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

# Modules from ComfyUI
import folder_paths


class LoadPhiMultimodal:
    """Node to load Phi multimodal model."""

    # Node setup for ComfyUI
    CATEGORY = "phi"
    FUNCTION = "execute"
    OUTPUT_NODE = False
    RETURN_TYPES = ("phi_model", "phi_processor", "phi_config")

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "model": (
                    ["Phi-4-multimodal-instruct"],
                    {
                        "default": "Phi-4-multimodal-instruct",
                        "tooltip": "The name of the model to load."
                    }
                ),
            }
        }

    def execute(self, model):
        # Model files should be placed in ./ComfyUI/models/microsoft
        microsoft_folder = folder_paths.get_folder_paths("microsoft")[0]
        model_path = os.path.join(microsoft_folder, model)

        phi_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
            # if you do not use Ampere or later GPUs, change attention to "eager"
            _attn_implementation='flash_attention_2',
        ).cuda()

        # For best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
        phi_processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True
        )

        # Load generation config
        phi_config = GenerationConfig.from_pretrained(model_path)

        return (phi_model, phi_processor, phi_config)