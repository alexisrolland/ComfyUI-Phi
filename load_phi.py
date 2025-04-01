import numpy as np
import os
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import folder_paths


class LoadPhi:
    """Node to load Phi model."""

    # Node setup for ComfyUI
    CATEGORY = "phi"
    FUNCTION = "execute"
    OUTPUT_NODE = False
    RETURN_TYPES = ("phi_model", "phi_tokenizer")

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "model": (
                    ["Phi-3.5-mini-instruct"],
                    {
                        "default": "Phi-3.5-mini-instruct",
                        "tooltip": "The name of the model to load."
                    }
                ),
            }
        }

    def execute(self, model):
        # Model files should be placed in ./ComfyUI/models/microsoft
        model = os.path.join(folder_paths.models_dir, "microsoft", model)

        phi_model = AutoModelForCausalLM.from_pretrained(
            model,
            local_files_only=True,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True
        )

        phi_tokenizer = AutoTokenizer.from_pretrained(
            model,
            local_files_only=True,
            trust_remote_code=True
        )

        return (phi_model, phi_tokenizer)