from transformers import pipeline
import comfy.model_management as mm


class RunPhi:
    """Node to run Phi model."""

    # Node setup for ComfyUI
    CATEGORY = "phi"
    FUNCTION = "execute"
    OUTPUT_NODE = False
    RETURN_TYPES = ("STRING",)
    RETURN_NAME = ("text",)

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "phi_model": ("phi_model",),
                "phi_tokenizer": ("phi_tokenizer",),
                "system_message": ("STRING", {
                    "default": "You are an AI assistant that's helpful and efficient.",
                    "multiline": True
                }),
                "instruction": ("STRING", {
                    "default": "What is the answer to life the universe and everything. Give me just the answer. No bla bla...",
                    "multiline": True
                }),
                "return_full_text": ("BOOLEAN", {
                    "default": False
                }),
                "do_sample": ("BOOLEAN", {
                    "default": False
                }),
                "temperature": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.01,
                    "step": 0.01
                }),
                "max_new_tokens": ("INT", {
                    "default": 500,
                    "min": 1
                }),
                "offload_after_use": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": "If True, Phi will unload from VRAM to your offload device after running, (note that comfy-unload-model cannot unload Phi)"
                }),
            }
        }

    def execute(self, phi_model, phi_tokenizer, system_message, instruction, return_full_text, do_sample, temperature, max_new_tokens, offload_after_use):
        # Prepare messages
        messages = [ 
            {"role": "system", "content": system_message},
            {"role": "user", "content": instruction}
        ] 
        # make sure phi is sent to the device?
        if offload_after_use:
            
            device = mm.get_torch_device()
            print(f"sending Phi to {device}")
            phi_model.to(device)
            mm.soft_empty_cache()

        # Build pipeline
        pipe = pipeline("text-generation", model=phi_model, tokenizer=phi_tokenizer)
        generation_args = { 
            "return_full_text": return_full_text,
            "do_sample": do_sample,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens
        } 

        # Generate
        output = pipe(messages, **generation_args) 
        response = output[0]["generated_text"]

        # Convert dictionary to text if returning full text
        if return_full_text:
            response = str(response)

        # offload Phi
        if offload_after_use:
            offload_device = mm.unet_offload_device()
            print("Offloading Phi model...")
            phi_model.to(offload_device)
            mm.soft_empty_cache()

        return (response,)