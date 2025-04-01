from transformers import pipeline


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
            }
        }

    def execute(self, phi_model, phi_tokenizer, system_message, instruction, return_full_text, do_sample, temperature, max_new_tokens):
        # Prepare messages
        messages = [ 
            {"role": "system", "content": system_message},
            {"role": "user", "content": instruction}
        ] 

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

        return (response,)