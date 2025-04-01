import numpy as np
from PIL import Image


class RunPhiMultimodal:
    """Node to run Phi multimodal model."""

    # Node setup for ComfyUI
    CATEGORY = "phi"
    FUNCTION = "execute"
    OUTPUT_NODE = False
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("prompt", "response",)

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "phi_model": ("phi_model",),
                "phi_processor": ("phi_processor",),
                "phi_config": ("phi_config",),
                "image": ("IMAGE",),
                "instruction": ("STRING", {
                    "default": "Describe this image",
                    "multiline": True
                }),
                "max_new_tokens": ("INT", {
                    "default": 1000,
                    "min": 1
                }),
            }
        }

    def tensor2pil(self, image):
        batch_count = image.size(0) if len(image.shape) > 3 else 1
        if batch_count > 1:
            out = []
            for i in range(batch_count):
                out.extend(self.tensor2pil(image[i]))
            return out
        return [Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))]

    def execute(self, phi_model, phi_processor, phi_config, image, instruction,  max_new_tokens):
        # Define prompt structure
        user_prompt = '<|user|>'
        image_prompt = ''
        assistant_prompt = '<|assistant|>'
        prompt_suffix = '<|end|>'
        start_index = 1

        # Prepare images in the prompt
        images = self.tensor2pil(image) # Convert tensor to PIL image
        for index, value in enumerate(images, start=start_index):
            image_prompt += f"<|image_{index}|>"

        prompt = f'{user_prompt}{image_prompt}{instruction}{prompt_suffix}{assistant_prompt}'
        inputs = phi_processor(
            text=prompt,
            images=images,
            return_tensors='pt'
        ).to('cuda:0')
        
        # Generate response
        generate_ids = phi_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            generation_config=phi_config,
            num_logits_to_keep=1
        )
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = phi_processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return (prompt, response,)