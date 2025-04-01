import numpy as np
from PIL import Image


class RunPhiVision:
    """Node to run Phi model with vision."""

    # Node setup for ComfyUI
    CATEGORY = "phi"
    FUNCTION = "execute"
    OUTPUT_NODE = False
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("instruction", "response",)

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "phi_model": ("phi_model",),
                "phi_processor": ("phi_processor",),
                "image": ("IMAGE",),
                "instruction": ("STRING", {
                    "default": "Describe this image",
                    "multiline": True
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
            },
            "optional": {
                "image_example": ("IMAGE",),
                "response_example": ("STRING", {
                    "multiline": True
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

    def execute(self, phi_model, phi_processor, image, instruction, do_sample, temperature, max_new_tokens, image_example=None, response_example=''):
        # Convert tensor to PIL image
        images = self.tensor2pil(image)
        start_index = 1
        placeholder = ""
        additional_instruction = ""

        # Prepare data if example is provided
        example_image = []
        if (image_example is not None and len(response_example) > 0):
            example_image = [self.tensor2pil(image_example)[0]] # Enforce a single example
            start_index = 2
            additional_instruction = f"\nHere is an example of an image and its description.\nImage: <|image_1|>\nDescription: {response_example}\nCan you describe this new image?"

        # Prepare images placeholders in the prompt
        for index, value in enumerate(images, start=start_index):
            placeholder += f"\n<|image_{index}|>"

        # Prepare prompt
        messages = [{"role": "user", "content": instruction + additional_instruction + placeholder}]
        prompt = phi_processor.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        # Prepare generation arguments
        inputs = phi_processor(prompt, example_image + images, return_tensors="pt").to("cuda:0")
        generate_args = {}
        if do_sample:
            generate_args["do_sample"] = do_sample
            generate_args["temperature"] = temperature
        else:
            generate_args["do_sample"] = do_sample

        # Generate
        generate_ids = phi_model.generate(
            **inputs, 
            eos_token_id=phi_processor.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            **generate_args
        )

        # Remove input tokens 
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
        response = phi_processor.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]

        return (messages[0]["content"], response,)