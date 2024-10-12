import numpy as np
import os

from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, pipeline

import folder_paths

class LoadPhi:
    """Node to download and load Phi model."""

    # Node setup for ComfyUI
    CATEGORY = "Phi"
    FUNCTION = "execute"
    OUTPUT_NODE = False
    RETURN_TYPES = ("PHI_MODEL", "PHI_TOKENIZER")

    def __init__(self):
        # Set models path to ./ComfyUI/models/microsoft
        self.model_path = os.path.join(folder_paths.models_dir, "microsoft")

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "model": (
                    [
                        "microsoft/Phi-3.5-mini-instruct",
                        #"microsoft/Phi-3.5-MoE-instruct"
                    ],
                    {"default": "microsoft/Phi-3.5-mini-instruct"}
                ),
            }
        }

    def execute(self, model):
        PHI_MODEL = AutoModelForCausalLM.from_pretrained(
            model,
            cache_dir=self.model_path,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True
        )

        PHI_TOKENIZER = AutoTokenizer.from_pretrained(
            model,
            cache_dir=self.model_path
        )

        return (PHI_MODEL, PHI_TOKENIZER)


class LoadPhiVision:
    """Node to download and load Phi model with vision."""

    # Node setup for ComfyUI
    CATEGORY = "Phi"
    FUNCTION = "execute"
    OUTPUT_NODE = False
    RETURN_TYPES = ("PHI_MODEL", "PHI_PROCESSOR")

    def __init__(self):
        # Set models path to ./ComfyUI/models/microsoft
        self.model_path = os.path.join(folder_paths.models_dir, "microsoft")

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "model": (
                    ["microsoft/Phi-3.5-vision-instruct"],
                    {"default": "microsoft/Phi-3.5-vision-instruct"}
                ),
            }
        }

    def execute(self, model):
        # Set _attn_implementation='eager' if you don't have flash_attn installed
        PHI_MODEL = AutoModelForCausalLM.from_pretrained(
            model,
            cache_dir=self.model_path,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
            #_attn_implementation="flash_attention_2",
            _attn_implementation="eager"
        )

        # For best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
        PHI_PROCESSOR = AutoProcessor.from_pretrained(
            model,
            cache_dir=self.model_path,
            trust_remote_code=True,
            num_crops=16
        )

        return (PHI_MODEL, PHI_PROCESSOR)


class RunPhi:
    """Node to run Phi model."""

    # Node setup for ComfyUI
    CATEGORY = "Phi"
    FUNCTION = "execute"
    OUTPUT_NODE = False
    RETURN_TYPES = ("STRING",)

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "PHI_MODEL": ("PHI_MODEL",),
                "PHI_TOKENIZER": ("PHI_TOKENIZER",),
                "system_message": ("STRING", {
                    "default": "You are a helpful AI assistant.",
                    "multiline": True
                }),
                "instruction": ("STRING", {
                    "default": "What is the answer to life the universe and everything",
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

    def execute(self, PHI_MODEL, PHI_TOKENIZER, system_message, instruction, return_full_text, do_sample, temperature, max_new_tokens):
        # Prepare messages
        messages = [ 
            {"role": "system", "content": system_message},
            {"role": "user", "content": instruction}
        ] 

        # Build pipeline
        pipe = pipeline("text-generation", model=PHI_MODEL, tokenizer=PHI_TOKENIZER)
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


class RunPhiVision:
    """Node to run Phi model with vision."""

    # Node setup for ComfyUI
    CATEGORY = "Phi"
    FUNCTION = "execute"
    OUTPUT_NODE = False
    RETURN_TYPES = ("STRING",)

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "PHI_MODEL": ("PHI_MODEL",),
                "PHI_PROCESSOR": ("PHI_PROCESSOR",),
                "IMAGE": ("IMAGE",),
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

    def execute(self, PHI_MODEL, PHI_PROCESSOR, IMAGE, instruction, do_sample, temperature, max_new_tokens):
        # Convert tensor to PIL image
        images = self.tensor2pil(IMAGE)

        # Prepare images placeholders in the prompt
        placeholder = ''
        for index, value in enumerate(images, start=1):
            placeholder += f"<|image_{index}|>\n"

        # Prepare prompt
        messages = [{"role": "user", "content": placeholder + instruction}]
        prompt = PHI_PROCESSOR.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        # Prepare generation arguments
        inputs = PHI_PROCESSOR(prompt, images, return_tensors="pt").to("cuda:0")
        generate_args = {}
        if do_sample:
            generate_args["do_sample"] = do_sample
            generate_args["temperature"] = temperature
        else:
            generate_args["do_sample"] = do_sample

        # Generate
        generate_ids = PHI_MODEL.generate(
            **inputs, 
            eos_token_id=PHI_PROCESSOR.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            **generate_args
        )

        # Remove input tokens 
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = PHI_PROCESSOR.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]

        return (response,)