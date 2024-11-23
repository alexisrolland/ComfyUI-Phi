# ComfyUI Phi

Custom ComfyUI nodes to run Microsoft's Phi models. Supported versions:

- [microsoft/Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)
- [microsoft/Phi-3.5-vision-instruct](https://huggingface.co/microsoft/Phi-3.5-vision-instruct)
- **To Be Done**: [microsoft/Phi-3.5-MoE-instruct](https://huggingface.co/microsoft/Phi-3.5-MoE-instruct)

When running the nodes for the first time, they will download the model files from Hugging Face hub and will place them in the folder `.\ComfyUI\models\microsoft`.

## Getting started

Go to the ComfyUI folder `.\ComfyUI\custom_nodes`, clone this repository and install Python dependencies:

```sh
# Clone repo
git clone https://github.com/alexisrolland/ComfyUI-Phi.git

# Install dependencies
..\..\python_embeded\python.exe -s -m pip install -r .\ComfyUI-Phi\requirements.txt
```

## Updates

* `2.0.0`: This major version introduces new inputs to provide a pair of image and response examples to the node Run Phi Vision.

## Example

Drag and drop the image in ComfyUI to reload the workflow.

![Example](workflow.png)