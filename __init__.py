from . import nodes

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "LoadPhi": nodes.LoadPhi,
    "LoadPhiVision": nodes.LoadPhiVision,
    "RunPhi": nodes.RunPhi,
    "RunPhiVision": nodes.RunPhiVision
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadPhi": "Load Phi",
    "LoadPhiVision": "Load Phi Vision",
    "RunPhi": "Run Phi",
    "RunPhiVision": "Run Phi Vision"
}