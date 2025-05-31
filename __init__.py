from . import load_phi
from . import load_phi_multimodal
from . import load_phi_vision
from . import run_phi
from . import run_phi_multimodal
from . import run_phi_vision


import folder_paths, os
folder_paths.add_model_folder_path("microsoft", os.path.join(folder_paths.models_dir, "microsoft"))

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "LoadPhi": load_phi.LoadPhi,
    "LoadPhiMultimodal": load_phi_multimodal.LoadPhiMultimodal,
    "LoadPhiVision": load_phi_vision.LoadPhiVision,
    "RunPhi": run_phi.RunPhi,
    "RunPhiMultimodal": run_phi_multimodal.RunPhiMultimodal,
    "RunPhiVision": run_phi_vision.RunPhiVision
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadPhi": "Load Phi",
    "LoadPhiMultimodal": "Load Phi Multimodal",
    "LoadPhiVision": "Load Phi Vision",
    "RunPhi": "Run Phi",
    "RunPhiMultimodal": "Run Phi Multimodal",
    "RunPhiVision": "Run Phi Vision"
}