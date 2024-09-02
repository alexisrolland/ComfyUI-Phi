from . import nodes

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    #"DownloadAndLoadPhi": nodes.DownloadAndLoadPhi,
    "DownloadAndLoadPhiVision": nodes.DownloadAndLoadPhiVision,
    #"RunPhi": nodes.RunPhi,
    "RunPhiVision": nodes.RunPhiVision
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    #"DownloadAndLoadPhi": "Download And Load Phi Model",
    "DownloadAndLoadPhiVision": "Download And Load Phi Model With Vision",
    #"RunPhi": "Run Phi Model",
    "RunPhiVision": "Run Phi Model With Vision"
}