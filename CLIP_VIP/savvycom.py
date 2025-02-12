import torch
import torch.nn as nn

# Load the model
model_path = "./assets/pretrain_clipvip_base_32.pt"  
model = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)

if isinstance(model, dict) and "state_dict" in model:
    print("Loaded state_dict, please load into a model architecture before proceeding.")
else:
    print("Model loaded successfully.")
print(model)
