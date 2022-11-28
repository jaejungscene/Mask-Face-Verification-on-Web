import os
import torch
from .iresnet import iresnet18
def get_model():
    base_dir = "C:/Users/gmk_0/source/repos/pythonProject/IT2/Computer-Vision-Project"
    weights = torch.load(os.path.join(base_dir,"weights/backbone-r18.pth"), map_location=torch.device("cpu"))
    model = iresnet18()
    model.load_state_dict(weights)
    print("params: {:,}".format(sum([p.data.nelement() for p in model.parameters()])))
    return model 