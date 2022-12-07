import os
import torch
from .iresnet import iresnet18
def get_model(base_dir):
    weights = torch.load(os.path.join(base_dir,"weights/baseline-finetune.pth"), map_location=torch.device("cpu"))
    model = iresnet18()
    model.load_state_dict(weights)
    print("params: {:,}".format(sum([p.data.nelement() for p in model.parameters()])))
    return model 