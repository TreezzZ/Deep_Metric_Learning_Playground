import torch.nn as nn

from torchvision.models import resnet50, googlenet


def get_model(arch, embed_dim, is_frozen):
    if arch.lower() == "resnet50":
        model = resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, embed_dim)
    elif arch.lower() == "googlenet":
        model = googlenet(pretrained=True)
        model.avgpool = nn.AdaptiveMaxPool2d(1)
        model.dropout = nn.Identity()
        model.fc = nn.Linear(1024, embed_dim)
    else:
        raise ValueError(f"Can not find arch name {arch}")

    if is_frozen:
        for module in filter(lambda m: type(m) == nn.BatchNorm2d, model.modules()):
            module.eval()
            module.train = lambda _: None
    return model