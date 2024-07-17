from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights
from torchvision.models.segmentation.deeplabv3 import DeepLabV3 as DeepLabV3Type
from typing import Callable


def load_torch_model() -> tuple[DeepLabV3Type, Callable]:
    """Loads the DeepLabV3 model from the torch vision libary

    Returns:
        tuple[DeepLabV3Type, Callable]: A tuple of the model and a preprocess.
    """
    weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    preprocess = weights.transforms()
    model = deeplabv3_mobilenet_v3_large(weights=weights)
    model.eval()
    return model, preprocess