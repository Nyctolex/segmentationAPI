from abc import ABC, abstractmethod
from torchvision.models.segmentation.deeplabv3 import DeepLabV3 as DeepLabV3Type
from torch import Tensor, no_grad
import torch
from torch.nn import Module
from typing import Optional, Callable
from onnxruntime.capi.onnxruntime_inference_collection import InferenceSession
import numpy as np
import os
from uuid import uuid4
import onnxruntime as rt 
from torch import onnx
from PIL.Image import Image as PILImage
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from numpy import ndarray
import numpy as np

type ImageType  = str | Tensor | PILImage


def torch_to_onnx(model: Module, dummy_batch: Tensor) -> InferenceSession:
    """Converting a torch module to an onnx module

    Args:
        model (Module): The torch module
        dummy_batch (Tensor): a valid input batch for the torch model

    Returns:
        InferenceSession: The onnx model
    """
    export_path  = f'temp_model_{uuid4()}.onnx'
    onnx.export(model, dummy_batch, export_path, export_params=True)
    onnx_model = rt.InferenceSession(export_path)
    os.remove(export_path)
    return onnx_model


class ImageToVector:
    
    @classmethod
    def load_image(cls, path: str) -> PILImage:
        input_image = Image.open(path)
        input_image = input_image.convert("RGB")
        return input_image

    @classmethod
    def to_tensor(cls, image: str | ndarray | PILImage | Tensor) -> Tensor:
        if isinstance(image, Tensor):
            return image
        if isinstance(image, str):
            image: PILImage = cls.load_image(image)
        if isinstance(image, PILImage):
            return pil_to_tensor(image)
        if isinstance(image, ndarray):
            return torch.from_numpy(image)
        raise TypeError(f'Type {type(image)} is not supported')

    @classmethod
    def to_numpy(cls, image:  str | ndarray | PILImage | Tensor) -> ndarray:
        if isinstance(image, ndarray):
            return image
        if isinstance(image, str):
            image: PILImage = cls.load_image(image)
        if isinstance(image, PILImage):
            return np.array(image)
        if isinstance(image, Tensor):
            return image.detach().cpu().numpy()
        raise TypeError(f'Type {type(image)} is not supported')
    


class ModelWrapper():
    def __init__(self, model, preprocessor: Optional[Callable] = None):
        self.__model = None
        self.model = model
        self.__preprocessor = None
        self.preprocessor = preprocessor

    @property
    @abstractmethod
    def supported_model_types(self):
        """Return a tuple of supported model types for the subclass."""
        raise NotImplementedError

    @property
    def model(self):
        return self.__model
    
    @property
    def preprocessor(self):
        assert self.__preprocessor, 'preprocessor is not defined'
        return self.__preprocessor
    
    @preprocessor.setter
    def preprocessor(self, new_preprocessor):
        self.__preprocessor = new_preprocessor
    
    @model.setter
    def model(self, new_model):
        print(new_model, type(new_model))
        if not self.validate_model_type(new_model):
            raise TypeError(f"Unsupported model type. Supported types are: {self.supported_model_types}")
        self.__model = new_model


    def validate_model_type(self, new_model) -> bool:
        return isinstance(new_model, self.supported_model_types)
    
    
    @abstractmethod
    def predict_single(self, *args, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def predict_batch(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplemented
    

# class TorchSegmentationWrapper(ModelWrapper):
#     @property
#     def supported_model_types(self):
#         return (Module,)

#     def predict_batch(self, args: tuple):
#         """Predicts the output of the model for a batch of images.

#         Args:
#             args (tuple): A tuple containing input data for the model

#         Returns:
#             Tensor: the model's prediction
#         """
#         if self.preprocessor:

#         pass
    
#     def predict_single(self, args: tuple) -> Tensor | dict:
#         """Predicts the output of the model for a single image.

#         Args:
#             image (Tensor): The image tensor of shape [3, H, W]

#         Returns:
#             Tensor: The model prediction (logits)
#         """

#         # assert len(image.shape) == 3, 'Missmatching tensor dimensions. The image should be of shape [3, H, W]'
#         # assert image.shape[0] == 3, f'Missmatching tensor dimensions. The image should have 3 collor images but found {image.shape[1]}'
#         return self.predict_batch(image.unsqueeze(0)).squeeze(0)



class OnnxWrapper(ModelWrapper):

    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)
        self.input_names = [x.name for x in self.model.get_inputs()]
        self.output_names = [x.name for x in self.model.get_outputs()]


    @property
    def supported_model_types(self):
        return (InferenceSession,)

    def predict_batch(self, *args) -> list:
        """Prediction of a batch
        Args:
            any amount of numpy args (batched)
        Returns:
            list: list of the model's prediction
        """
        model_input = {x[0].name: x[1] for x in zip(self.input_names, args)}
        return self.model.run(self.input_names, model_input)
    
    def predict_single(self, *args):
        """Prediction of a single data point (unbatched)
        Args:
            any amount of numpy args (unbatched)
        Returns:
            Tensor: list of the model's prediction
        """
        # Expand dimmentions
        args = (x[np.newaxis, :] for x in args)
        return self.predict_batch(*args)





class SegmentationModelAI():
    def __init__(self, model, preprocessor: Optional[Callable] = None):
        #TODO support torch and ONNX
        self.model = model
        self.model.eval()
        self.preprocessor = preprocessor

    def predict_batch(self, batch: Tensor) -> Tensor:
        return self.model(batch)

    def predict_single(self, tensor: Tensor) -> Tensor:
        return self.predict_batch(tensor.unsqueeze(0))

    def __call__(self, image: ImageType) -> Tensor:
        tensor  = ImageToTensor.convert(image)
        if self.preprocessor:
            tensor = self.preprocessor(tensor)
        return self.predict_single(tensor)