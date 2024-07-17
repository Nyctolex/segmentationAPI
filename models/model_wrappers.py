from abc import ABC, abstractmethod
from torchvision.models.segmentation.deeplabv3 import DeepLabV3 as DeepLabV3Type
from torch import Tensor
import torch
from torch.nn import Module

from typing import Optional, Callable
from onnxruntime.capi.onnxruntime_inference_collection import InferenceSession
import numpy as np
from PIL.Image import Image as PILImage
import numpy as np

from models.utils import ImageToVector


type ImageType  = str | Tensor | PILImage | np.array


class ModelWrapper(ABC):
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
        return self.__preprocessor
    
    @preprocessor.setter
    def preprocessor(self, new_preprocessor):
        if new_preprocessor is not None and not callable(new_preprocessor):
            raise TypeError('The preprocessor should be callable')
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
    

class TorchSegmentationWrapper(ModelWrapper):
    def __init__(self, model: Module, preprocessor: Optional[Callable] = None):
        super().__init__(model, preprocessor)
        self.model.eval()

    @property
    def supported_model_types(self):
        return (Module,)

    def predict_batch(self, images: Tensor) -> Tensor:
        """Predicts the output of the model for a batch of images.

        Args:
            images (Tensor): A tensor of images of shape [N, 3, H, W]

        Returns:
            Tensor: the model's prediction
        """
        
        if self.preprocessor:
            images = self.preprocessor(images)
        with torch.no_grad():
            return self.model(images)['out']
    
    def predict_single(self, image: ImageType) -> Tensor | dict:
        """Predicts the output of the model for a single image.

        Args:
            image (Tensor): The image tensor of shape [3, H, W]

        Returns:
            Tensor: The model prediction (logits)
        """
        image: Tensor = ImageToVector.to_tensor(image)
        assert len(image.shape) == 3, 'Missmatching tensor dimensions. The image should be of shape [3, H, W]'
        assert image.shape[0] == 3, f'Missmatching tensor dimensions. The image should have 3 collor images but found {image.shape[1]}'
        return self.predict_batch(image.unsqueeze(0)).squeeze(0)


class OnnxSegmentationWrapper(ModelWrapper):

    def __init__(self, model: InferenceSession, preprocessor: Optional[Callable] = None):
        super().__init__(model, preprocessor)
        self.input_names = [x.name for x in self.model.get_inputs()]
        self.output_names = [x.name for x in self.model.get_outputs()]


    @property
    def supported_model_types(self):
        return (InferenceSession,)

    def predict_batch(self, images: np.array, use_preprocessor = True) -> np.array:
        """Prediction of a batch

        Args:
            images (np.array): batch of images in shape [N, 3, H, W]
            use_preprocessor (bool): should it use the preprocessor

        Returns:
            np.array: the result of the model
        """
        if use_preprocessor and self.preprocessor:
            images = self.preprocessor(images)
        model_input = {self.model.get_inputs()[0].name: images}
        return self.model.run(self.output_names, model_input)[0]
    
    @staticmethod
    def permute_image(image: np.array) -> np.array:
        """
        Permute image of [H, W, 3] to [3, H, W] if needed
        """
        if image.shape[0] != 3 and image.shape[-1] == 3:
            image = np.transpose(image, (2, 1, 0))
        return image
    
    def predict_single(self, image: ImageType, use_preprocessor=False) -> np.array:
        """Prediction of a single data point (unbatched)

        Args:
            image (ImageType): The input image (unbatched) of shape [3, H, W]

        Returns:
            np.array: The prediction of the model
        """
        if self.preprocessor:
            image = self.preprocessor(image)
        image = ImageToVector.to_numpy(image)
        # Expand dimmentions
        image = image[np.newaxis, :]
        return self.predict_batch(image, use_preprocessor=False)[0]



class SegmentationModelAI:
    def __init__(self, model, preprocessor: Optional[Callable] = None):
        if isinstance(model, Module):
            self.model = TorchSegmentationWrapper(model, preprocessor)
        elif isinstance(model, InferenceSession):
            self.model = OnnxSegmentationWrapper(model, preprocessor)
        else:
            raise TypeError(f'Given model type ({type(model)}) is not supported. The supported types are: {Module}, {InferenceSession}')

    def __call__(self, image: ImageType) -> np.ndarray:
        res = self.model.predict_single(image)
        if isinstance(res, Tensor):
            res = res.detach().cpu().numpy()
        return res