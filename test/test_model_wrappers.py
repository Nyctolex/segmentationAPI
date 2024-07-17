
from model_wrappers import ImageToVector, torch_to_onnx
from PIL.Image import Image as PILImage
import unittest
import pytest
import torch
import numpy as np
from typing import Callable
from onnxruntime.capi.onnxruntime_inference_collection import InferenceSession

class TestImageToVector(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def prepare_fixture(self, image_path: str):
        self.image_path = image_path

    def test_load_image(self):
        image  = ImageToVector.load_image(self.image_path)
        self.assertTrue(isinstance(image, PILImage))

    def test_to_tensor(self):
        # test from path
        tensor = ImageToVector.to_tensor(self.image_path)
        self.assertTrue(isinstance(tensor, torch.Tensor))

        # From PIL
        image = ImageToVector.load_image(self.image_path)
        tensor = ImageToVector.to_tensor(image)
        self.assertTrue(isinstance(tensor, torch.Tensor))

        # From numpy
        image = np.array(image)
        tensor = ImageToVector.to_tensor(image)
        self.assertTrue(isinstance(tensor, torch.Tensor))

        # from Tensor
        image = torch.Tensor(image)
        tensor = ImageToVector.to_tensor(image)
        self.assertTrue(isinstance(tensor, torch.Tensor))

    def test_to_numpy(self):
        # test from path
        image = ImageToVector.to_numpy(self.image_path)
        self.assertTrue(isinstance(image, np.ndarray))

        # From PIL
        image = ImageToVector.load_image(self.image_path)
        image = ImageToVector.to_numpy(image)
        self.assertTrue(isinstance(image, np.ndarray))

        # from numpy
        image = np.array(image)
        image = ImageToVector.to_numpy(image)
        self.assertTrue(isinstance(image, np.ndarray))

        # From torch
        image = torch.Tensor(image)
        image = ImageToVector.to_numpy(image)
        self.assertTrue(isinstance(image, np.ndarray))



def test_torch_to_onnx(torch_model: torch.nn.Module, torch_preprocessor: Callable, image_path: str):
    dummpy_batch: torch.Tensor = ImageToVector.to_tensor(image_path).unsqueeze(0)
    dummpy_batch = torch_preprocessor(dummpy_batch)
    onnx_model: InferenceSession = torch_to_onnx(torch_model, dummpy_batch)
    assert isinstance(onnx_model, InferenceSession)
    




