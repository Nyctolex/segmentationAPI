
from models.model_wrappers import TorchSegmentationWrapper, OnnxSegmentationWrapper, SegmentationModelAI
from PIL.Image import Image as PILImage
import unittest
import pytest
import torch
import numpy as np
from typing import Callable
from skimage.transform import resize
from onnxruntime.capi.onnxruntime_inference_collection import InferenceSession
from PIL.Image import Image
from models.utils import ImageToVector, torch_to_onnx


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

class TestTorchSegmentationWrapper(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def prepare_fixture(self, torch_model: torch.nn.Module, torch_preprocessor: Callable, image_path: str):
        self.model = torch_model
        self.preprocessor = torch_preprocessor
        self.image_path = image_path
    
    def test_supported_model_types(self):
        with self.assertRaises(TypeError):
            # Should raise an error
            TorchSegmentationWrapper(model='wrong model type')
        # Should run without error
        TorchSegmentationWrapper(model=self.model)

    def test_predict_single(self):
        wrapper = TorchSegmentationWrapper(model=self.model, preprocessor=self.preprocessor)
        res = wrapper.predict_single(self.image_path)
        assert isinstance(res, torch.Tensor)
        assert len(res.shape) == 3


class TestOnnxSegmentationWrapper(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def prepare_fixture(self, torch_model: torch.nn.Module, torch_preprocessor: Callable, image_path: str, example_image:Image):
        # Converting the torch to onnx model
        dummy_batch = ImageToVector.to_tensor(image_path)
        dummy_batch = torch_preprocessor(dummy_batch).unsqueeze(0)
        self.model = torch_to_onnx(torch_model, dummy_batch)
        self.image_path = image_path
        self.example_image = example_image
    
    def test_supported_model_types(self):
        with self.assertRaises(TypeError):
            # Should raise an error
            OnnxSegmentationWrapper(model='wrong model type')
        # Should run without error
        OnnxSegmentationWrapper(model=self.model)
    
    def test_predict_batch(self):
        wrapper = OnnxSegmentationWrapper(model=self.model)
        image = ImageToVector.to_numpy(self.example_image)
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = resize(image, (3, 520, 649))
        image = image[np.newaxis, :]
        res = wrapper.predict_batch(image)
        assert isinstance(res, np.ndarray)
        assert len(res.shape) == 4

    def test_predict_single(self):
        preprocessor = lambda data: resize(data.astype(np.float32),(3, 520, 649))
        wrapper = OnnxSegmentationWrapper(model=self.model, preprocessor = preprocessor)
        image = ImageToVector.to_numpy(self.example_image)
        image = np.transpose(image, (2, 0, 1))
        res = wrapper.predict_single(image)
        assert isinstance(res, np.ndarray)
        assert len(res.shape) == 3

class TestSegmentationModelAI(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def prepare_fixture(self, torch_model: torch.nn.Module, torch_preprocessor: Callable, image_path: str, example_image:Image):
        # Converting the torch to onnx model
        dummy_batch = ImageToVector.to_tensor(image_path)
        dummy_batch = torch_preprocessor(dummy_batch).unsqueeze(0)
        self.onnx_model = torch_to_onnx(torch_model, dummy_batch)
        self.torch_model = torch_model
        self.torch_preprocessor = torch_preprocessor
        self.onnx_preprocessor = None
        self.image_path = image_path
        self.example_image = example_image

    def test_call(self):
        # test unsupported models
        with self.assertRaises(TypeError):
            SegmentationModelAI('str', self.torch_preprocessor)
            SegmentationModelAI('str', 'str')
        # Testing using a torch model
        torch_version =  SegmentationModelAI(self.torch_model, self.torch_preprocessor)
        torch_res = torch_version(self.example_image)
        assert isinstance(torch_res, np.ndarray)

        # Testing using an onnx model
        preprocessor = lambda data: resize(np.array(data, dtype=np.float32),(3, 520, 649))
        onnx_version = SegmentationModelAI(self.onnx_model, preprocessor)
        onnx_res = onnx_version(self.example_image)
        assert isinstance(onnx_res, np.ndarray)