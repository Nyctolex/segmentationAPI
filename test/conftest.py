import pytest
import os
from fastapi.testclient import TestClient
from main import app
import urllib
from models.load_model import load_torch_model
from torch.nn import Module
from typing import Callable

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def image_path() -> str:
    """Getting the path to a testing image. If the image is not in the system,
    the function would try to download a sample image.

    Raises:
        FileNotFoundError: If the system couldnt find and download an image.

    Returns:
        str: The path of the image
    """
    filename = './test/deeplab1.png'
    if not os.path.exists(filename):
        url = "https://github.com/pytorch/hub/raw/master/images/deeplab1.png"
        try: urllib.URLopener().retrieve(url, filename)
        except: urllib.request.urlretrieve(url, filename)
        if not os.path.exists(filename):
            raise FileNotFoundError('No testing image file was found')
    return filename


@pytest.fixture
def torch_model() -> Module:
    """Loading the pytorch model for testing"""
    model, _ = load_torch_model()
    return model

@pytest.fixture
def torch_preprocessor() -> Callable:
    """Loading the preprocessing function of the pytorch model"""
    _, preprocess = load_torch_model()
    return preprocess


