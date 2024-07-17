import matplotlib.pyplot as plt
from numpy import ndarray
import onnxruntime as rt
from onnxruntime.capi.onnxruntime_inference_collection import InferenceSession
from torch.nn import Module
from uuid import uuid4
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torch import Tensor, onnx
from PIL import Image
import numpy as np
from PIL.Image import Image as PILImage
from typing import Optional
import urllib
import os


def vector_to_pil(image: Tensor | PILImage | np.ndarray) -> PILImage:
    if isinstance(image, PILImage):
        return image
    elif isinstance(image, np.ndarray):
        return Image.fromarray(image.astype(np.uint8))
    elif isinstance(image, Tensor):
        return to_pil_image(image)
    raise TypeError('Image is of unsupported type')


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


def segment_prediction_to_image(logits: np.array, outputsize: tuple[int, int] = None) -> PILImage:
    """Creates an image from a segment model's prediction. The function return a numpy image where each class is colored
    with a different color.

    Args:
        logits (np.array): The logit prediction of the model (unbatched)
        outputsize (tuple[int, int], optional): The output's image size. Defaults to None.

    Returns:
        np.array: The color image
    """
    output_predictions = logits.argmax(0)
    n_classes = logits.shape[0]
    # create the color palette according to the number of classes
    palette = np.array([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = np.array([i for i in range(n_classes)])[:, None] * palette
    colors = (colors % 255).astype("uint8")
    colored_image = Image.fromarray(output_predictions.astype(np.uint8))
    if outputsize:
        colored_image = colored_image.resize(outputsize)
    colored_image.putpalette(colors)
    return colored_image.convert('RGB')


def MSE(vector1: np.ndarray, vector2: np.ndarray) -> float:
    assert len(vector1.shape) == len(vector2.shape)
    assert all([s[0] == s[1] for s in zip(vector1.shape, vector2.shape) ])
    return ((vector1 - vector2)**2).mean()


def get_compair_image(img1: np.array, img2: np.array) -> np.array:
    """Gets two images and return an image that has white pixels on places where the two images mismatch.

    Args:
        img1 (np.array): The first image
        img2 (np.array): The second image

    Returns:
        np.array: The difference image
    """
    mask = np.any(img1 != img2, axis=0)
    # Create an output array for visualization
    visualization_array = np.zeros_like(img1[0], dtype=np.uint8)  # Initialize as black image
    visualization_array[mask] = 255  # Set differing pixels to white (255)
    return visualization_array


def visulize_diff(source_image: np.ndarray, torch_pred: np.array, onnx_prediction: np.array, text: Optional[str] = None):
    """Visualizing the difference between the results of the two segmentation models

    Args:
        source_image (np.ndarray): The original image used for prediction
        torch_pred (np.array): The prediction of the torch model (logits, unbachted)
        onnx_prediction (np.array): The prediction of the onnx model (logits, unbachted)
        text (Optional[str], optional): Additional text that would be added to the title. Defaults to None.
    """

    fig, axes = plt.subplots(2, 2)
    axes[0, 0].imshow(segment_prediction_to_image(torch_pred))
    axes[0,0].set_title('Torch result')

    axes[0, 1].imshow(segment_prediction_to_image(onnx_prediction))
    axes[0,1].set_title('Onnx result')

    diff_img = get_compair_image(torch_pred, onnx_prediction)
    axes[1, 0].imshow(diff_img, cmap='gray')
    axes[1,0].set_title('Difference (white)')

    axes[1, 1].imshow(source_image)
    axes[1,1].set_title('Source image')

    title = f'Pytorch pred vs Onnx pred'
    if text:
        title  = f'{title}\n{text}'
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def load_dummy_image() -> Image.Image:
    url, filename = ("https://github.com/pytorch/hub/raw/master/images/deeplab1.png", "deeplab1.png")
    try: urllib.URLopener().retrieve(url, filename)
    except: urllib.request.urlretrieve(url, filename)
    input_image = Image.open(filename)
    input_image = input_image.convert("RGB")
    return input_image