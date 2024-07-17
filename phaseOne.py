

from utils.util import timeit

import torch


import argparse
import numpy as np
import urllib
from PIL import Image
from model_wrappers import torch_to_onnx
from test.load_test_model import load_torch_model
from onnxruntime.capi.onnxruntime_inference_collection import InferenceSession
from utils.losses import MSE
import matplotlib.pyplot as plt
from typing import Optional
from loguru import logger


def segment_prediction_to_image(logits: np.array, outputsize: tuple[int, int] = None) -> np.array:
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
    return np.array(colored_image)


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







def main():
    # import an example image
    logger.debug('Importing test image')
    input_image = load_dummy_image()
    #load torch model
    logger.debug('Loading torch model')
    torch_model, torch_preprocessor = load_torch_model()
    torch_image = torch_preprocessor(input_image).unsqueeze(0)
    np_image = torch_image.detach().cpu().numpy()

    #getting the onnx version of the torch model
    onnx_model = torch_to_onnx(torch_model, torch_image)

    #Compair timings
    onnx_input = ([n.name for n in onnx_model.get_outputs()], {onnx_model.get_inputs()[0].name: np_image})
    logger.debug('Testing onnx model runtime')
    onnx_time = timeit(onnx_model.run, onnx_input, tests_num=20)
    logger.debug('Testing torch model  runtime')
    with torch.no_grad():
        torch_time = timeit(torch_model, (torch_image,), tests_num=20)


    # Get the prediction of each model
    with torch.no_grad():
        torch_res: torch.Tensor = torch_model(torch_image)['out']
    torch_res: np.ndarray = torch_res.detach().cpu().numpy()[0]
    onnx_res: np.ndarray = onnx_model.run(*onnx_input)[0][0]

    # get l2 loss
    loss = MSE(onnx_res, torch_res)

    # get the source image as (H,W,3)
    image = np.transpose(np_image[0], (1, 2, 0))

    meta_data = f'Torch runtime: {torch_time:.2f} secs'
    meta_data += f'\nOnnx runtime: {onnx_time:.2f} secs'
    meta_data += f'\nL2 loss between predictions: {loss:.5f}'
    visulize_diff(image, torch_res, onnx_res, meta_data)






if __name__ == '__main__':
    main()
    # #TODO
    # parser = argparse.ArgumentParser(description = 'Say hello')
    # parser.add_argument('name', help='your name, enter it')
    # args = parser.parse_args()