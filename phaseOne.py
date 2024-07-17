

from models.utils import visulize_diff
from models.utils import load_dummy_image
from utils.util import timeit

import torch
import argparse
import numpy as np
from models.utils import torch_to_onnx
from models.load_model import load_torch_model
from onnxruntime.capi.onnxruntime_inference_collection import InferenceSession
from models.utils import MSE
from loguru import logger
from PIL import Image


def main():
    # import an example image
    logger.debug('Importing test image')
    input_image = load_dummy_image()
    #load torch model
    logger.debug('Loading torch model')
    torch_model, torch_preprocessor = load_torch_model()

    input_image = input_image.resize((649, 520), Image.BILINEAR)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("pathh", type=str,  nargs='?', default=None, help="Path to the image to import for this demo")
    args = parser.parse_args()
    main()