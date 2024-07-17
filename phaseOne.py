

from utils.util import timeit

import torch


import argparse
import numpy as np
import urllib
from PIL import Image
from model_wrappers import OnnxWrapper, torch_to_onnx


from torchvision.transforms.functional import pil_to_tensor





def compair_loss():
    pass

def visulize_diff():
    pass


def load_dummy_image() -> Image.Image:
    url, filename = ("https://github.com/pytorch/hub/raw/master/images/deeplab1.png", "deeplab1.png")
    try: urllib.URLopener().retrieve(url, filename)
    except: urllib.request.urlretrieve(url, filename)
    input_image = Image.open(filename)
    input_image = input_image.convert("RGB")
    return input_image

def main():
    # import an example image
    input_image = load_dummy_image()
    
    torch_model, torch_preprocessor = load_torch_model()
    torch_image = torch_preprocessor(input_image).unsqueeze(0)
    np_image = torch_image.detach().cpu().numpy()
    torch_model(torch_image)


    onnx_model = torch_to_onnx(torch_model, torch_image)

    #TODO: to device
    onnx_model_input = ([n.name for n in onnx_model.get_outputs()], {onnx_model.get_inputs()[0].name: np_image})
    onnx_model.run(*onnx_model_input)
    onnx_time = timeit(onnx_model.run, onnx_model_input, tests_num=20)
    with torch.no_grad():
        torch_time = timeit(torch_model, (torch_image,), tests_num=20)

    print(onnx_time, torch_time)






if __name__ == '__main__':
    main()
    # #TODO
    # parser = argparse.ArgumentParser(description = 'Say hello')
    # parser.add_argument('name', help='your name, enter it')
    # args = parser.parse_args()