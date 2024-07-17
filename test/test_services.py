from services.segmentationInfer import segmentation_infer
from PIL import Image

def test_segmentation_infer(image_path: str):
    image = Image.open(image_path)
    res = segmentation_infer(image)
    assert isinstance(res, bytes)
    