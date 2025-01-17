from PIL import Image
from io import BytesIO
from models.load_model import load_torch_model
from models.model_wrappers import SegmentationModelAI, ImageType, OnnxSegmentationWrapper, ImageToVector
from models.utils import ImageToVector, load_dummy_image, torch_to_onnx
from skimage.transform import resize
import numpy as np

from models.utils import segment_prediction_to_image, vector_to_pil
# Loading the model
model, preprocessor = load_torch_model()
dummy_batch = preprocessor(load_dummy_image()).unsqueeze(0)

# Converting it to an onnx model
model = torch_to_onnx(model, dummy_batch)
# setting up the wrapper
def preprocess(image):
    image = vector_to_pil(image)
    image = image.resize((649, 520), Image.BILINEAR)
    processed  = preprocessor(image)
    return processed.detach().cpu().numpy().astype(np.float32)
model = SegmentationModelAI(model, preprocess)


def segmentation_infer(image: Image.Image) -> bytes:
    """Preforms infer using a segmentation model of the given image, and returns a visualization of the result.

    Args:
        image (Image.Image): The image for infrence.

    Returns:
        bytes: An image, visualizing the prediction of the model.
    """
    image = image.convert("RGB")
    original_size = image.size
    prediction = model(image)
    processed_image = segment_prediction_to_image(prediction).resize(original_size)
    

    buffered = BytesIO()
    processed_image.save(buffered, format="JPEG")
    return buffered.getvalue()