from PIL import Image
from io import BytesIO
from models.load_model import load_torch_model
from models.model_wrappers import SegmentationModelAI, ImageType 
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
preprocessor = lambda data: resize(data.astype(np.float32),(1, 3, 520, 649))
model = SegmentationModelAI(model, preprocessor)


def segmentation_infer(image: Image.Image) -> bytes:
    image = image.convert("RGB")
    original_size = image.size
    prediction = model(image)
    processed_image = segment_prediction_to_image(prediction).resize(original_size)
    

    buffered = BytesIO()
    processed_image.save(buffered, format="JPEG")
    return buffered.getvalue()