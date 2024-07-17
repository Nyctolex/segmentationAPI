from PIL import Image
from io import BytesIO
from models.load_model import load_torch_model
from models.model_wrappers import SegmentationModelAI, ImageType
from models.utils import segment_prediction_to_image, vector_to_pil
model, preprocessor = load_torch_model()
model = SegmentationModelAI(model, preprocessor)


def segmentation_infer(image: Image.Image) -> bytes:
    image = image.convert("RGB")
    prediction = model(image)
    processed_image = segment_prediction_to_image(prediction)
    

    buffered = BytesIO()
    processed_image.save(buffered, format="JPEG")
    return buffered.getvalue()