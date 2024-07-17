from PIL import Image
from io import BytesIO
from models.load_model import load_torch_model
from models.model_wrappers import SegmentationModelAI
from models.utils import segment_prediction_to_image, vector_to_pil
model, preprocessor = load_torch_model()
model = SegmentationModelAI(model, preprocessor)


def process_image(image: Image.Image) -> Image.Image:
    # Example process: Convert image to grayscale
    return image.convert('L')  # 'L' mode for grayscale

def segmentation_infer(image) -> bytes:
    processed_image = model(image)
    processed_image = segment_prediction_to_image(processed_image)
    processed_image = vector_to_pil(processed_image)
    

    buffered = BytesIO()
    processed_image.save(buffered, format="JPEG")
    return buffered.getvalue()