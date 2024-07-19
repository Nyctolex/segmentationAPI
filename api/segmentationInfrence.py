from fastapi import APIRouter, UploadFile, Depends, HTTPException, File
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from services.segmentationInfer import segmentation_infer
from io import BytesIO
from PIL import Image, UnidentifiedImageError
from fastapi.responses import FileResponse

class ErrorResponseModel(BaseModel):
    detail: str

router = APIRouter(
    prefix="/api/infer",
    tags=["Images"],
    responses={404: {"description": "Not found"}},
)




@router.post("/", responses={
    415: {"model": ErrorResponseModel},
    500: {"model": ErrorResponseModel},
})
async def process_image_endpoint(file: UploadFile = File(...)):
    """
    An end point that gets an image and return an image with the visulized result of the segmentation model.
    """
    allowed_types = ["image/jpeg", "image/png"]
    # validate content type
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=415, detail="Unsupported Media Type")
    content = await file.read()
    try:
       image = Image.open(BytesIO(content))
    except UnidentifiedImageError:
        raise  HTTPException(status_code=500, detail="Internal Server Error: Failed to process the image.")
    
    processed_image = segmentation_infer(image)
    return StreamingResponse(BytesIO(processed_image), media_type="image/jpeg")