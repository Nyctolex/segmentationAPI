from fastapi import APIRouter, UploadFile, Depends, HTTPException, File
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from services.segmentationInfer import segmentation_infer
from io import BytesIO
from PIL import Image, UnidentifiedImageError
class ImageProcessRequest(BaseModel):
    file: UploadFile

router = APIRouter(
    prefix="/api/infer",
    tags=["Images"],
    responses={404: {"description": "Not found"}},
)




@router.post("/")
async def process_image_endpoint(image: UploadFile = File(...)):
    # TODO: handle errors
    allowed_types = ["image/jpeg", "image/png"]
    # validate content type
    if image.content_type not in allowed_types:
        raise HTTPException(status_code=415, detail="Unsupported Media Type")
    content = await image.read()
    try:
       image = Image.open(BytesIO(content))
    except UnidentifiedImageError:
        raise  HTTPException(status_code=500, detail="Internal Server Error: Failed to process the image.")
    
    processed_image = segmentation_infer(image)
    return StreamingResponse(BytesIO(processed_image), media_type="image/jpeg")

# @router.post("/")
# async def infer_image(
#     image_file: UploadFile = File(...),
# ):

#     not_found = HTTPException(status_code=404, detail=f"Survey with name {survey_name} was not found")
#     permission_denied = HTTPException(status_code=403, detail="You don't have permission to access this resource")
#     try:
#         survey_in_db = get_survey(survey_name=survey_name)
#         if not current_user.user_type == UserType.ADMIN and not validate_survey_edit_access(survey_in_db, current_user):
#             raise permission_denied
#         image_type = filename.split('.')[-1]

#         image = await image_file.read()
#         save_logo(survey_name, image, image_type)

#     except DoesNotExist:
#         raise not_found
#     return JSONResponse(status_code=200, content={"message": "Image updated successfully"})