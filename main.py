
import uvicorn
from fastapi import FastAPI, Response, Request
from api import segmentationInfrence
app = FastAPI()
app.include_router(segmentationInfrence.router)


if __name__ == "__main__":
    # uvicorn api.main:app --reload
    uvicorn.run(app)