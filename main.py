
import uvicorn
from fastapi import FastAPI
from api import segmentationInfrence
from fastapi.middleware.cors import CORSMiddleware
origins = [
    "*"
]

app = FastAPI()
app.include_router(segmentationInfrence.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TODO:  pydentic , docstring, another model

if __name__ == "__main__":
    uvicorn.run(app)