
import uvicorn
from fastapi import FastAPI, Response, Request
from api import segmentationInfrence
app = FastAPI()
app.include_router(segmentationInfrence.router)

# TODO: README, pydentic, requirments, middleware

if __name__ == "__main__":
    # uvicorn main:app --reload    
    uvicorn.run(app)