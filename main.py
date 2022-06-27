from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
from models.resnet import run_inference
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthcheck")
async def root():
    return {"message": "ok"}

@app.post("/inference/image")
async def recieve_file(file: UploadFile):
    results = run_inference(file.file)
    return {"inference": results}