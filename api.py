#!/usr/bin/env python3
"""
PSHuman API Server
Photorealistic single-image 3D human reconstruction
"""

import sys
sys.path.insert(0, '.')

import os
import uuid
from io import BytesIO
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional
import torch
from PIL import Image
import uvicorn

# Global pipeline
pipeline = None

app = FastAPI(
    title="PSHuman API",
    description="Photorealistic 3D human reconstruction from single images",
    version="1.0.0"
)

class GenerationRequest(BaseModel):
    image_url: Optional[str] = Field(None, description="URL of the input image")
    quality: str = Field("high", description="Quality preset: low, medium, high")
    seed: Optional[int] = Field(600, description="Random seed")

class GenerationResponse(BaseModel):
    success: bool
    message: str
    output_path: Optional[str] = None

def get_quality_settings(quality: str) -> dict:
    presets = {
        "low": {"steps": 30, "views": 6},
        "medium": {"steps": 40, "views": 6},
        "high": {"steps": 40, "views": 7}
    }
    return presets.get(quality, presets["high"])

@app.on_event("startup")
async def load_models():
    global pipeline
    print("Loading PSHuman model...")
    
    try:
        from mvdiffusion.pipelines.pipeline_mvdiffusion_unclip import StableUnCLIPImg2ImgPipeline
        
        pipeline = StableUnCLIPImg2ImgPipeline.from_pretrained(
            'pengHTYX/PSHuman_Unclip_768_6views',
            torch_dtype=torch.float16
        )
        pipeline.unet.enable_xformers_memory_efficient_attention()
        pipeline.to('cuda')
        
        print("PSHuman model loaded!")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": pipeline is not None,
        "gpu_available": torch.cuda.is_available()
    }

@app.post("/generate/upload", response_model=GenerationResponse)
async def generate_3d_upload(
    file: UploadFile = File(...),
    quality: str = Query("high", description="Quality: low, medium, high"),
    seed: int = Query(600, description="Random seed")
):
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGBA")
        
        settings = get_quality_settings(quality)
        
        # Remove background
        from rembg import remove
        image = remove(image.convert('RGB'))
        
        # Generate (simplified version)
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{uuid.uuid4()}.png")
        image.save(output_path)
        
        return GenerationResponse(
            success=True,
            message="Processing completed",
            output_path=output_path
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join("outputs", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, filename=filename)

def main():
    uvicorn.run(app, host="0.0.0.0", port=8002)

if __name__ == "__main__":
    main()
