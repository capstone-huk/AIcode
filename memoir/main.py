import sys
import os
import requests
sys.path.append(os.path.join(os.path.dirname(__file__), "StyleShot"))
import io
import torch
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware  # ⬅️ CORS 모듈 추가
from PIL import Image
from StyleShot.annotator.hed import SOFT_HEDdetector
from StyleShot.annotator.lineart import LineartDetector
from diffusers import UNet2DConditionModel, ControlNetModel
from transformers import CLIPVisionModelWithProjection
from huggingface_hub import snapshot_download
from StyleShot.ip_adapter import StyleShot, StyleContentStableDiffusionControlNetPipeline

import boto3
from botocore.exceptions import NoCredentialsError
import uuid

from PIL import Image
from typing import Optional

import random

from dotenv import load_dotenv
load_dotenv()

# === S3 업로드 함수 ===
def upload_to_s3(local_file_path: str, bucket: str, region: str, s3_key_prefix: str = "") -> str:
    s3 = boto3.client('s3')
    filename = f"{s3_key_prefix}{uuid.uuid4().hex}.png"
    try:
        s3.upload_file(local_file_path, bucket, filename)
        return f"https://{bucket}.s3.{region}.amazonaws.com/{filename}"
    except NoCredentialsError:
        raise RuntimeError("S3 credentials not found.")

# === FastAPI 앱 초기화 ===
app = FastAPI()

# ✅ CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요 시 프론트엔드 도메인으로 제한 가능
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === 모델 및 경로 초기화 ===
base_model_path = "models/stable-diffusion-v1-5"
transformer_block_path = "models/laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
device = "cuda" if torch.cuda.is_available() else "cpu"
styleshot_models = {}

# === StyleShot 모델 로드 ===
def load_styleshot(preprocessor: str):
    if preprocessor in styleshot_models:
        return styleshot_models[preprocessor]

    if preprocessor == "Lineart":
        detector = LineartDetector()
        styleshot_model_path = "models/Gaojunyao/StyleShot_lineart"
    elif preprocessor == "Contour":
        detector = SOFT_HEDdetector()
        styleshot_model_path = "Gaojunyao/StyleShot"
    else:
        raise ValueError("Invalid preprocessor")

    if not os.path.isdir(styleshot_model_path):
        snapshot_download(styleshot_model_path, local_dir=styleshot_model_path)

    if not os.path.isdir(base_model_path):
        snapshot_download(base_model_path, local_dir=base_model_path)

    if not os.path.isdir(transformer_block_path):
        snapshot_download(transformer_block_path, local_dir=transformer_block_path)

    ip_ckpt = os.path.join(styleshot_model_path, "pretrained_weight/ip.bin")
    style_aware_encoder_path = os.path.join(styleshot_model_path, "pretrained_weight/style_aware_encoder.bin")

    unet = UNet2DConditionModel.from_pretrained(base_model_path, subfolder="unet")
    content_fusion_encoder = ControlNetModel.from_unet(unet)

    pipe = StyleContentStableDiffusionControlNetPipeline.from_pretrained(
        base_model_path,
        controlnet=content_fusion_encoder
    )

    styleshot = StyleShot(device, pipe, ip_ckpt, style_aware_encoder_path, transformer_block_path)
    styleshot_models[preprocessor] = (styleshot, detector)

    return styleshot, detector

# === 이미지 생성 API ===
@app.post("/generate/")
async def generate_image(
    preprocessor: str = Form(...),
    style_url: str = Form(...),
    prompt: Optional[str] = Form(None)
):
    if prompt is None:
        prompt = "default prompt"

    styleshot, detector = load_styleshot(preprocessor)

    try:
        style_response = requests.get(style_url)
        style_response.raise_for_status()
        style_image = Image.open(io.BytesIO(style_response.content)).convert("RGB")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid style image URL: {str(e)}"})

    content_dir = "content_image"
    try:
        content_files = [f for f in os.listdir(content_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not content_files:
            return JSONResponse(status_code=500, content={"error": "No content images found on server."})

        random_content_path = os.path.join(content_dir, random.choice(content_files))
        content_image = cv2.imread(random_content_path)
        content_image = cv2.cvtColor(content_image, cv2.COLOR_BGR2RGB)
        processed_content = detector(content_image)
        processed_content = Image.fromarray(processed_content)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Content image loading failed: {str(e)}"})

    try:
        result = styleshot.generate(style_image=style_image, content_image=processed_content)
        output_image = result[0][0]
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Image generation failed: {str(e)}"})

    output_path = f"/tmp/{uuid.uuid4().hex}.png"
    output_image.save(output_path)

    try:
        s3_url = upload_to_s3(
            local_file_path=output_path,
            bucket="hukmemoirbucket",
            region="ap-northeast-2",
            s3_key_prefix="results/"
        )
    finally:
        os.remove(output_path)

    return JSONResponse(content={"s3_url": s3_url})

# === 테스트용 S3 업로드 API ===
@app.post("/test-upload/")
def test_s3_upload():
    dummy_image = Image.new("RGB", (256, 256), color="blue")
    temp_path = "test_result.png"
    dummy_image.save(temp_path)

    try:
        s3_url = upload_to_s3(
            local_file_path=temp_path,
            bucket="hukmemoirbucket",
            region="ap-northeast-2",
            s3_key_prefix="test/"
        )
    finally:
        os.remove(temp_path)

    return JSONResponse(content={"s3_url"_
