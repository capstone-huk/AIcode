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

from dotenv import load_dotenv
load_dotenv()

def upload_to_s3(local_file_path: str, bucket: str, region: str, s3_key_prefix: str = "") -> str:
    s3 = boto3.client('s3')
    filename = f"{s3_key_prefix}{uuid.uuid4().hex}.png"
    try:
        s3.upload_file(local_file_path, bucket, filename, ExtraArgs={'ACL': 'public-read'})
        return f"https://{bucket}.s3.{region}.amazonaws.com/{filename}"
    except NoCredentialsError:
        raise RuntimeError("S3 credentials not found.")
    
app = FastAPI()

# === 모델 초기화 ===
base_model_path = "models/stable-diffusion-v1-5"
transformer_block_path = "models/CLIP-ViT-H-14-laion2B-s32B-b79K"

device = "cuda" if torch.cuda.is_available() else "cpu"
styleshot_models = {}

def load_styleshot(preprocessor: str):
    if preprocessor in styleshot_models:
        return styleshot_models[preprocessor]

    if preprocessor == "Lineart":
        detector = LineartDetector()
        styleshot_model_path = "models/StyleShot_lineart" 
    elif preprocessor == "Contour":
        detector = SOFT_HEDdetector()
        styleshot_model_path = "Gaojunyao/StyleShot"
    else:
        raise ValueError("Invalid preprocessor")

    # 모델 다운로드
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

# === API 엔드포인트 ===
@app.post("/generate/")
async def generate_image(
    preprocessor: str = Form(...),
    style_url: str = Form(...),
    content_url: str = Form(...),
    prompt: Optional[str] = Form(None)
):
    if prompt is None:
        prompt = "default prompt"  # 또는 그냥 빈 문자열로 처리

    styleshot, detector = load_styleshot(preprocessor)

    # 스타일 이미지 다운로드
    style_response = requests.get(style_url)
    style_response.raise_for_status()
    style_image = Image.open(io.BytesIO(style_response.content)).convert("RGB")

    # 콘텐츠 이미지 다운로드
    content_response = requests.get(content_url)
    content_response.raise_for_status()
    content_array = np.frombuffer(content_response.content, np.uint8)
    content_image = cv2.imdecode(content_array, cv2.IMREAD_COLOR)
    content_image = cv2.cvtColor(content_image, cv2.COLOR_BGR2RGB)
    processed_content = detector(content_image)
    processed_content = Image.fromarray(processed_content)

    # 추론
    result = styleshot.generate(style_image=style_image, content_image=processed_content)
    output_image = result[0][0]

    # 결과 반환
    img_bytes = io.BytesIO()
    output_image.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    output_path = f"/tmp/{uuid.uuid4().hex}.png"
    output_image.save(output_path)  # ✅ 파일 저장
    
    # 👉 S3 업로드
    s3_url = upload_to_s3(
        local_file_path=output_path,
        bucket="hukmemoirbucket",
        region="ap-northeast-2",
        s3_key_prefix="results/"
    )
    os.remove(output_path)  # 💡 서버 공간 정리
    
    return JSONResponse(content={"s3_url": s3_url})

@app.post("/test-upload/")
def test_s3_upload():
    # 1. 더미 이미지 생성
    dummy_image = Image.new("RGB", (256, 256), color="blue")
    temp_path = "test_result.png"
    dummy_image.save(temp_path)

    # 2. S3 업로드
    try:
        s3_url = upload_to_s3(
            local_file_path=temp_path,
            bucket="hukmemoirbucket",
            region="ap-northeast-2",
            s3_key_prefix="test/"
        )
        os.remove(temp_path)  # 업로드 후 파일 정리
        return {"s3_url": s3_url}
    except Exception as e:
        return {"error": str(e)}
