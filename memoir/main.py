import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "StyleShot"))
import io
import torch
import cv2
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from PIL import Image
from StyleShot.annotator.hed import SOFT_HEDdetector
from StyleShot.annotator.lineart import LineartDetector
from diffusers import UNet2DConditionModel, ControlNetModel
from transformers import CLIPVisionModelWithProjection
from huggingface_hub import snapshot_download
from StyleShot.ip_adapter import StyleShot, StyleContentStableDiffusionControlNetPipeline

app = FastAPI()

# === 모델 초기화 ===
base_model_path = "runwayml/stable-diffusion-v1-5"
transformer_block_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
device = "cuda" if torch.cuda.is_available() else "cpu"
styleshot_models = {}

def load_styleshot(preprocessor: str):
    if preprocessor in styleshot_models:
        return styleshot_models[preprocessor]

    if preprocessor == "Lineart":
        detector = LineartDetector()
        styleshot_model_path = "Gaojunyao/StyleShot_lineart"
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
    prompt: str = Form(...),
    preprocessor: str = Form(...),
    style: UploadFile = File(...),
    content: UploadFile = File(...)
):
    styleshot, detector = load_styleshot(preprocessor)

    # 스타일 이미지
    style_image = Image.open(io.BytesIO(await style.read())).convert("RGB")

    # 콘텐츠 이미지
    content_array = np.frombuffer(await content.read(), np.uint8)
    content_image = cv2.imdecode(content_array, cv2.IMREAD_COLOR)
    content_image = cv2.cvtColor(content_image, cv2.COLOR_BGR2RGB)
    processed_content = detector(content_image)
    processed_content = Image.fromarray(processed_content)

    # 추론
    result = styleshot.generate(style_image=style_image, prompt=[[prompt]], content_image=processed_content)
    output_image = result[0][0]

    # 결과 이미지 응답
    img_bytes = io.BytesIO()
    output_image.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    return StreamingResponse(img_bytes, media_type="image/png")
