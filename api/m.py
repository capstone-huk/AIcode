import os
import uuid
import asyncio
import requests
import numpy as np
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.cluster import KMeans
from typing import List
import webcolors
from openai import OpenAI
import boto3

# ✅ 환경변수 로드
load_dotenv()

# ✅ OpenAI API Key로 클라이언트 생성 (최신 방식)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ S3 설정
s3 = boto3.client(
    "s3",
    region_name=os.getenv("AWS_S3_REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)
BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME")
REGION = os.getenv("AWS_S3_REGION")

# ✅ FastAPI 앱 및 CORS 설정
app = FastAPI()
origins = [
    "http://localhost:3000",  # 개발용
    "https://your-frontend.com",  # 실제 배포 주소
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 요청 모델 정의
class ImageURLs(BaseModel):
    image_urls: List[str]

# ✅ 색상 추출 및 이름 변환 함수들
CSS_COLOR_MAP = {
    "black": "#000000", "white": "#ffffff", "red": "#ff0000", "green": "#008000",
    "blue": "#0000ff", "yellow": "#ffff00", "cyan": "#00ffff", "magenta": "#ff00ff",
    "gray": "#808080", "orange": "#ffa500", "pink": "#ffc0cb", "purple": "#800080",
    "brown": "#a52a2a", "beige": "#f5f5dc", "olive": "#808000", "navy": "#000080",
    "teal": "#008080"
}

def extract_dominant_colors(img: Image.Image, num_colors=5):
    img = img.convert("RGB").resize((150, 150))
    img_data = np.array(img).reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_colors, random_state=42).fit(img_data)
    return [tuple(map(int, c)) for c in kmeans.cluster_centers_]

def closest_color(requested_color):
    min_distance = float("inf")
    closest_name = None
    for name, hex_val in CSS_COLOR_MAP.items():
        r_c, g_c, b_c = tuple(int(hex_val[i:i+2], 16) for i in (1, 3, 5))
        distance = sum((x - y) ** 2 for x, y in zip((r_c, g_c, b_c), requested_color))
        if distance < min_distance:
            min_distance = distance
            closest_name = name
    return closest_name

def rgb_to_nearest_name(rgb):
    try:
        return webcolors.rgb_to_name(rgb)
    except ValueError:
        return closest_color(rgb)

# ✅ 단일 이미지 처리 함수
async def process_single_image(image_url: str):
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))

        colors = extract_dominant_colors(image)
        color_names = [rgb_to_nearest_name(c) for c in colors]
        unique_colors = list(set(color_names))

        prompt = f"An abstract painting using these colors: {', '.join(unique_colors)}"

        # ✅ 최신 방식으로 이미지 생성
        response = client.images.generate(
            model="dall-e-2",
            prompt=prompt,
            n=1,
            size="512x512"
        )
        gen_url = response.data[0].url

        img_data = requests.get(gen_url).content
        filename = f"{uuid.uuid4()}.png"
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=filename,
            Body=img_data,
            ContentType="image/png",
            ACL="public-read"
        )

        return {
            "source_image": image_url,
            "prompt": prompt,
            "colors": unique_colors,
            "s3_url": f"https://{BUCKET_NAME}.s3.{REGION}.amazonaws.com/{filename}"
        }

    except Exception as e:
        return {
            "source_image": image_url,
            "error": str(e)
        }

# ✅ 여러 이미지 처리 API
@app.post("/generate/abstract/batch/")
async def generate_multiple(data: ImageURLs):
    tasks = [process_single_image(url) for url in data.image_urls]
    results = await asyncio.gather(*tasks)
    return {"results": results}

# ✅ 테스트 API
@app.get("/test-generate")
async def test_generate():
    try:
        with open("test.png", "rb") as f:
            image = Image.open(f)
            image.load()

        colors = extract_dominant_colors(image)
        color_names = [rgb_to_nearest_name(c) for c in colors]
        unique_colors = list(set(name for name in color_names if isinstance(name, str)))

        if not unique_colors:
            prompt = "A vibrant abstract painting in modern digital style"
        else:
            prompt = f"A beautiful abstract painting in modern digital art style, dominated by {', '.join(unique_colors)} colors, with flowing brushstrokes and vibrant tones"

        response = client.images.generate(
            model="dall-e-2",
            prompt="A colorful digital painting of a cyberpunk city at night",
            n=1,
            size="512x512"
        )
        gen_url = response.data[0].url

        img_data = requests.get(gen_url).content
        filename = f"test-{uuid.uuid4()}.png"
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=filename,
            Body=img_data,
            ContentType="image/png",
            ACL="public-read"
        )
        s3_url = f"https://{BUCKET_NAME}.s3.{REGION}.amazonaws.com/{filename}"

        return {
            "message": "Test successful",
            "prompt": prompt,
            "colors": unique_colors,
            "s3_url": s3_url
        }

    except Exception as e:
        return {
            "message": "Test failed",
            "error": str(e)
        }
