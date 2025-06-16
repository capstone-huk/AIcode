import os
import uuid
import asyncio
import requests
import openai
import boto3
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

# Load .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# S3 설정
s3 = boto3.client(
    "s3",
    region_name=os.getenv("AWS_S3_REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)
BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME")
REGION = os.getenv("AWS_S3_REGION")

# FastAPI + CORS
app = FastAPI()
origins = [
    "http://localhost:3000",  # 개발용
    "https://your-frontend.com",  # 실제 프론트 주소
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 요청 모델
class ImageURLs(BaseModel):
    image_urls: List[str]

# 색상 추출 함수
def extract_dominant_colors(img: Image.Image, num_colors=5):
    img = img.convert('RGB').resize((150, 150))
    img_data = np.array(img).reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_colors, random_state=42).fit(img_data)
    colors = kmeans.cluster_centers_.astype(int)
    return [tuple(color) for color in colors]

# 색상명 변환
# 대표적인 색상 이름과 hex 코드 직접 정의
CSS_COLOR_MAP = {
    "black": "#000000",
    "white": "#ffffff",
    "red": "#ff0000",
    "green": "#008000",
    "blue": "#0000ff",
    "yellow": "#ffff00",
    "cyan": "#00ffff",
    "magenta": "#ff00ff",
    "gray": "#808080",
    "orange": "#ffa500",
    "pink": "#ffc0cb",
    "purple": "#800080",
    "brown": "#a52a2a",
    "beige": "#f5f5dc",
    "olive": "#808000",
    "navy": "#000080",
    "teal": "#008080"
}

def closest_color(requested_color):
    min_distance = float("inf")
    closest_name = None

    for name, hex_val in CSS_COLOR_MAP.items():
        r_c, g_c, b_c = tuple(int(hex_val[i:i+2], 16) for i in (1, 3, 5))
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        distance = rd + gd + bd

        if distance < min_distance:
            min_distance = distance
            closest_name = name

    return closest_name

def rgb_to_nearest_name(rgb):
    try:
        return webcolors.rgb_to_name(rgb)
    except ValueError:
        return closest_color(rgb)


# 단일 이미지 처리
async def process_single_image(image_url: str):
    try:
        # 1. 원본 이미지 다운로드
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))

        # 2. 색 추출 + 이름화
        colors = extract_dominant_colors(image)
        color_names = [rgb_to_nearest_name(c) for c in colors]

        # 3. 프롬프트 생성
        prompt = f"An abstract painting using these colors: {', '.join(color_names)}"

        # 4. OpenAI 생성
        ai_response = openai.Image.create(
            prompt=prompt,
            n=1,
            size="512x512"
        )
        gen_url = ai_response["data"][0]["url"]

        # 5. 생성 이미지 다운로드
        img_data = requests.get(gen_url).content
        filename = f"{uuid.uuid4()}.png"

        # 6. S3 업로드
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=filename,
            Body=img_data,
            ContentType="image/png",
            ACL="public-read"
        )

        s3_url = f"https://{BUCKET_NAME}.s3.{REGION}.amazonaws.com/{filename}"

        return {
            "source_image": image_url,
            "prompt": prompt,
            "colors": color_names,
            "s3_url": s3_url
        }

    except Exception as e:
        return {
            "source_image": image_url,
            "error": str(e)
        }

# 병렬 처리 API
@app.post("/generate/abstract/batch/")
async def generate_multiple(data: ImageURLs):
    tasks = [process_single_image(url) for url in data.image_urls]
    results = await asyncio.gather(*tasks)
    return {"results": results}

@app.get("/test-generate")
async def test_generate():
    try:
        with open("test.png", "rb") as f:
            image = Image.open(f)
            image.load()

        colors = extract_dominant_colors(image)

        # 🎯 문자열만 필터링 (None 등 제거)
        color_names = [rgb_to_nearest_name(c) for c in colors]
        filtered_color_names = [name for name in color_names if isinstance(name, str) and name.strip()]

        # 🎯 중복 제거
        unique_colors = list(set(filtered_color_names))

        # ✅ 프롬프트 생성 (빈 리스트 방어 포함)
        if not unique_colors:
            prompt = "A vibrant abstract painting in modern digital style"
        else:
            prompt = (
                f"A beautiful abstract painting in modern digital art style, "
                f"dominated by {', '.join(unique_colors)} colors, with flowing brushstrokes and vibrant tones"
            )

        print("🔵 생성 프롬프트:", prompt)
        
        print("✅ prompt =", prompt, type(prompt))
        print("✅ model =", "dall-e-2", type("dall-e-2"))
        print("✅ size =", "512x512", type("512x512"))





        # ✅ 최신 방식으로 OpenAI client 사용
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.images.generate(
            model="dall-e-2",
            prompt=prompt,
            n=1,
            size="512x512"
        )
        gen_url = response.data[0].url

        # 이미지 다운로드 후 S3 업로드
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
            "colors": color_names,
            "s3_url": s3_url
        }

    except Exception as e:
        return {
            "message": "Test failed",
            "error": str(e) if str(e) else "알 수 없는 에러"
        }
