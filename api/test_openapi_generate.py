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

# 환경 변수 로드
load_dotenv()

# OpenAI 클라이언트 설정
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# S3 설정
s3 = boto3.client(
    "s3",
    region_name=os.getenv("AWS_S3_REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)
BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME")
REGION = os.getenv("AWS_S3_REGION")

# FastAPI 앱 생성 및 CORS 설정
app = FastAPI()
origins = [
    "http://localhost:3000",
    "https://your-frontend-url.com",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 색상 이름 매핑
CSS_COLOR_MAP = {
    "black": "#000000", "white": "#ffffff", "red": "#ff0000", "green": "#008000", "blue": "#0000ff",
    "yellow": "#ffff00", "cyan": "#00ffff", "magenta": "#ff00ff", "gray": "#808080", "orange": "#ffa500",
    "pink": "#ffc0cb", "purple": "#800080", "brown": "#a52a2a", "beige": "#f5f5dc", "olive": "#808000",
    "navy": "#000080", "teal": "#008080", "lime": "#00ff00", "maroon": "#800000", "gold": "#ffd700",
    "silver": "#c0c0c0", "indigo": "#4b0082", "coral": "#ff7f50", "khaki": "#f0e68c"
}

def closest_color(requested_color):
    min_distance = float("inf")
    closest_name = None
    for name, hex_val in CSS_COLOR_MAP.items():
        r_c, g_c, b_c = tuple(int(hex_val[i:i+2], 16) for i in (1, 3, 5))
        distance = sum((c1 - c2) ** 2 for c1, c2 in zip(requested_color, (r_c, g_c, b_c)))
        if distance < min_distance:
            min_distance = distance
            closest_name = name
    return closest_name

def extract_dominant_colors(img: Image.Image, num_colors=5):
    img = img.convert("RGB").resize((150, 150))
    img_data = np.array(img).reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_colors, random_state=42).fit(img_data)
    return [tuple(map(int, color)) for color in kmeans.cluster_centers_]

def rgb_to_nearest_name(rgb):
    return closest_color(rgb)

class ImageURLs(BaseModel):
    image_urls: List[str]

async def process_image_from_url(image_url: str):
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        colors = extract_dominant_colors(image)
        color_names = [rgb_to_nearest_name(c) for c in colors]
        unique_colors = list(set(color_names))

        prompt = (
            f"A beautiful abstract painting in modern digital art style, "
            f"dominated by {', '.join(unique_colors)} colors, with flowing brushstrokes and vibrant tones"
        )

        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024"
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
        s3_url = f"https://{BUCKET_NAME}.s3.{REGION}.amazonaws.com/{filename}"

        return {"source_image": image_url, "prompt": prompt, "s3_url": s3_url}
    except Exception as e:
        return {"source_image": image_url, "error": str(e)}

@app.post("/generate-image")
async def generate_image_from_urls(data: ImageURLs):
    tasks = [process_image_from_url(url) for url in data.image_urls]
    results = await asyncio.gather(*tasks)
    return {"results": results}

@app.get("/test-generate")
async def test_generate():
    try:
        with open("test.png", "rb") as f:
            image = Image.open(f)
            image.load()

        colors = extract_dominant_colors(image)
        color_names = [rgb_to_nearest_name(c) for c in colors]
        unique_colors = list(set(color_names))

        prompt = (
            f"A beautiful abstract painting in modern digital art style, "
            f"dominated by {', '.join(unique_colors)} colors, with flowing brushstrokes and vibrant tones"
        )

        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024"
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
        
        print(s3_url)

        return {
            "message": "Test successful",
            "prompt": prompt,
            "colors": unique_colors,
            "s3_url": s3_url
        }
        

    except Exception as e:
        return {"message": "Test failed", "error": str(e)}
