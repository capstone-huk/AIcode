#!/bin/bash

set -e  # 에러 발생 시 중단

echo "🔽 StyleShot 모델 및 종속 모델 다운로드 시작..."

# 기본 디렉토리 구조 설정
MODEL_DIR="models"
mkdir -p $MODEL_DIR

# huggingface-cli 설치 확인
if ! command -v huggingface-cli &> /dev/null; then
  echo "⚠️ huggingface-cli가 설치되어 있지 않습니다. 설치 중..."
  pip install -q huggingface_hub
fi

# git-lfs 설치 확인
if ! command -v git-lfs &> /dev/null; then
  echo "⚠️ git-lfs가 설치되어 있지 않습니다. 설치 중..."
  sudo apt-get update && sudo apt-get install -y git-lfs
  git lfs install
fi

# 모델 리스트
declare -A models=(
  ["StyleShot_lineart"]="Gaojunyao/StyleShot_lineart"
  ["StyleShot_contour"]="Gaojunyao/StyleShot"
  ["stable-diffusion-v1-5"]="runwayml/stable-diffusion-v1-5"
  ["CLIP-ViT-H-14-laion2B-s32B-b79K"]="laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
)

# 모델 다운로드
for dir in "${!models[@]}"; do
  path="$MODEL_DIR/$dir"
  repo="${models[$dir]}"

  if [ ! -d "$path" ]; then
    echo "📦 $repo → $path"
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$repo', local_dir='$path', local_dir_use_symlinks=False)"
  else
    echo "✅ $repo 이미 존재: $path"
  fi
done

echo "🎉 모든 모델 다운로드 완료!"
