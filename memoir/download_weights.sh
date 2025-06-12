#!/bin/bash

set -e  # 에러 발생 시 중단

echo "🔽 StyleShot 모델 및 종속 모델 다운로드 시작..."

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

# snapshot_download로 받을 모델들
declare -A snapshot_models=(
  ["StyleShot_lineart"]="Gaojunyao/StyleShot_lineart"
  ["CLIP-ViT-H-14-laion2B-s32B-b79K"]="laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
)

# git clone으로 받을 모델
declare -A git_models=(
  ["stable-diffusion-v1-5"]="https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5"
)

# snapshot_download 모델들 다운로드
for dir in "${!snapshot_models[@]}"; do
  path="$MODEL_DIR/$dir"
  repo="${snapshot_models[$dir]}"

  if [ ! -d "$path" ]; then
    echo "📦 snapshot: $repo → $path"
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$repo', local_dir='$path', local_dir_use_symlinks=False)"
  else
    echo "✅ 이미 존재: $path"
  fi
done

# git clone 모델들 다운로드
for dir in "${!git_models[@]}"; do
  path="$MODEL_DIR/$dir"
  url="${git_models[$dir]}"

  if [ ! -d "$path" ]; then
    echo "📦 git clone: $url → $path"
    GIT_LFS_SKIP_SMUDGE=1 git clone "$url" "$path"
    echo "➡️ Git LFS pull 시작 (파일 다운로드)"
    cd "$path"
    git lfs pull
    cd -
  else
    echo "✅ 이미 존재: $path"
  fi
done

echo "🎉 모든 모델 다운로드 완료!"
