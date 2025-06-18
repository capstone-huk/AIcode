# Memoir AI (StyleShot + FastAPI)

이 레포지토리는 **이화여자대학교 2025-1학기 캡스톤디자인 ‘그로쓰’** 수업의 **28팀 HUK** 프로젝트인 **memoir**의 AI 부분을 다루고 있습니다.

본 프로젝트는 open source AI 모델인 [`StyleShot`](https://github.com/open-mmlab/StyleShot)을 사용하여, **사용자가 선택한 스타일 이미지**를 바탕으로 **개인화된 전시 티켓 이미지를 생성**하는 기능을 FastAPI 서버로 제공합니다.

---

## 1. 레포지토리 및 모델 클론

먼저 프로젝트 레포지토리와 StyleShot 모델을 로컬에 클론합니다.

```bash
# memoir AI 코드 클론
git clone https://github.com/capstone-huk/AIcode.git

# StyleShot 모델 클론 (memoir 폴더 내부에서 실행)
cd AIcode/memoir
git clone https://github.com/open-mmlab/StyleShot
```


## 2. 환경 설정

아래 명령어를 통해 Python 가상환경을 만들고 필요한 패키지를 설치합니다.

```bash
# Conda 가상환경 생성 및 활성화
conda create -n memoir python=3.10 -y
conda activate memoir

# requirements.txt 설치
pip install -r requirements.txt
```


### 3. 모델 가중치 다운로드

StyleShot 모델을 실행하기 위해 필요한 가중치(weight) 파일을 다운로드합니다.  
아래 명령어를 실행하면 `memoir/models/` 폴더가 생성되고, 그 안에 필요한 모델 파일들이 저장됩니다.

```bash
# 실행 권한 부여
chmod +x download_weights.sh

# 가중치 다운로드 스크립트 실행
./download_weights.sh
```


### 4. 환경변수 설정

S3 연동을 위해 `.env` 파일을 설정해야 합니다.  
`memoir` 폴더 내에 `.env` 파일을 생성하고 메일로 안내받은 코드를 넣어 저장합니다.


### 5. main.py 함수 실행

`main.py`는 FastAPI 기반 서버의 메인 코드로, 다음과 같은 주요 기능을 포함합니다:

- 사용자가 선택한 스타일 이미지(`style_reference`)의 URL들을 **List 형태**로 입력받습니다.
- 각 스타일 이미지 URL에 대해 **병렬로 티켓 이미지를 생성**합니다. (`/generate/` 엔드포인트 사용)
  - `content_image` 폴더에 있는 사진들 중 무작위로 하나를 선택하여 StyleShot 모델에 입력합니다.
  - 선택된 스타일 이미지와 content 이미지를 조합해 **티켓 이미지를 생성**합니다.
  - 생성된 이미지를 **S3에 업로드**하고, 결과 이미지의 URL을 응답으로 반환합니다.
- 테스트용 엔드포인트인 `/test-upload/`를 통해 S3 업로드 기능을 확인할 수 있습니다.
- 그 외에도 CORS 설정, S3 업로드 함수, FastAPI 앱 초기화 관련 코드가 포함되어 있습니다.


### 6. 서버 실행

아래 명령어를 통해 로컬에서 FastAPI 서버를 실행할 수 있습니다:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```


### 7. 엔드포인트 테스트

FastAPI 서버가 실행 중일 때, 아래의 명령어로 API를 테스트할 수 있습니다.

---

#### `/test-upload/` (S3 업로드 테스트)

```bash
curl -X POST http://localhost:8000/test-upload/
```

#### `/generate/` (스타일 이미지 기반 티켓 생성)
```bash
curl -X POST http://localhost:8000/generate/ \
  -F "style_url=https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Golde33443.jpg/640px-Golde33443.jpg"
```
해당 API는 사용자가 입력한 스타일 이미지 URL을 기반으로 content 이미지를 합성하여
새로운 티켓 이미지를 생성한 후, 결과를 S3에 저장하고 해당 URL을 반환합니다.


---

## 프로젝트 요약

- 본 레포지토리는 이화여자대학교 2025-1학기 캡스톤디자인 '그로쓰' 수업의 28팀 **HUK**의 프로젝트 **memoir**에서, **AI 이미지 생성 서버** 부분만을 다룹니다.
- 사용자는 스타일 이미지 URL을 입력하면, 이를 기반으로 전시 티켓 이미지를 생성할 수 있습니다.
- 이미지 생성에는 오픈소스 모델 **StyleShot**을 사용하며, FastAPI를 통해 API로 배포됩니다.
- 최종 생성된 티켓 이미지는 **AWS S3**에 저장되며, 클라이언트는 해당 URL을 응답으로 받습니다.

### 기술 스택

- **AI 모델**: [StyleShot](https://github.com/open-mmlab/StyleShot)
- **서버 프레임워크**: FastAPI
- **모델 배포 방식**: 로컬 또는 EC2 환경에서 `uvicorn` 실행
- **이미지 저장**: AWS S3 연동
- **병렬 처리**: 여러 스타일 이미지 요청을 동시에 처리

---

## 참고

- StyleShot 원본 레포지토리: https://github.com/open-mmlab/StyleShot  
- 티켓 이미지는 개인적인 감상 기록 및 전시 아카이빙 경험을 위해 생성됩니다.

---
