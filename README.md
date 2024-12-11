### ImageAugmentation.ipynb
##### [코드 바로가기 ▶](https://github.com/capstone-huk/AIcode/blob/main/ImageAugmentation.ipynb)

Google Drive에 저장된 이미지 파일을 변환하고 증강하는 코드

- Google Drive와 연동하여 이미지를 처리
- HEIC 및 JPG 이미지를 PNG로 변환
- `torchvision.transforms`를 활용해 다양한 이미지 증강 수행
  - 랜덤 수평 및 수직 뒤집기
  - 랜덤 회색조 변환
  - 채도 및 밝기 조정
  - 랜덤 회전
  - 30% 확률로 랜덤 노이즈 추가
- 각 이미지에 대해 5개의 증강된 이미지 생성

### OpenBankingFraudDetection.ipynb
##### [코드 바로가기 ▶](https://github.com/capstone-huk/AIcode/blob/main/OpenBankingFraudDetection.ipynb)

카드 거래 내역에서 이상 거래를 탐지할 수 있는 알고리즘

- scikit-learn의 Isolation Forest 알고리즘 사용
- 해외 카드 거래 내역 더미 데이터셋 사용
- 추후 오픈뱅킹을 통해 카드 거래 내역을 가져올 예정
