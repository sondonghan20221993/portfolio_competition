<div align="center">
  <h1>이미지변형과 모델의 정확도</h1>
  <p>이미지 변형에 대한 모델의 내성을 높이기 위해 학습 이미지에 변형을 적용하는 것이 목표이다. </p>
</div>
<br />

## Introduction

모델 학습 시 이미지에 다양한 변형을 적용하여, 실제 환경에서 목표 물체가 손상되거나 외관이 달라졌을 때에도 대응할 수 있는 강인한 모델을 제작하는 것을 목표로 프로젝트를 진행합니다.

<br /><br />

## Tech Stack

This project leverages the following technologies and tools:  

- 🧠 PyTorch - 딥러닝 학습을 위한 프레임워크
- 🧮 NumPy — 행렬, 수치(숫자, 벡터등)연산
- 📸 OpenCV - 이미지 변형, 전처리
- 📊 Grad-CAM - 모델의 이미지 예측 시각화
- 📊 Matplotlib - 결과, 데이터 시각화
- 🚀 CUDA - 학습을 위한 GPU가속

<br />

## Features

프로젝트의 핵심 기능

- ### 이미지 변형을 활용한 모델 학습
  다양한 노이즈(salt-paper, 다른이미지 합성 등)를 적용시켜 딥러닝 모델을 학습해
  모델의 강인성과 일반화 성능을 향상시킵니다.
- ### 정확도 분석 + Grad-CAM 시각화
  이미지 변형 유형별로 모델 성능을 평가하고, 정확도 변화를 시각화하여 강점과 약점 파악 및
  입력 이미지의 어떤 영역을 기준으로 판단했는지 시각화하여, 모델의 판단 근거를 해석한다.
- ### GPU 가속 학습
  CUDA를 지원하는 GPU를 활용하여 대규모 데이터셋 학습 속도를 크게 향상시킵니다.
- ### 데이터 전처리 및 처리
  OpenCV, NumPy, Pandas 등을 사용하여 이미지를 효율적으로 전처리하고, 데이터셋을 관리하여 모델 학습에 최적화합니다.
<br /><br />

## Performance

- **이미지 변형**  
  ![noising img](/design_folder/assets/Noising_Data.png)

- **Grad-CAM 시각화**  
  ![Gradcam_img](/design_folder/assets/gradcam.png)
<br /><br />

## Future Development Plans  


<br /><br />

This project is licensed under the [GPL-3.0 license](https://github.com/dwiwijaya/dwiwijaya.com/blob/master/LICENSE).  
Feel free to use, modify, and share it while adhering to the terms.
