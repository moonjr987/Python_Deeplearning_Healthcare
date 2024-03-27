<h2>2023 헬스케어 AI 경진대회 코드</h2>


<h5>사용된 기술: Tensorflow, Keras</h5>
<h5>모델: VGG16</h5>
<h5>팀에서 맡은 부분: preprocessing.py(전처리)</h5>
<h5>전처리 과정에서 f1_score를 올리기 위해 전이학습을 시킨 후, 레이어의 수를 추가하였습니다.</h5>
<h5>프로젝트 진행: 데이터셋 전처리 방법으로 이미지 데이터와 JSON형식의 라벨 데이터를 입력받아, 이미지를 이진 분류 라벨 (0 또는 1)로 변환하였습니다. 모델 학습 과정으로 TensorFlow를 사용하여 간단한 컨볼루션 신경망(CNN) 모델을 정의하고 훈련하는데 사용하며 Conv2D로 이미지의 특징을 추출하였고, MaxPooling2D로이미지를 다운샘플링하며 계산 비용을 줄였습니다. 또 Dropout을 사용하여 일부 뉴런을 비활성화하여 과적합을 방지하였습니다. 최종적으로 테스트 데이터 3000개에 대하여 sklearn.metrics.f1_score를 사용하여 f1_score 출력하였습니다.</h5>
