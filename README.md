# 영상처리를 활용한 협동로봇 제어대회
## 1. 프로젝트 요약
### 1-1. 설명
3가지 색상의 블록으로 이루어진 블록 탑을 색상에 따라 이동시킨 후, 원래 순서대로 다시 쌓는 과제가 주어진다. Manipulator와 mono usb camera를 활용하여 주어진 과제를 수행한다.
### 1-2. 수상 이력
- 영상처리를 활용한 협동로봇 제어대회 최우수상
### 1-3. 개발 규모
- 인원 : 3명
- 일시 : 2023.09.04 ~ 2023.09.15
### 1-4. 개발환경
- Ubuntu 22.04 Desktop
- Python : Yolov3
### 1-5. 하드웨어
- DOBOT Magician Lite
- mono usb camera

## 2. 데모 영상
[![영상처리를 활용한 협동로봇 제어대회 1](http://img.youtube.com/vi/XaDWIcv2s80/0.jpg)](https://youtu.be/XaDWIcv2s80?t=0s)
[![영상처리를 활용한 협동로봇 제어대회 2](http://img.youtube.com/vi/9p8cvShjsBM/0.jpg)](https://youtu.be/9p8cvShjsBM?t=0s)

## 3. 실행 방법
<pre><code># Clone Repository
git clone https://github.com/Ohsechan/robot_arm_hackathon.git
cd robot_arm_hackathon
# 가상환경 새로 만들고
python -m venv yolov3
# 가상환경 실행
source yolov3/Scripts/activate
# 필요한 모듈 설치
pip install opencv-python
# 의존성 설치
pip install -r requirements.txt
# 실행
python3 detect.py --weights ./runs/train/yolov3_dobot/weights/best.pt --source 0 --img 640 --conf-thres 0.5</code></pre>
