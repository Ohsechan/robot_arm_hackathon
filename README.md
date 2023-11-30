## Project
This is the source code that won the Grand Prize at the hackathon organized by pinklab on September 7, 2023.
The task involved coding a control program to use a real robotic arm to rearrange a randomly stacked tower of blocks according to their colors and then rebuild it in the correct order.

{% include video id="XaDWIcv2s80" provider="youtube" %}

<iframe width="1920" height="1080" src="https://www.youtube.com/embed/XaDWIcv2s80?list=PLx5EbqT-6Y09HxXUWvCjNI92XtDfAoq-j" title="pinklab contest demo 1" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

## Quick start

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
  pip install -r requirements.txt</code></pre>

<pre><code>python3 detect.py --weights ./runs/train/yolov3_dobot/weights/best.pt --source 0 --img 640 --conf-thres 0.5</code></pre>
