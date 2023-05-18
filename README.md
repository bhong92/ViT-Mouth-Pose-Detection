# Setup
## 1) Create new anaconda environment
conda create -n ATCC python=3.7.7  
conda activate ATCC  
## 2) Packages for Clicking
conda install -c conda-forge opencv=4.5.3 dlib=19.22.0 sk-video=1.1.10  
python -m pip install tensorflow-gpu  
python -m pip install numpy  
python -m pip install av  
python -m pip install editdistance  
python -m pip install configargparse  
python -m pip install six  
python -m pip install moviepy==0.2.3.5  
python -m pip install opencv-python  
python -m pip install imageio-ffmpeg  
python -m pip install tensorflow_addons  
## 3) Packages for Eye/Face Tracking
conda install pytorch torchvision torchaudio cpuonly -c pytorch  
python -m pip install omegaconf  
python -m pip install face-alignment  
python -m pip install fvcore  
python -m pip install pandas  
python -m pip install pyyaml  
python -m pip install scipy  
python -m pip install tensorboardX  
python -m pip install yacs  
python -m pip install mediapipe  
python -m pip install timm  
python -m pip install playsound==1.2.2  
## 4) Packages for Speech recognition
python -m pip install SpeechRecognition  
python -m pip install pyttsx3  
python -m pip install openai  
python -m pip install pocketsphinx  
python -m pip install pyautogui  
python -m pip install gtts  
python -m pip install playsound  
python -m pip install pyaudio  
python -m pip install AppOpener  
python -m pip install grpcio-status==1.46.3  
python -m pip install grpcio==1.46.3  
python -m pip install google-api-python-client  
python -m pip install google-cloud-speech  
# References
[1] Gaze_estimator: https://github.com/YW-Ma/pytorch_mpiigaze_demo  

[2] https://pyimagesearch.com/2021/04/05/opencv-face-detection-with-haar-cascades/

[3] https://github.com/aravindpai/Speech-Recognition/blob/master/Speech%20Recognition.ipynb

[4] https://www.kaggle.com/c/tensorflow-speech-recognition-challenge

[5] https://towardsdatascience.com/facial-mapping-landmarks-with-dlib-python-160abcf7d672

[6] https://pypi.org/project/pocketsphinx/

[7] https://codelabs.developers.google.com/codelabs/cloud-speech-text-python3#0

[8] https://www.sciencedirect.com/science/article/pii/S2667241322000039

[9] https://keras.io/examples/vision/image_classification_with_vision_transformer/
