import pyautogui

MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

MARGIN_X = 0.15   
MARGIN_Y = 0.15  

SMOOTHING = 0.7

CLICK_DISTANCE_LIMIT = 0.04 

SCREEN_W, SCREEN_H = pyautogui.size()