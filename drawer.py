import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw

from config import MARGIN_X, MARGIN_Y

FINGER_COLORS = {
    "thumb":  (180, 100, 255),
    "index":  (255, 150,  50),
    "middle": ( 50, 210, 100),
    "ring":   ( 50, 150, 255),
    "pinky":  (200,  80, 200),
}

FINGER_GROUPS = {
    "thumb":  [1, 2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20],
}

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

def get_color(idx):
    for finger, group in FINGER_GROUPS.items():
        if idx in group:
            return FINGER_COLORS[finger]
    return (220, 220, 220)

def draw_hand(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], get_color(a), 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(pts):
        color = get_color(i)
        r = 8 if i == 0 else 5
        cv2.circle(frame, (x, y), r, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), r, (255, 255, 255), 1, cv2.LINE_AA)

def draw_text(frame, text, pos, color=(255, 255, 255), size=18):
    img_pil = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype("font.ttf", size)
    
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, fill=(color[0], color[1], color[2]), font=font)

    frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def draw_active_zone(frame):
    h, w = frame.shape[:2]
    x1 = int(MARGIN_X * w)
    y1 = int(MARGIN_Y * h)
    x2 = int((1 - MARGIN_X) * w)
    y2 = int((1 - MARGIN_Y) * h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), 1)
    draw_text(frame, "zona ativa", (x1 + 6, y1 + 18), color=(100, 100, 100), size=10)