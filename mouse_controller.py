import numpy as np
import pyautogui

from config import SCREEN_W, SCREEN_H, MARGIN_X, MARGIN_Y, SMOOTHING, CLICK_DISTANCE_LIMIT

def map_to_screen(val, margin, screen_size):
    val_clamped = np.clip(val, margin, 1 - margin)
    return int((val_clamped - margin) / (1 - 2 * margin) * screen_size)

def move_cursor(point, smooth_x, smooth_y): 
    raw_x = map_to_screen(point.x, MARGIN_X, SCREEN_W)
    raw_y = map_to_screen(point.y, MARGIN_Y, SCREEN_H)

    new_smooth_x = int(smooth_x * SMOOTHING + raw_x * (1 - SMOOTHING))
    new_smooth_y = int(smooth_y * SMOOTHING + raw_y * (1 - SMOOTHING))
    pyautogui.moveTo(smooth_x, smooth_y)

    return new_smooth_x, new_smooth_y

def press_mouse_button(point1, point2, pressing, mouse_button):
    dist = ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2) ** 0.5

    if dist < CLICK_DISTANCE_LIMIT and not pressing:
        pyautogui.mouseDown(button=mouse_button)
        return True
    elif dist > CLICK_DISTANCE_LIMIT and pressing:
        pyautogui.mouseUp(button=mouse_button)
        return False
    
    return pressing