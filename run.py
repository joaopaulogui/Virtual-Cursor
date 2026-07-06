import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import urllib.request
import os
import threading
import queue
import time

from config import MODEL_PATH, MODEL_URL, SCREEN_W, SCREEN_H
from drawer import draw_hand, draw_active_zone, draw_text
from mouse_controller import move_cursor, press_mouse_button

TIP_THUMB  = 4
TIP_INDEX  = 8
TIP_MIDDLE = 12
TIP_RING = 16
TIP_PINKY = 20

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0  # remove o delay artificial de 0.1s por chamada do pyautogui

if not os.path.exists(MODEL_PATH):
    print("Modelo não encontrado. Baixando modelo...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Modelo baixado.")


def main():
    smooth_x, smooth_y = SCREEN_W // 2, SCREEN_H // 2
    left_button = False
    right_button = False
    middle_button = False

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # fila com tamanho 1: a janela sempre mostra o frame mais recente disponível,
    # sem acumular atraso caso a exibição fique lenta (ex: janela minimizada)
    frame_queue = queue.Queue(maxsize=1)
    stop_flag = threading.Event()

    def capture_loop():
        nonlocal smooth_x, smooth_y, left_button, right_button
        prev_time = time.time()

        with vision.HandLandmarker.create_from_options(options) as detector:
            while cap.isOpened() and not stop_flag.is_set():
                ok, frame = cap.read()
                if not ok:
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = detector.detect(mp_image)

                draw_active_zone(frame)

                if result.hand_landmarks:
                    landmarks = result.hand_landmarks[0]
                    draw_hand(frame, landmarks)

                    tip = landmarks[TIP_INDEX]
                    smooth_x, smooth_y = move_cursor(tip, smooth_x, smooth_y)

                    thumb = landmarks[TIP_THUMB]
                    middle = landmarks[TIP_MIDDLE]
                    ring = landmarks[TIP_RING]
                    pinky = landmarks[TIP_PINKY]

                    left_button = press_mouse_button(thumb, middle, left_button, "left")
                    right_button = press_mouse_button(thumb, ring, right_button, "right")
                    middle_button = press_mouse_button(thumb, pinky, middle_button, "middle")

                    status = "botão esquerdo" if left_button else "botão direito" if right_button else "botão do meio" if middle_button else "movendo"
                    color = (0, 80, 255) if left_button or right_button else (50, 210, 100)
                    draw_text(frame, status, (16, 36), color, size=12)
                    draw_text(frame, f"cursor: ({smooth_x}, {smooth_y})", (16, 64), color, size=12)
                else:
                    draw_text(frame, "Procurando mão...", (16, 36), color=(180, 180, 180), size=16)

                # FPS real do processamento (câmera + tracking + controle do mouse)
                now = time.time()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now
                draw_text(frame, f"FPS: {fps:.0f}", (16, 92), color=(255, 255, 255), size=12)
                print(f"FPS: {fps:.0f}")

                # sempre mantém apenas o frame mais recente na fila
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                frame_queue.put(frame)

    thread = threading.Thread(target=capture_loop, daemon=True)
    thread.start()

    while not stop_flag.is_set():
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue

        cv2.imshow("Mouse Virtual  |  Q para sair", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_flag.set()
            break

    stop_flag.set()
    thread.join(timeout=2)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()