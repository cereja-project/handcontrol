import cv2
import mediapipe as mp
import statistics
from calango.devices import Mouse, Capture

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cam = Capture()
mouse = Mouse(noise_threshold=10)
with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=1) as hands:
    counter = 0
    pos_x = []
    pos_y = []

    while True:
        image = next(cam)

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)
        if not results.multi_hand_landmarks:
            continue
        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mask = image[:, :, :] * 0
        w, h = mouse.window_size

        if results.multi_hand_landmarks:
            x = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w
            y = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h
            pos_x.append(x)
            pos_y.append(y)
            if counter % 5 == 0:
                x, y = statistics.median(pos_x), statistics.median(pos_y)
                pos_x.clear()
                pos_y.clear()
                mouse.position = (x, y)
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                        mask, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow('MediaPipe Hands', mask)
        counter += 1
        if cv2.waitKey(5) & 0xFF == 27:
            break
cam.stop()
