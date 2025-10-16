import cv2
import mediapipe as mp
import numpy as np

class LibrasCapture:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)
        
    def extract_landmarks(self, landmarks):
        """Extrai coordenadas dos 21 pontos da mão"""
        coords = []
        for lm in landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        return np.array(coords)
    
    def get_frame_with_hands(self):
        """Captura frame e detecta mãos"""
        ret, frame = self.cap.read()
        if not ret:
            return None, None, None
            
        # Espelha a imagem para ficar mais natural
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        landmarks_list = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Desenha os pontos da mão
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                # Extrai coordenadas
                coords = self.extract_landmarks(hand_landmarks)
                landmarks_list.append(coords)
        
        return frame, landmarks_list, results
    
    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
