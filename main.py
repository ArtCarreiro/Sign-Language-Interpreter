#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intérprete de Libras com IA
Reconhece gestos da Língua Brasileira de Sinais e converte em voz
"""

import cv2
import time
import sys
import os

# Adicionar pasta src ao path
sys.path.append('src')

from capture import LibrasCapture
from model import LibrasModel
from speak import LibrasSpeaker

class LibrasInterpreter:
    def __init__(self):
        print("Inicializando Intérprete de Libras...")
        
        # Componentes do sistema
        self.capture = LibrasCapture()
        self.model = LibrasModel()
        self.speaker = LibrasSpeaker(use_offline=True)
        
        # Controles
        self.last_prediction = None
        self.last_speak_time = 0
        self.speak_cooldown = 2.0  # Cooldown de 2 segundos entre falas
        self.prediction_history = []
        self.history_size = 10
        
        print("Sistema inicializado com sucesso!")
        print("Pressione 'q' para sair, 'r' para resetar histórico")
    
    def get_stable_prediction(self, current_prediction):
        """Obtém predição estável baseada no histórico"""
        if current_prediction:
            self.prediction_history.append(current_prediction)
        
        # Mantém apenas as últimas N predições
        if len(self.prediction_history) > self.history_size:
            self.prediction_history.pop(0)
        
        # Se temos pelo menos 5 predições
        if len(self.prediction_history) >= 5:
            # Conta a predição mais frequente
            from collections import Counter
            most_common = Counter(self.prediction_history).most_common(1)
            if most_common and most_common[0][1] >= 3:  # Pelo menos 3 ocorrências
                return most_common[0][0]
        
        return None
    
    def should_speak(self, prediction):
        """Determina se deve falar a predição"""
        current_time = time.time()
        
        # Verifica cooldown
        if current_time - self.last_speak_time < self.speak_cooldown:
            return False
        
        # Verifica se é uma nova predição
        if prediction != self.last_prediction:
            return True
        
        return False
    
    def draw_info(self, frame, prediction, confidence, fps):
        """Desenha informações na tela"""
        height, width = frame.shape[:2]
        
        # Fundo para o texto
        cv2.rectangle(frame, (10, 10), (width - 10, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (width - 10, 120), (255, 255, 255), 2)
        
        # Título
        cv2.putText(frame, "Interprete de Libras com IA", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Predição atual
        if prediction:
            text = f"Letra detectada: {prediction}"
            conf_text = f"Confianca: {confidence:.2f}"
            cv2.putText(frame, text, (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, conf_text, (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        else:
            cv2.putText(frame, "Nenhuma letra detectada", (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Instruções
        cv2.putText(frame, "Pressione 'q' para sair | 'r' para resetar", 
                   (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run(self):
        """Loop principal da aplicação"""
        print("\nIniciando reconhecimento de Libras...")
        print("Posicione sua mão em frente à câmera e faça o sinal de uma letra")
        
        fps_counter = 0
        fps_start_time = time.time()
        fps = 0
        
        try:
            while True:
                # Captura frame e detecta mãos
                frame, landmarks_list, results = self.capture.get_frame_with_hands()
                
                if frame is None:
                    print("Erro ao capturar vídeo!")
                    break
                
                prediction = None
                confidence = 0.0
                
                # Se detectou mãos, faz predição
                if landmarks_list:
                    prediction, confidence = self.model.predict(landmarks_list)
                
                # Obtém predição estável
                stable_prediction = self.get_stable_prediction(prediction)
                
                # Se deve falar a predição
                if stable_prediction and self.should_speak(stable_prediction):
                    print(f"Falando: {stable_prediction}")
                    self.speaker.speak_letter(stable_prediction)
                    self.last_prediction = stable_prediction
                    self.last_speak_time = time.time()
                
                # Calcula FPS
                fps_counter += 1
                if fps_counter >= 30:
                    fps = fps_counter / (time.time() - fps_start_time)
                    fps_counter = 0
                    fps_start_time = time.time()
                
                # Desenha informações
                self.draw_info(frame, prediction, confidence, fps)
                
                # Mostra frame
                cv2.imshow('Interprete de Libras', frame)
                
                # Controles de teclado
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.prediction_history.clear()
                    self.last_prediction = None
                    print("Histórico resetado!")
        
        except KeyboardInterrupt:
            print("\nInterrompido pelo usuário")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Limpa recursos"""
        print("Finalizando sistema...")
        self.capture.release()
        print("Sistema finalizado!")

def main():
    """Função principal"""
    print("=" * 50)
    print("   INTÉRPRETE DE LIBRAS COM IA")
    print("=" * 50)
    
    try:
        interpreter = LibrasInterpreter()
        interpreter.run()
    except Exception as e:
        print(f"Erro ao inicializar o sistema: {e}")
        print("Verifique se a webcam está conectada e as dependências instaladas")

if __name__ == "__main__":
    main()
