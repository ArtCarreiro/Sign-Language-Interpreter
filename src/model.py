import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import os

class LibrasClassifier(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_classes=26):
        super(LibrasClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class LibrasModel:
    def __init__(self, model_path="models/libras_model.pth", 
                 label_encoder_path="models/label_encoder.pkl"):
        # Letras do alfabeto brasileiro de sinais
        self.classes = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
            'U', 'V', 'W', 'X', 'Y', 'Z'
        ]
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LibrasClassifier(num_classes=len(self.classes))
        self.model_path = model_path
        self.label_encoder_path = label_encoder_path
        
        # Cria diretório se não existir
        os.makedirs("models", exist_ok=True)
        
        # Tenta carregar modelo treinado
        if os.path.exists(model_path):
            self.load_model()
        else:
            print("Modelo não encontrado. Criando modelo básico...")
            self.create_basic_model()
    
    def create_basic_model(self):
        """Cria um modelo básico para demonstração"""
        # Salva o modelo não treinado
        torch.save(self.model.state_dict(), self.model_path)
        
        # Cria label encoder
        label_encoder = LabelEncoder()
        label_encoder.fit(self.classes)
        with open(self.label_encoder_path, 'wb') as f:
            pickle.dump(label_encoder, f)
        
        self.label_encoder = label_encoder
        print(f"Modelo básico criado em: {self.model_path}")
    
    def load_model(self):
        """Carrega modelo treinado"""
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        
        # Carrega label encoder
        with open(self.label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        print("Modelo carregado com sucesso!")
    
    def predict(self, landmarks):
        """Faz predição com base nos landmarks da mão"""
        if landmarks is None or len(landmarks) == 0:
            return None, 0.0
            
        # Normaliza os landmarks
        landmarks_normalized = self.normalize_landmarks(landmarks[0])
        
        # Converte para tensor
        input_tensor = torch.FloatTensor(landmarks_normalized).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            if confidence.item() > 0.7:  # Limiar de confiança
                predicted_class = self.classes[predicted.item()]
                return predicted_class, confidence.item()
            else:
                return None, confidence.item()
    
    def normalize_landmarks(self, landmarks):
        """Normaliza os landmarks para melhor performance"""
        landmarks = np.array(landmarks)
        
        # Normalização simples (pode ser melhorada)
        landmarks_normalized = (landmarks - landmarks.mean()) / (landmarks.std() + 1e-8)
        
        return landmarks_normalized
