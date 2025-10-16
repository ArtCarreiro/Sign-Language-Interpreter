import pyttsx3
import threading

class LibrasSpeaker:
    def __init__(self, use_offline=True):
        self.use_offline = use_offline
        
        # Usar pyttsx3 (funciona offline)
        self.engine = pyttsx3.init()
        # Configurar velocidade e volume
        self.engine.setProperty('rate', 150)  # Velocidade da fala
        self.engine.setProperty('volume', 0.8)  # Volume (0.0 a 1.0)
        
        # Tentar definir voz em português (se disponível)
        voices = self.engine.getProperty('voices')
        for voice in voices:
            if 'portuguese' in voice.name.lower() or 'brazil' in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                break
    
    def speak_offline(self, text):
        """Fala usando pyttsx3 (offline)"""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Erro na síntese de voz offline: {e}")
    
    def speak(self, text):
        """Fala o texto (usa thread para não bloquear)"""
        if not text or text.strip() == "":
            return
            
        def speak_thread():
            self.speak_offline(text)
        
        thread = threading.Thread(target=speak_thread)
        thread.daemon = True
        thread.start()
    
    def speak_letter(self, letter):
        """Fala uma letra do alfabeto"""
        letter_names = {
            'A': 'A', 'B': 'Bê', 'C': 'Cê', 'D': 'Dê', 'E': 'É',
            'F': 'Éfe', 'G': 'Gê', 'H': 'Agá', 'I': 'I', 'J': 'Jota',
            'K': 'Cá', 'L': 'Éle', 'M': 'Ême', 'N': 'Ene', 'O': 'Ó',
            'P': 'Pê', 'Q': 'Quê', 'R': 'Érre', 'S': 'Esse', 'T': 'Tê',
            'U': 'U', 'V': 'Vê', 'W': 'Dáblio', 'X': 'Xis', 'Y': 'Ípsilon', 'Z': 'Zê'
        }
        
        letter_text = letter_names.get(letter.upper(), letter)
        self.speak(f"Letra {letter_text}")
