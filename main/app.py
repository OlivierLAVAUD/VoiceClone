import warnings
import os
import random
import numpy as np
import torch
import gradio as gr
from chatterbox.tts import ChatterboxTTS
from typing import Optional, Tuple
from datetime import datetime
import soundfile as sf

# Désactivation des warnings spécifiques
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TEXT_LENGTH = 2000
MAX_TEXT_SPLIT = 500
RECORDINGS_DIR = "voice_cloning_recordings"  # Répertoire modifié
DEFAULT_TEXT = """Once when I was six years old I saw a magnificent picture in a book, called True Stories from Nature, about the primeval forest. It was a picture of a boa constrictor in the act of swallowing an animal. Here is a copy of the drawing.
In the book it said: "Boa constrictors swallow their prey whole, without chewing it. After that they are not able to move, and they sleep through the six months that they need for digestion."
I pondered deeply, then, over the adventures of the jungle. And after some work with a colored pencil I succeeded in making my first drawing. My Drawing Number One. It looked something like this:
I showed my masterpiece to the grown-ups, and asked them whether the drawing frightened them. But they answered: "Frighten? Why should any one be frightened by a hat?"
My drawing was not a picture of a hat. It was a picture of a boa constrictor digesting an elephant. But since the grown-ups were not able to understand it, I made another drawing: I drew the inside of the boa constrictor, so that the grown-ups could see it clearly. They always need to have things explained. My Drawing Number Two looked like this:
The grown-ups' response, this time, was to advise me to lay aside my drawings of boa constrictors, whether from the inside or the outside, and devote myself instead to geography, history, arithmetic, and grammar.
That is why, at the age of six, I gave up what might have been a magnificent career as a painter. I had been disheartened by the failure of my Drawing Number One and my Drawing Number Two. Grown-ups never understand anything by themselves, and it is tiresome for children to be always and forever explaining things to them.
So then I chose another profession, and learned to pilot airplanes. I have flown a little over all parts of the world; and it is true that geography has been very useful to me. At a glance I can distinguish China from Arizona. If one gets lost in the night, such knowledge is valuable."""


class TTSService:
    """Encapsulates all TTS related functionality"""
    def __init__(self):
        self.model = None
        
    def load_model(self) -> ChatterboxTTS:
        """Lazy loading of the TTS model"""
        if self.model is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = ChatterboxTTS.from_pretrained(DEVICE)
        return self.model
    
    @staticmethod
    def set_seed(seed: int) -> None:
        """Set all random seeds for reproducibility"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
    
    @staticmethod
    def validate_inputs(text: str, audio_path: Optional[str]) -> Tuple[str, Optional[str]]:
        """Validate inputs before processing"""
        if not text.strip():
            raise gr.Error("🚨 Please enter some text to synthesize")
        if len(text) > MAX_TEXT_LENGTH:
            raise gr.Error(f"📜 Text too long (max {MAX_TEXT_LENGTH} characters). Please split your text into smaller parts.")
        if audio_path and not os.path.exists(audio_path):
            raise gr.Error("🔊 Reference audio file not found")
        return text, audio_path
    
    @staticmethod
    def save_audio(audio: Optional[Tuple[int, np.ndarray]], prefix: str = "reference") -> Optional[str]:
        """Save recorded audio to file and return path"""
        if audio is None:
            return None
            
        sr, data = audio
        os.makedirs(RECORDINGS_DIR, exist_ok=True)
        filename = f"{RECORDINGS_DIR}/{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        sf.write(filename, data, sr)
        return filename
    
    @staticmethod
    def split_long_text(text: str, max_length: int = MAX_TEXT_SPLIT) -> list[str]:
        """Split long text into chunks respecting sentence boundaries"""
        sentences = []
        current_chunk = ""
        
        for sentence in text.split('.'):
            if len(current_chunk) + len(sentence) < max_length:
                current_chunk += sentence + '.'
            else:
                if current_chunk:
                    sentences.append(current_chunk)
                current_chunk = sentence + '.'
        
        if current_chunk:
            sentences.append(current_chunk)
        
        return sentences
    
    def generate_speech(
        self,
        text: str,
        audio_prompt: Optional[Tuple[int, np.ndarray]],
        exaggeration: float,
        temperature: float,
        seed_num: int,
        cfg_weight: float
    ) -> Tuple[int, np.ndarray]:
        """Main generation method"""
        try:
            audio_prompt_path = self.save_audio(audio_prompt, "reference")
            text, audio_prompt_path = self.validate_inputs(text, audio_prompt_path)
            
            if seed_num != 0:
                self.set_seed(int(seed_num))

            model = self.load_model()
            
            # Découpage du texte si nécessaire
            if len(text) > MAX_TEXT_SPLIT:  # Seuil pour le découpage
                text_chunks = self.split_long_text(text)
                full_audio = []
                for chunk in text_chunks:
                    wav = model.generate(
                        chunk,
                        audio_prompt_path=audio_prompt_path,
                        exaggeration=exaggeration,
                        temperature=temperature,
                        cfg_weight=cfg_weight,
                    )
                    full_audio.append(wav.squeeze(0).numpy())
                
                # Concaténation des audios et sauvegarde
                final_audio = np.concatenate(full_audio)
                output_path = self.save_audio((model.sr, final_audio), "output")
                return model.sr, final_audio
            else:
                wav = model.generate(
                    text,
                    audio_prompt_path=audio_prompt_path,
                    exaggeration=exaggeration,
                    temperature=temperature,
                    cfg_weight=cfg_weight,
                )
                # Sauvegarde du fichier de sortie
                output_path = self.save_audio((model.sr, wav.squeeze(0).numpy()), "output")
                return model.sr, wav.squeeze(0).numpy()
        except Exception as e:
            raise gr.Error(f"❌ Generation failed: {str(e)}")

def create_interface() -> gr.Blocks:
    """Create and configure the Gradio interface"""
    tts_service = TTSService()
    
    with gr.Blocks(title="🎤 VoiceClone - Unlimited Chatterbox ", theme="soft") as demo:
        # Header avec emojis
        gr.Markdown("# 🎤 VoiceClone - Unlimited Chatterbox 🎧")
        gr.Markdown("Clone voices and generate speech with AI magic! ✨")
        
        # Input Section
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## ⚙️ Input Parameters")
                
                text_input = gr.Textbox(
                    value=DEFAULT_TEXT,
                    label=f"📝 Text to synthesize (max {MAX_TEXT_LENGTH} chars)",
                    max_lines=10,
                    placeholder="Enter your text here...",
                    interactive=True
                )
                
                with gr.Group():
                    ref_audio = gr.Audio(
                        sources=["upload", "microphone"],
                        type="numpy",
                        label="🎤 Reference Audio (Wav)"
                    )
               
                # Parameter Sliders avec emojis
                exaggeration = gr.Slider(0.25, 2, step=0.05, value=0.5,
                                       label="🎚️ Exaggeration (Neutral = 0.5)")
                cfg_weight = gr.Slider(0.0, 1, step=0.05, value=0.5,
                                     label="⏱️ CFG/Pace Control")
                
                with gr.Accordion("🔧 Advanced Options", open=False):
                    seed_num = gr.Number(value=0, label="🎲 Random seed (0 = random)", precision=0)
                    temp = gr.Slider(0.05, 5, step=0.05, value=0.8,
                                   label="🌡️ Temperature (higher = more random)")
                
                generate_btn = gr.Button("✨ Generate Speech", variant="primary")

            # Output Section
            with gr.Column(scale=1):
                gr.Markdown("## 🔊 Output")
                audio_output = gr.Audio(label="🎧 Generated Speech", interactive=True)
                gr.Markdown("""
                **💡 Tips:** 
                - For best results, use clear reference audio under 10 seconds ⏱️
                - For long texts (>500 chars), the system will automatically split the text ✂️
                - All generated files are saved in the 'voice_cloning_recordings' folder 📁
                """)

        # Event Handling
        generate_btn.click(
            fn=tts_service.generate_speech,
            inputs=[
                text_input,
                ref_audio,
                exaggeration,
                temp,
                seed_num,
                cfg_weight,
            ],
            outputs=audio_output,
            api_name="generate"
        )

    return demo

if __name__ == "__main__":
    # Création du répertoire d'enregistrement s'il n'existe pas
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    
    app = create_interface()
    app.queue(max_size=10).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )