import sys
import os
import importlib.util
import subprocess
import time
import torch
import numpy as np
import soundfile as sf

from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import StableDiffusionPipeline
from moviepy.editor import ImageSequenceClip, AudioFileClip

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QTextEdit,
    QPushButton, QFileDialog, QSplashScreen
)
from PySide6.QtGui import QPixmap, QColor, QFont, QPalette
from PySide6.QtCore import Qt

# --- Gerekli kütüphane kontrolü ---
def check_and_install():
    required_libs = ["PySide6", "transformers", "torch", "moviepy", "Pillow", "diffusers", "numpy", "soundfile"]
    for lib in required_libs:
        if importlib.util.find_spec(lib) is None:
            print(f"Yükleniyor: {lib}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

# --- Model Yükleyici ---
class ModelYukleyici:
    def __init__(self):
        self.model_name = "CerebrumTech/cere-llama-3-8b-tr"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None

    def load_model(self):
        if self.model is None:
            print("Model yükleniyor...")
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map="auto"
            )
            print("Model yüklendi.")
        return self.model, self.tokenizer

    def generate(self, prompt, max_length=100):
        if self.model is None:
            self.load_model()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_length)
        cevap = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return cevap

# --- Görsel Üretici ---
class ImageGenerator:
    def __init__(self):
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.pipeline = self.pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

    def generate_images(self, prompt, num_images=1, output_dir="images"):
        os.makedirs(output_dir, exist_ok=True)
        result = self.pipeline(prompt, num_inference_steps=50, num_images_per_prompt=num_images)
        images = result.images
        paths = []
        for idx, image in enumerate(images):
            path = os.path.join(output_dir, f"image_{idx+1}.png")
            image.save(path)
            paths.append(path)
        return paths

# --- Müzik Üretici ---
class MusicGenerator:
    def __init__(self):
        pass

    def generate(self, prompt, duration=5, output_path="output.wav"):
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        frequency = 440
        audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
        sf.write(output_path, audio_data, sample_rate)
        return output_path

# --- Video Oluşturucu ---
class VideoOlusturucu:
    def __init__(self):
        pass

    def create_video(self, image_paths, audio_path, output_path="output_video.mp4", fps=1):
        clip = ImageSequenceClip(image_paths, fps=fps)
        if audio_path:
            audio = AudioFileClip(audio_path)
            clip = clip.set_audio(audio)
        clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
        return output_path

# --- Ana Arayüz ---
class MainWindow(QMainWindow):
    def __init__(self, model_loader, image_generator, music_generator, video_creator):
        super().__init__()
        self.model_loader = model_loader
        self.image_generator = image_generator
        self.music_generator = music_generator
        self.video_creator = video_creator

        self.setWindowTitle("AI Multimedya Masaüstü Uygulaması")
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        layout = QVBoxLayout()

        self.prompt_label = QLabel("Soru / İstek:")
        self.prompt_text = QTextEdit()
        self.prompt_text.setFixedHeight(100)
        layout.addWidget(self.prompt_label)
        layout.addWidget(self.prompt_text)

        self.answer_label = QLabel("Model Yanıtı:")
        self.answer_text = QTextEdit()
        self.answer_text.setReadOnly(True)
        layout.addWidget(self.answer_label)
        layout.addWidget(self.answer_text)

        self.generate_video_button = QPushButton("Video Oluştur")
        self.generate_video_button.clicked.connect(self.on_generate_video)
        layout.addWidget(self.generate_video_button)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def on_generate_video(self):
        prompt = self.prompt_text.toPlainText()
        if not prompt.strip():
            return

        answer = self.model_loader.generate(prompt)
        self.answer_text.setText(answer)

        image_paths = self.image_generator.generate_images(prompt, num_images=3)
        music_path = self.music_generator.generate(prompt, duration=5)

        output_path = "output_video.mp4"
        self.video_creator.create_video(image_paths, music_path, output_path)

        dlg = QFileDialog(self)
        dlg.setWindowTitle("Video kaydedildi")
        dlg.setFileMode(QFileDialog.AnyFile)
        dlg.setNameFilters(["MP4 Files (*.mp4)"])
        dlg.selectFile(output_path)
        dlg.exec()

# --- Ana Fonksiyon ---
def main():
    check_and_install()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    dark_palette = QPalette()
    dark_color = QColor(45, 45, 45)
    disabled_color = QColor(127, 127, 127)
    dark_palette.setColor(QPalette.Window, dark_color)
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(30, 30, 30))
    dark_palette.setColor(QPalette.AlternateBase, dark_color)
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Disabled, QPalette.Text, disabled_color)
    dark_palette.setColor(QPalette.Button, dark_color)
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.Disabled, QPalette.ButtonText, disabled_color)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Disabled, QPalette.Highlight, QColor(80, 80, 80))
    dark_palette.setColor(QPalette.HighlightedText, Qt.white)
    dark_palette.setColor(QPalette.Disabled, QPalette.HighlightedText, disabled_color)
    app.setPalette(dark_palette)

    pixmap = QPixmap(400, 300)
    pixmap.fill(Qt.black)
    splash = QSplashScreen(pixmap)
    splash.setFont(QFont('Arial', 16))
    splash.showMessage("Yükleniyor...", Qt.AlignBottom | Qt.AlignCenter, Qt.white)
    splash.show()
    app.processEvents()

    model_loader = ModelYukleyici()
    model_loader.load_model()
    image_gen = ImageGenerator()
    music_gen = MusicGenerator()
    video_gen = VideoOlusturucu()

    time.sleep(1)
    splash.finish(None)

    window = MainWindow(model_loader, image_gen, music_gen, video_gen)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
