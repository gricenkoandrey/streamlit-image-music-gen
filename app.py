import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from magenta.models.music_vae import TrainedModel
from magenta.models.music_vae import configs
import tensorflow.compat.v1 as tf
import magenta.music
import gdown
import os

# Отключаем TensorFlow 2.x, чтобы использовать 1.x
tf.disable_v2_behavior()

# Функция для скачивания модели, если она ещё не была скачана
def download_model_if_needed():
    # Укажите путь, куда нужно скачать файлы
    model_dir = "models/music_vae"
    os.makedirs(model_dir, exist_ok=True)
    
    files = {
        "cat-mel_2bar_big.ckpt": "19hgLTFeQ5MypjoEI_uq41cC6PbJ7SgB_",  # замените на ваш ID для .ckpt
        "cat-mel_2bar_big.ckpt.index": "1YBbzkRvwhfuhKZ1RZVML4Qy5avRIDxdY",       # замените на ваш ID для .index
    }
    
    # Загружаем каждый файл
    for filename, file_id in files.items():
        file_path = os.path.join(model_dir, filename)
        if not os.path.exists(file_path):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, file_path, quiet=False)

# Загрузка модели Stable Diffusion для генерации изображений
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe.to("cuda")  # Использование GPU, или замените на "cpu", если у вас нет GPU

# Загрузка модели MusicVAE для генерации музыки
download_model_if_needed()  # Загружаем модель MusicVAE, если её нет
config_name = 'cat-mel_2bar_big'
config = configs.CONFIG_MAP[config_name]
music_model = TrainedModel(config, batch_size=1, checkpoint_dir_or_path='models/music_vae/cat-mel_2bar_big.ckpt')

def generate_image(prompt):
    """Генерация изображения на основе текстового описания"""
    image = pipe(prompt).images[0]
    return image

def generate_music():
    """Генерация музыки с помощью модели MusicVAE"""
    z = music_model.sample(n=1, length=80, temperature=0.5)
    return z

# Интерфейс Streamlit
st.title("Генератор художественных произведений и музыки")

# Генерация изображений
st.header("Генерация изображений")
prompt = st.text_input("Введите описание для изображения:")
if prompt:
    st.text("Генерация изображения...")
    image = generate_image(prompt)
    st.image(image)

# Генерация музыки
st.header("Генерация музыки")
music_prompt = st.text_input("Введите описание для музыки:")
if music_prompt:
    st.text("Генерация музыки...")
    music = generate_music()
    music_path = 'generated_music.mid'
    magenta.music.sequence_proto_to_midi(music, music_path)
    st.text(f"Музыка сгенерирована и сохранена в {music_path}")
    st.audio(music_path)

