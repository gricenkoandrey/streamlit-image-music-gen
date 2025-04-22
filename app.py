import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from magenta.models.music_vae import TrainedModel
from magenta.models.music_vae import configs
import tensorflow.compat.v1 as tf
import magenta.music
tf.disable_v2_behavior()

# Загрузка модели Stable Diffusion для генерации изображений
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe.to("cuda")  # Использование GPU, или замените на "cpu", если у вас нет GPU

# Загрузка модели MusicVAE для генерации музыки
config_name = 'cat-mel_2bar_big'
config = configs.CONFIG_MAP[config_name]
# Убедитесь, что указали правильный путь к контрольной точке
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
