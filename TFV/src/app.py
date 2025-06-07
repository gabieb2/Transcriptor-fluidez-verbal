import gradio as gr
import whisper

#Definimos parámetros de whisper
model = whisper.load_model("turbo")


def procesar_audio(audio_file):
    return 

iface = gr.Interface(
    fn=procesar_audio,
    inputs=gr.Audio(type="filepath", label="Subí un archivo de audio"),
    outputs="text",
    title="Procesador de Audio"
)

iface.launch()