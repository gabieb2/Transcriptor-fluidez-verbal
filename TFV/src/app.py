import torch
import gradio as gr
from faster_whisper import WhisperModel
import pandas as pd
import tempfile
import torchaudio
import torchaudio.transforms as T
from collections import Counter
import librosa
import noisereduce as nr
import soundfile as sf


#Importación de modelo whisper
model_size = "large-v2"
model = WhisperModel(model_size, device="cuda" if torch.cuda.is_available() else "cpu")

vad_model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    trust_repo=True
)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

#Método para detectar latencia usando el modelo Silero VAD
def detectar_inicio_voz_silero(audio_file):
    wav, sample_rate = torchaudio.load(audio_file)

    # Pasar a mono si tiene más de un canal
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)

    # Remuestrear a 16000 Hz si es necesario
    if sample_rate != 16000:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
        wav = resampler(wav)
        sample_rate = 16000

    wav = wav.squeeze()
    speech_timestamps = get_speech_timestamps(
        wav,
        vad_model,
        sampling_rate=sample_rate,
        threshold=0.3,
        min_speech_duration_ms=50,
        min_silence_duration_ms=50,
        window_size_samples=512
    )
    if speech_timestamps:
        return speech_timestamps[0]['start'] / sample_rate
    else:
        return 0.0
    
#Función para el filtrado de audio, eliminando ruidos de fondo

def filtrar_audio(audio_file):
    wav, sample_rate = librosa.load(audio_file, sr=None)

    # Aplicar reducción de ruido
    reducido = nr.reduce_noise(y=wav, sr=sample_rate)

    # Guardar resultado
    sf.write("limpio.wav", reducido, sample_rate)

    return "limpio.wav"


#Defino función pricipal para procesar el audio y transcribirlo
def procesar_audio(audio_file):
    if audio_file is None:
        raise gr.Error("No audio file submitted! Please upload or record an audio file before submitting your request.")

    audio_file = filtrar_audio(audio_file)
    inicio_voz = detectar_inicio_voz_silero(audio_file)

    # Generar salida vacía en texto y tabla, pero con la latencia
    output_text = f"(Inicio de voz detectado por VAD: {inicio_voz:.2f}s)"
    tabla_vacia = []
    archivo_csv = None

    return output_text, tabla_vacia, archivo_csv, inicio_voz

def calcular_estadisticas(tabla_data, tiempo_vad):
    if tabla_data is None or len(tabla_data) == 0:
        return (
            f"Inicio de voz (VAD): {tiempo_vad:.2f} segundos\n"
            "Tiempo hasta la primera palabra: N/A\n"
            "Cantidad total de palabras: 0\n"
            "Cantidad de palabras repetidas: 0\n"
            "Palabras repetidas: Ninguna"
        )
    
    df = pd.DataFrame(tabla_data, columns=["Palabra", "Inicio (s)", "Fin (s)", "Duración (s)"])

    df["Inicio (s)"] = pd.to_numeric(df["Inicio (s)"], errors='coerce')
    df["Fin (s)"] = pd.to_numeric(df["Fin (s)"], errors='coerce')
    df["Duración (s)"] = pd.to_numeric(df["Duración (s)"], errors='coerce')

    df = df.dropna(subset=["Inicio (s)"])

    if df.empty:
        return (
            f"Inicio de voz (VAD): {tiempo_vad:.2f} segundos\n"
            "Tiempo hasta la primera palabra: N/A\n"
            "Cantidad total de palabras: 0\n"
            "Cantidad de palabras repetidas: 0\n"
            "Palabras repetidas: Ninguna"
        )

    
    total_palabras = len(df)
    conteo_palabras = Counter(df["Palabra"].str.lower())
    repetidas = sum(1 for c in conteo_palabras.values() if c > 1)
    palabras_repetidas_detalle = [f"'{palabra}': {count} veces" for palabra, count in conteo_palabras.items() if count > 1]
    detalle_repetidas = ", ".join(palabras_repetidas_detalle) if palabras_repetidas_detalle else "Ninguna"

    texto = (
        f"Latencia de primera palabra: {tiempo_vad:.2f} segundos\n"
        f"Cantidad total de palabras: {total_palabras}\n"
        f"Cantidad de palabras repetidas: {repetidas}\n"
        f"Palabras repetidas: {detalle_repetidas}"
    )
    return texto

with gr.Blocks() as demo:
    gr.Markdown("## Transcriptor de audio para análisis de fluidez verbal")

    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Subí tu archivo de audio o grabá con el micrófono")

    transcribe_btn = gr.Button("Transcribir Audio")

    trans_text = gr.Textbox(label="Texto transcripto")

    tabla = gr.Dataframe(
        headers=["Palabra", "Inicio (s)", "Fin (s)", "Duración (s)"],
        label="Tabla de tiempos de palabras",
        interactive=True
    )

    csv_output = gr.File(label="Descargar CSV")

    clear_btn = gr.Button("Limpiar")

    estadisticas = gr.Textbox(label="Estadísticas", interactive=False)

    tiempo_vad_state = gr.State(0.0)

    transcribe_btn.click(
        fn=procesar_audio,
        inputs=[audio_input],
        outputs=[trans_text, tabla, csv_output, tiempo_vad_state]
    )

    tabla.change(
        fn=calcular_estadisticas,
        inputs=[tabla, tiempo_vad_state],
        outputs=[estadisticas]
    )

    clear_btn.click(
        lambda: ("", [], None, 0.0, ""),
        inputs=[],
        outputs=[trans_text, tabla, csv_output, tiempo_vad_state, estadisticas]
    )

demo.launch()
