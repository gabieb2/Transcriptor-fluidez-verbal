import torch
import gradio as gr
from faster_whisper import WhisperModel
import pandas as pd
import tempfile
import torchaudio
import torchaudio.transforms as T
import librosa
import noisereduce as nr
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter  # <-- Importamos collections aquí

# Cargar modeloss
model_size = "large-v2"
model = WhisperModel(model_size, device="cuda" if torch.cuda.is_available() else "cpu")

vad_model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    trust_repo=True
)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

def filtrar_audio(audio_file):
    wav, sr = librosa.load(audio_file, sr=None)
    wav_denoised = nr.reduce_noise(y=wav, sr=sr)
    sf.write("limpio.wav", wav_denoised, sr)
    return "limpio.wav"

def detectar_inicio_voz_silero(audio_file):
    wav, sr = torchaudio.load(audio_file)
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != 16000:
        wav = T.Resample(sr, 16000)(wav)
        sr = 16000
    wav = wav.squeeze()
    speech_ts = get_speech_timestamps(
        wav, vad_model, sampling_rate=sr,
        threshold=0.3, min_speech_duration_ms=50,
        min_silence_duration_ms=50, window_size_samples=512
    )
    return speech_ts, sr, wav.numpy()

# Transcripción completa
def procesar_audio(audio_file):
    if audio_file is None:
        raise gr.Error("Subí un archivo primero.")
    clean = filtrar_audio(audio_file)
    segments, _ = model.transcribe(clean, beam_size=5, word_timestamps=True)

    text = ""
    table = []
    for seg in segments:
        text += seg.text + " "
        for w in seg.words:
            start = seg.start + w.start
            end   = seg.start + w.end
            table.append([w.word, round(start,2), round(end,2), round(end-start,2)])

    return text.strip(), table

# Solo latencia + waveform plot + latencia texto
def plot_vad(audio_file):
    if audio_file is None:
        raise gr.Error("Subí un archivo primero.")
    clean = filtrar_audio(audio_file)
    speech_ts, sr, wav = detectar_inicio_voz_silero(clean)

    # Calcular latencia (inicio primera voz detectada)
    if speech_ts:
        latencia = speech_ts[0]['start'] / sr
    else:
        latencia = 0.0

    # Generar figura
    fig, ax = plt.subplots()
    times = np.linspace(0, len(wav)/sr, num=len(wav))
    ax.plot(times, wav)
    for seg in speech_ts:
        start = seg['start']/sr
        end   = seg['end']/sr
        ax.axvspan(start, end, alpha=0.3)  # sombrear segmento
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud")
    ax.set_title("Forma de onda + Segmentos de Voz Detectados")
    return fig, f"Latencia detectada: {latencia:.2f} segundos"

# Función para exportar CSV desde tabla editable, con nombre personalizado e incluir estadísticas
def exportar_csv(tabla_data, nombre_archivo, estadisticas_texto):
    import pandas as pd

    if tabla_data is None or len(tabla_data) == 0:
        return None
    if not nombre_archivo or nombre_archivo.strip() == "":
        nombre_archivo = "tabla_palabras"
    if not nombre_archivo.endswith(".csv"):
        nombre_archivo += ".csv"

    df = pd.DataFrame(tabla_data, columns=["Palabra","Inicio (s)","Fin (s)","Duración (s)"]) if not isinstance(tabla_data, pd.DataFrame) else tabla_data
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix=nombre_archivo.replace(".csv","")+"_")

    with open(tmp.name, 'w', encoding='utf-8') as f:
        df.to_csv(f, index=False)
        f.write("\n")  # línea en blanco
        f.write("ESTADÍSTICAS\n")
        for line in estadisticas_texto.split('\n'):
            f.write(line + "\n")

    return tmp.name

# Función para calcular estadísticas a partir de la tabla editable y latencia
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
    gr.Markdown("## Transcriptor y Analizador para evaluaciones congnitivas de Fluidez Verbal ")

    audio_in = gr.Audio(type="filepath", label="Subí o grabá un audio")

    with gr.Row():
        btn_trans = gr.Button("Transcribir")
        btn_vad   = gr.Button("Detectar Latencia")

    out_text   = gr.Textbox(label="Transcripción", interactive=False)
    latencia_text = gr.Textbox(label="Latencia detectada", interactive=False)
    estadisticas = gr.Textbox(label="Estadísticas", interactive=False)

    out_table  = gr.Dataframe(headers=["Palabra","Inicio (s)","Fin (s)","Duración (s)"], interactive=True)
    out_plot   = gr.Plot()

    gr.Markdown("### Exportar tabla modificada a CSV")
    nombre_csv = gr.Textbox(label="Nombre del archivo CSV", placeholder="ejemplo.csv")
    btn_export = gr.Button("Crear CSV")
    out_csv    = gr.File(label="Descargar CSV")

    tiempo_vad_state = gr.State(0.0)

    btn_trans.click(fn=procesar_audio, inputs=[audio_in], 
                    outputs=[out_text, out_table])

    btn_vad.click(fn=plot_vad, inputs=[audio_in], 
                  outputs=[out_plot, latencia_text])

    btn_export.click(fn=exportar_csv, 
                     inputs=[out_table, nombre_csv, estadisticas], 
                     outputs=[out_csv])

    # Actualizar estadísticas al cambiar tabla o latencia
    out_table.change(fn=calcular_estadisticas, inputs=[out_table, tiempo_vad_state], outputs=[estadisticas])

    # Extraer latencia numérica del texto para usar en estadísticas
    def guardar_latencia(texto_latencia):
        try:
            return float(texto_latencia.split(": ")[1].split()[0])
        except:
            return 0.0

    latencia_text.change(fn=guardar_latencia, inputs=[latencia_text], outputs=[tiempo_vad_state])

    # También actualizar estadísticas cuando cambia latencia
    latencia_text.change(
        fn=lambda x: calcular_estadisticas(out_table.value, guardar_latencia(x)),
        inputs=[latencia_text],
        outputs=[estadisticas]
    )

    gr.Button("Limpiar").click(
        lambda: ("", [], None, None, "", None, 0.0), 
        inputs=[], 
        outputs=[out_text, out_table, out_csv, out_plot, nombre_csv, estadisticas, tiempo_vad_state]
    )

demo.launch()
