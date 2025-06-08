import torch
import gradio as gr
from faster_whisper import WhisperModel
import pandas as pd
import tempfile

model_size = "large-v2"
model = WhisperModel(model_size, device="cuda" if torch.cuda.is_available() else "cpu")

def procesar_audio(audio_file):
    if audio_file is None:
        raise gr.Error("No audio file submitted! Please upload or record an audio file before submitting your request.")
    
    segments, info = model.transcribe(audio_file, beam_size=5, word_timestamps=True)

    output_text = ""
    output_timestamps = "Timestamps por palabra:\n"

    word_timings = []
    words_data = []

    for segment in segments:
        output_text += segment.text + " "
        for word_info in segment.words:
            start = word_info.start
            end = word_info.end
            word = word_info.word
            duration = round(end - start, 2)

            word_timings.append({
                "word": word,
                "start": start,
                "end": end,
                "duration": duration
            })

            output_timestamps += f"[{start:.2f}s - {end:.2f}s]: {word}\n"

            words_data.append({
                "Palabra": word,
                "Inicio (s)": round(start, 2),
                "Fin (s)": round(end, 2),
                "Duración (s)": duration
            })

    # Crear DataFrame
    df = pd.DataFrame(words_data)

    # Guardar CSV temporal para descarga
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(tmp_file.name, index=False)

    # Retornamos texto, json, tabla (lista de listas) y archivo CSV
    table_data = df.values.tolist()
    return output_text.strip() + "\n\n" + output_timestamps, word_timings, table_data, tmp_file.name

iface = gr.Interface(
    fn=procesar_audio,
    inputs=gr.Audio(type="filepath", label="Subí tu archivo de audio o grabá con el micrófono"),
    outputs=[
        gr.Textbox(label="Texto transcripto"),
        gr.JSON(label="Tiempos de palabras (para análisis)"),
        gr.Dataframe(
            headers=["Palabra", "Inicio (s)", "Fin (s)", "Duración (s)"],
            label="Tabla de tiempos de palabras",
            interactive=True
        ),
        gr.File(label="Descargar CSV")
    ],
    title="Transcriptor de audio para análisis de fluidez verbal",
    submit_btn="Transcribir Audio",
    clear_btn="Limpiar"
)

iface.launch()
