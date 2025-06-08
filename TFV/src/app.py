import torch
import gradio as gr
from faster_whisper import WhisperModel
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_read

import tempfile
import os


model_size = "large-v2"
BATCH_SIZE = 8
FILE_LIMIT_MB = 1000

model = WhisperModel(model_size, device="cuda" if torch.cuda.is_available() else "cpu")


def procesar_audio(audio_file):
    if audio_file is None:
        raise gr.Error("No audio file submitted! Please upload or record an audio file before submitting your request.")
    
    segments, info = model.transcribe(audio_file, beam_size=5, word_timestamps=True)

    output_text = ""
    output_timestamps = "Timestamps por palabra:\n"

    word_timings = []
    words_data = []   # ✅ Declaramos la lista acá

    for segment in segments:
        output_text += segment.text + " "
        for word_info in segment.words:
            start = word_info.start
            end = word_info.end
            word = word_info.word

            # Para el JSON
            word_timings.append({
                "word": word,
                "start": start,
                "end": end,
                "duration": end - start
            })

            # Para el string de timestamps
            output_timestamps += f"[{start:.2f}s - {end:.2f}s]: {word}\n"

            # ✅ Para la Dataframe
            words_data.append({
                "Palabra": word,
                "Inicio (s)": start,
                "Fin (s)": end
            })

    table_data = []
    for wd in words_data:
         table_data.append([wd["Palabra"], wd["Inicio (s)"], wd["Fin (s)"]])

    # Retornamos todo: texto, json, dataframe
    return output_text.strip() + "\n\n" + output_timestamps, word_timings, table_data


iface = gr.Interface(
        fn=procesar_audio,
        inputs=gr.Audio(type="filepath", label="Subí tu archivo de audio o grabá con el micrófono"),
        outputs=[
            gr.Textbox(label="Texto transcripto"),
            gr.JSON(label="Tiempos de palabras (para análisis)"),
            gr.Dataframe(
            headers=["Palabra", "Inicio (s)", "Fin (s)"],
            label="Tabla de tiempos de palabras",
            interactive=True  # Esto es lo que hace que se pueda editar
                        )
        ],
        title="Transcriptor de audio para análisis de fluidez verbal",
        submit_btn="Transcribir Audio",
        clear_btn="Limpiar"
    )


iface.launch()


