import torch
import gradio as gr
from faster_whisper import WhisperModel
import pandas as pd
import tempfile
from collections import Counter

model_size = "large-v2"
model = WhisperModel(model_size, device="cuda" if torch.cuda.is_available() else "cpu")

def procesar_audio(audio_file):
    if audio_file is None:
        raise gr.Error("No audio file submitted! Please upload or record an audio file before submitting your request.")
    
    segments, info = model.transcribe(audio_file, beam_size=5, word_timestamps=True)

    output_text = ""
    output_timestamps = "Timestamps por palabra (globales):\n"

    words_data = []

    for segment in segments:
        output_text += segment.text + " "
        for word_info in segment.words:
            absolute_start = segment.start + word_info.start
            absolute_end = segment.start + word_info.end
            duration = round(absolute_end - absolute_start, 2)

            output_timestamps += f"[{absolute_start:.2f}s - {absolute_end:.2f}s]: {word_info.word}\n"

            words_data.append({
                "Palabra": word_info.word,
                "Inicio (s)": round(absolute_start, 2),
                "Fin (s)": round(absolute_end, 2),
                "Duración (s)": duration
            })

    df = pd.DataFrame(words_data)

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(tmp_file.name, index=False)

    table_data = df.values.tolist()
    return output_text.strip() + "\n\n" + output_timestamps, table_data, tmp_file.name

def calcular_estadisticas(tabla_data):
    # tabla_data es lista de listas: [[Palabra, Inicio, Fin, Duración], ...]
    if tabla_data is None or len(tabla_data) == 0:
        return "Tiempo hasta la primera palabra: N/A\nCantidad total de palabras: 0\nCantidad de palabras repetidas: 0\nPalabras repetidas: Ninguna"
    
    df = pd.DataFrame(tabla_data, columns=["Palabra", "Inicio (s)", "Fin (s)", "Duración (s)"])

    # Tiempo hasta la primera palabra
    tiempo_primera_palabra = df["Inicio (s)"].min()

    # Cantidad total de palabras
    total_palabras = len(df)

    # Conteo de palabras en minúscula
    conteo_palabras = Counter(df["Palabra"].str.lower())

    # Cantidad de palabras repetidas (apariciones > 1)
    repetidas = sum(1 for c in conteo_palabras.values() if c > 1)

    # Palabras repetidas con sus conteos
    palabras_repetidas_detalle = [f"'{palabra}': {count} veces" for palabra, count in conteo_palabras.items() if count > 1]
    detalle_repetidas = ", ".join(palabras_repetidas_detalle) if palabras_repetidas_detalle else "Ninguna"

    texto = (
        f"Tiempo hasta la primera palabra: {tiempo_primera_palabra:.2f} segundos\n"
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

    transcribe_btn.click(
        fn=procesar_audio,
        inputs=[audio_input],
        outputs=[trans_text, tabla, csv_output]
    )

    tabla.change(
        fn=calcular_estadisticas,
        inputs=[tabla],
        outputs=[estadisticas]
    )

    clear_btn.click(
        lambda: ("", [], None, ""),
        inputs=[],
        outputs=[trans_text, tabla, csv_output, estadisticas]
    )

demo.launch()
