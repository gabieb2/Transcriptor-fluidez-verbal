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

#Definimos parámetros del modelo 
#pipe = pipeline(
#    task="automatic-speech-recognition",
#    chunk_length_s=30,
#    device=device,
#    return_timestamps=True
#)



def procesar_audio(audio_file,):
    if audio_file is None:
        raise gr.Error("No audio file submitted! Please upload or record an audio file before submitting your request.")
      
    segments, info = model.transcribe(audio_file, beam_size=5, word_timestamps=True)
# Construimos el texto completo y los timestamps por palabra
    output_text = ""
    output_timestamps = "Timestamps por palabra:\n"
    
    for segment in segments:
        output_text += segment.text + " "
        for word_info in segment.words:
            start = word_info.start
            end = word_info.end
            word = word_info.word
            output_timestamps += f"[{start:.2f}s - {end:.2f}s]: {word}\n"
    
    return output_text.strip() + "\n\n" + output_timestamps

     

iface = gr.Interface(
    fn=procesar_audio,
    inputs=gr.Audio(type="filepath", label="Subí tu archivo de audio o grabá con el micrófono"),
    outputs="text",
    title="Procesador de Audio",
    submit_btn="Transcribir Audio"
)

iface.launch()



#
# def transcribe(inputs, task):
#     if inputs is None:
#         raise gr.Error("No audio file submitted! Please upload or record an audio file before submitting your request.")
#
#     text = pipe(inputs, batch_size=BATCH_SIZE, generate_kwargs={"task": task}, return_timestamps=True)["text"]
#     return  text


# demo = gr.Blocks()

# mf_transcribe = gr.Interface(
#     fn=transcribe,
#     inputs=[
#         gr.inputs.Audio(source="microphone", type="filepath", optional=True),
#         gr.inputs.Radio(["transcribe", "translate"], label="Task", default="transcribe"),
#     ],
#     outputs="text",
#     layout="horizontal",
#     theme="huggingface",
#     title="Whisper Large V3: Transcribe Audio",
#     description=(
#         "Transcribe long-form microphone or audio inputs with the click of a button! Demo uses the OpenAI Whisper"
#         f" checkpoint [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME}) and 🤗 Transformers to transcribe audio files"
#         " of arbitrary length."
#     ),
#     allow_flagging="never",
# )

# file_transcribe = gr.Interface(
#     fn=transcribe,
#     inputs=[
#         gr.inputs.Audio(source="upload", type="filepath", optional=True, label="Audio file"),
#         gr.inputs.Radio(["transcribe", "translate"], label="Task", default="transcribe"),
#     ],
#     outputs="text",
#     layout="horizontal",
#     theme="huggingface",
#     title="Whisper Large V3: Transcribe Audio",
#     description=(
#         "Transcribe long-form microphone or audio inputs with the click of a button! Demo uses the OpenAI Whisper"
#         f" checkpoint [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME}) and 🤗 Transformers to transcribe audio files"
#         " of arbitrary length."
#     ),
#     allow_flagging="never",
# )


# with demo:
#     gr.TabbedInterface([mf_transcribe, file_transcribe, yt_transcribe], ["Microphone", "Audio file", "YouTube"])

# demo.launch(enable_queue=True)
