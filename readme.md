---
title: "Transcriptor de Fluidez Verbal"
emoji: "🗣️"
colorFrom: "blue"
colorTo: "purple"
sdk: "gradio"
python_version: "3.10"
sdk_version: "3.38.0"
suggested_hardware: "t4-small"
suggested_storage: "small"
app_file: "app.py"
short_description: "Una app para transcripción de audio con timestamps usando modelos Whisper."
tags:
  - speech-recognition
  - transcription
  - whisper
  - gradio
---

# Transcriptor de Fluidez Verbal

Esta aplicación utiliza Gradio y modelos Whisper para transcribir audio con timestamps.
##
---

## Cómo usar

1. Subí o grabá un archivo de audio.
2. La app transcribirá el audio mostrando texto y tiempos.
3. Ideal para análisis de fluidez verbal.

---

## Dependencias

- gradio
- transformers
- torch

---

## Referencias

- Hugging Face Spaces: https://huggingface.co/docs/hub/spaces
- Modelo Whisper: https://huggingface.co/openai/whisper-large-v2
