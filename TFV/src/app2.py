import gradio as gr
from faster_whisper import WhisperModel

# --- TU CONFIGURACIÓN DE MODELO ---
model_size = "small"
try:
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    print("✅ Conectado a GPU")
except:
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    print("⚠️ Usando CPU (INT8)")

def fono_transcribe(audio):
    if audio is None:
        return "No se recibió audio. Revisa los permisos del micrófono."
    
    # Gradio en Colab pasa la ruta del archivo temporal
    segments, _ = model.transcribe(audio, beam_size=5, language="es")
    
    texto = " ".join([s.text for s in segments])
    
    # Mostrar últimas 20 palabras
    palabras = texto.strip().split()
    return "... " + " ".join(palabras[-20:]) if len(palabras) > 20 else " ".join(palabras)

# --- INTERFAZ ADAPTADA PARA COLAB ---
css = """
.gradio-container { background: linear-gradient(135deg, #FF8008, #FFC837) !important; }
#big-text textarea { 
    font-size: 50px !important; 
    font-weight: 900; 
    text-align: center; 
    color: white !important;
    background: rgba(255,255,255,0.2) !important;
    backdrop-filter: blur(10px);
    border-radius: 20px;
}
"""

with gr.Blocks(css=css) as demo:
    gr.HTML("<h1 style='color:white; text-align:center;'>FON0 AUMENTA2</h1>")
    
    output = gr.Textbox(label="", elem_id="big-text", lines=3)
    
    # En Colab, es vital que el audio no sea 'streaming' para evitar errores de permisos
    audio_input = gr.Audio(
        sources=["microphone"],
        type="filepath",
        label="Pulsa el icono del micro, habla y pulsa el cuadrado para terminar"
    )
    
    btn = gr.Button("TRANSCRIBIR", variant="primary")
    btn.click(fn=fono_transcribe, inputs=audio_input, outputs=output)

# IMPORTANTE: En Colab usa 'inline=False' si el iframe se bloquea, 
# o simplemente ejecútalo y haz clic en el link público que genera.
demo.launch(share=True, debug=True)