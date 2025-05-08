import streamlit as st
import openai
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
from pydub import AudioSegment
import tempfile
from PIL import Image
import fitz  # PyMuPDF
import os
import docx
import pandas as pd
import base64
import io
import numpy as np

# --- Configuration ---
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="OpenAI Assistant", layout="wide")
st.title("🧠 OpenAI Assistant: Voice + Prompt + Files + Vision")

# --- App State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "audio_chunks" not in st.session_state:
    st.session_state.audio_chunks = []
if "user_voice_text" not in st.session_state:
    st.session_state.user_voice_text = ""

# --- Text/Prompt & File Upload ---
st.header("📝 Prompt & File Upload")
prompt = st.text_area("Enter your prompt", height=150)

uploaded_image = st.file_uploader("Upload an image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
uploaded_file = st.file_uploader("Upload a file (PDF, TXT, DOCX, XLSX)", type=["pdf", "txt", "docx", "xlsx"])

file_text = ""
image_bytes = None

def extract_file_text(file):
    text = ""
    if file.type == "application/pdf":
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    elif file.type == "text/plain":
        text = file.read().decode("utf-8")
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        df = pd.read_excel(file)
        text = df.to_string(index=False)
    return text

def encode_image_to_base64(image_bytes, mime_type="image/png"):
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"

if uploaded_image:
    image_bytes = uploaded_image.read()
    image_display = Image.open(io.BytesIO(image_bytes))
    st.image(image_display, caption="Uploaded Image", use_container_width=True)

if uploaded_file:
    file_text = extract_file_text(uploaded_file)
    st.text_area("Extracted File Content", value=file_text, height=200)

# --- Voice Input via WebRTC ---
st.header("🎙️ Voice Recorder")

class AudioProcessor:
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        pcm = frame.to_ndarray()
        st.session_state.audio_chunks.append(pcm)
        return frame

ctx = webrtc_streamer(
    key="speech",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"audio": True, "video": False},
    audio_processor_factory=AudioProcessor
)

if st.button("⏹️ Transcribe Audio") and st.session_state.audio_chunks:
    audio_np = np.concatenate(st.session_state.audio_chunks).astype(np.int16)
    audio = AudioSegment(
        audio_np.tobytes(),
        frame_rate=16000,  # Default fallback sample rate
        sample_width=2,
        channels=1
    )
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        audio.export(f.name, format="wav")
        with open(f.name, "rb") as audio_file:
            transcript = openai.audio.transcriptions.create(model="whisper-1", file=audio_file)
            st.session_state.user_voice_text = transcript.text
            st.markdown(f"**🎤 Transcription:** {st.session_state.user_voice_text}")

# --- Combine Input and Submit ---
if st.button("🚀 Submit to OpenAI"):
    try:
        combined_input = prompt + "\n" + file_text + "\n" + st.session_state.user_voice_text

        if uploaded_image and image_bytes:
            mime_type = uploaded_image.type or "image/png"
            image_base64_url = encode_image_to_base64(image_bytes, mime_type)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": combined_input},
                    {"type": "image_url", "image_url": {
                        "url": image_base64_url,
                        "detail": "high"
                    }}
                ]
            }]
            model = "gpt-4-turbo"
        else:
            messages = [{"role": "user", "content": combined_input}]
            model = "gpt-4"

        st.session_state.messages += messages
        response = openai.chat.completions.create(
            model=model,
            messages=st.session_state.messages,
            max_tokens=1000
        )
        reply = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": reply})

        st.markdown("### 🤖 Assistant Response")
        st.write(reply)

        speech_response = openai.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=reply
        )
        st.audio(speech_response.content, format="audio/mp3")

    except Exception as e:
        st.error(f"❌ Error: {e}")
