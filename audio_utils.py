# import openai
# from dotenv import load_dotenv
# import os
# import tempfile
# from gtts import gTTS

# load_dotenv()
# client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# def transcribe_audio(audio_file):
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
#         tmp.write(audio_file.read())
#         tmp_path = tmp.name

#     with open(tmp_path, "rb") as f:
#         transcript = client.audio.transcriptions.create(
#             model="whisper-1",
#             file=f
#         )
#     return transcript.text

# def text_to_speech(text):
#     tts = gTTS(text)
#     temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
#     tts.save(temp_path)
#     return temp_path



import openai
import tempfile
from dotenv import load_dotenv
import os
from gtts import gTTS

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def transcribe_audio(audio_file):
    # `audio_file` is a file-like object (e.g., from Streamlit or a download)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    with open(tmp_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
    return transcript.text

def text_to_speech(text):
    tts = gTTS(text)
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    tts.save(temp_path)
    return temp_path

