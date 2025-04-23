# import streamlit as st
# from openai import OpenAI
# from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
# from dotenv import load_dotenv
# import os
# from rag_utils import process_pdf_and_ask


# # Load environment variables
# load_dotenv()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# # Load BERT model and tokenizer for intent classification
# @st.cache_resource
# def load_intent_model():
#     model = AutoModelForSequenceClassification.from_pretrained("yeniguno/bert-uncased-intent-classification")
#     tokenizer = AutoTokenizer.from_pretrained("yeniguno/bert-uncased-intent-classification")
#     return pipeline("text-classification", model=model, tokenizer=tokenizer)

# intent_pipe = load_intent_model()

# # === Streamlit UI ===
# st.title("🔍 Intent, Sentiment & PDF Analysis Bot")
# st.write("This bot can detect **intent**, **sentiment**, and also analyze content from uploaded **PDFs** using RAG + LLM.")

# # Input text area
# st.header("📨 Text-Based Analysis")
# user_input = st.text_area("Enter a message for intent and sentiment analysis", "", height=100)

# if st.button("Analyze Text"):
#     if user_input.strip() == "":
#         st.warning("⚠️ Please enter some text to analyze.")
#     else:
#         try:
#             # Intent classification
#             intent_result = intent_pipe(user_input)
#             intent_label = intent_result[0]["label"]

#             # Sentiment classification using OpenAI
#             response = client.chat.completions.create(
#                 model="gpt-3.5-turbo",
#                 messages=[
#                     {"role": "system", "content": "You are a sentiment analysis assistant. Respond with 'Positive', 'Negative', or 'Neutral' and give a brief reason."},
#                     {"role": "user", "content": user_input}
#                 ]
#             )
#             sentiment = response.choices[0].message.content.strip()

#             st.subheader("🧠 Detected Intent:")
#             st.info(intent_label)

#             st.subheader("💬 Predicted Sentiment:")
#             st.success(sentiment)

#         except Exception as e:
#             st.error(f"❌ Error: {e}")

# # === PDF Upload and RAG Section ===
# st.header("📄 Upload PDF for RAG + LLM Answering")
# pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# if pdf_file:
#     question = st.text_input("Ask a question based on the PDF content:")
#     if st.button("Ask PDF"):
#         if question.strip() == "":
#             st.warning("⚠️ Please enter a question for the uploaded PDF.")
#         else:
#             try:
#                 answer = process_pdf_and_ask(pdf_file, question)
#                 st.subheader("📚 Answer from Document:")
#                 st.success(answer)
#             except Exception as e:
#                 st.error(f"❌ Error processing PDF: {e}")






import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from dotenv import load_dotenv
import os
import tempfile
import base64
from rag_utils import process_pdf_and_ask, rag_answer_from_text
from audio_utils import transcribe_audio, text_to_speech
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load BERT intent classification model
@st.cache_resource
def load_intent_model():
    model = AutoModelForSequenceClassification.from_pretrained("yeniguno/bert-uncased-intent-classification")
    tokenizer = AutoTokenizer.from_pretrained("yeniguno/bert-uncased-intent-classification")
    return pipeline("text-classification", model=model, tokenizer=tokenizer)

intent_pipe = load_intent_model()

# === UI ===
st.title("🎙️ Multimodal RAG + Intent Bot")
st.write("Handles **text**, **audio**, and **PDFs** for intent, sentiment, and document Q&A.")

# === PDF Upload + Ask ===
st.header("📄 Upload PDF for Question Answering")
pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if pdf_file:
    st.success("✅ PDF uploaded. Ask a question via text or record your voice.")
    st.markdown("### ✍️ Type Your Question")
    question = st.text_input("Ask a question based on the PDF content:")

    st.markdown("### 🎤 Or Record Your Question")
    components.html(open("audio_recorder.html", "r").read(), height=300)

    uploaded_audio = st.file_uploader("Or upload recorded audio (WAV)", type=["wav"])

    final_question = question

    if st.button("Ask PDF"):
        if uploaded_audio:
            with st.spinner("🔊 Transcribing audio..."):
                final_question = transcribe_audio(uploaded_audio)
                st.info(f"Transcribed audio: {final_question}")
        if not final_question.strip():
            st.warning("⚠️ Please provide a text or audio question.")
        else:
            try:
                answer = process_pdf_and_ask(pdf_file, final_question)
                st.subheader("📚 Answer from Document:")
                st.success(answer)
                st.audio(text_to_speech(answer), format="audio/mp3")
            except Exception as e:
                st.error(f"❌ Error processing PDF: {e}")

# === Text Input ===
st.header("📝 Intent + Sentiment Analysis (Text Only)")
user_input = st.text_area("Enter a message", "", height=100)

if st.button("Analyze Text"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        try:
            intent_result = intent_pipe(user_input)
            intent_label = intent_result[0]["label"]

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a sentiment analysis assistant. Respond with 'Positive', 'Negative', or 'Neutral' and give a brief reason."},
                    {"role": "user", "content": user_input}
                ]
            )
            sentiment = response.choices[0].message.content.strip()

            st.subheader("🧠 Detected Intent:")
            st.info(intent_label)

            st.subheader("💬 Predicted Sentiment:")
            st.success(sentiment)

        except Exception as e:
            st.error(f"❌ Error: {e}")

