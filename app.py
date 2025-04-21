import streamlit as st
import openai
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# Load OpenAI API key from Streamlit secrets
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Load BERT model and tokenizer for intent classification
intent_model = AutoModelForSequenceClassification.from_pretrained("yeniguno/bert-uncased-intent-classification")
intent_tokenizer = AutoTokenizer.from_pretrained("yeniguno/bert-uncased-intent-classification")
intent_pipe = pipeline("text-classification", model=intent_model, tokenizer=intent_tokenizer)

# === Streamlit UI ===
st.title("🔍 Intent & Sentiment Analysis Bot")
st.write("Enter a message to understand the user's **intent** and **sentiment**.")

# Input text area
user_input = st.text_area("Input Text", "", height=100)

# Button to analyze
if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        try:
            # Intent classification
            intent_result = intent_pipe(user_input)
            intent_label = intent_result[0]["label"]

            # Sentiment classification using OpenAI GPT
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a sentiment analysis assistant. Respond with 'Positive', 'Negative', or 'Neutral' and give a brief reason."
                    },
                    {
                        "role": "user",
                        "content": user_input
                    }
                ]
            )
            sentiment = response.choices[0].message.content.strip()

            # Display results
            st.subheader("🧠 Detected Intent:")
            st.info(intent_label)

            st.subheader("💬 Predicted Sentiment:")
            st.success(sentiment)

        except Exception as e:
            st.error(f"❌ Error: {e}")
