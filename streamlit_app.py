import streamlit as st
import tensorflow as tf
import numpy as np
import sentencepiece as spm
import os

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="TarjumanAI - Arabic to English Translator",
    page_icon="🌐➡️📝",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ========== CUSTOM CSS ==========
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp {
        background: linear-gradient(to bottom right, #1A1A1D, #0D0D0D);
        color: #F0EAD6;
        font-family: 'Georgia', serif;
    }

    h1 {
        color: #4facfe !important;
        text-align: center;
        font-size: 3rem;
    }

    textarea {
        background-color: #262730 !important;
        color: #F0EAD6 !important;
        border: 1px solid #4facfe !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
    }

    .stButton > button {
        background: linear-gradient(90deg, #4facfe, #00f2fe);
        color: #0D0D0D;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0px 8px 16px rgba(79, 172, 254, 0.3);
    }

    .footer {
        position: relative;
        display: inline-block;
        color: #888;
        text-align: center;
        margin-top: 3rem;
        font-size: 0.9rem;
    }

    .footer span:hover::after {
        content: " TarjumanAI v1.0 | Powered by Team CodeRunners ";
        position: absolute;
        top: -30px;
        right: 0;
        transform: translateX(0%);
        background-color: #333;
        color: #fff;
        padding: 5px 10px;
        border-radius: 5px;
        white-space: nowrap;
        font-size: 0.8rem;
        opacity: 1;
        z-index: 10;
    }

    .block-container {
        padding-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# ========== LOAD TOKENIZERS ==========
@st.cache_resource
def load_tokenizers():
    source_tokenizer = spm.SentencePieceProcessor()
    target_tokenizer = spm.SentencePieceProcessor()

    source_tokenizer.load('source_tokenizer.subword.subwords')
    target_tokenizer.load('target_tokenizer.subword.subwords')

    return source_tokenizer, target_tokenizer

# ========== LOAD MODEL ==========
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('arabic_to_english_transformer_weights.weights.h5')
    return model

# ========== EVALUATION FUNCTION ==========
def evaluate(sentence, model, source_tokenizer, target_tokenizer, max_length=40):
    # Encode input sentence
    sentence = sentence.lower().strip()
    sentence_ids = source_tokenizer.encode(sentence)
    encoder_input = tf.expand_dims(sentence_ids, 0)

    # Prepare decoder input with <start> token
    start_token = target_tokenizer.bos_id()
    end_token = target_tokenizer.eos_id()

    decoder_input = [start_token]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(max_length):
        predictions = model([encoder_input, output])
        predictions = predictions[:, -1:, :]
        predicted_id = tf.argmax(predictions, axis=-1).numpy()[0][0]

        if predicted_id == end_token:
            break

        output = tf.concat([output, [[predicted_id]]], axis=-1)

    decoded_sentence = target_tokenizer.decode([int(i) for i in output.numpy()[0] if i != 0])
    return decoded_sentence

# ========== STREAMLIT UI ==========
st.title("🌐➡️📝 TarjumanAI")
st.markdown("##### Translate Arabic to English Seamlessly")

# Load everything
source_tokenizer, target_tokenizer = load_tokenizers()
model = load_model()

if model:
    st.success("✅ Model loaded successfully!")
else:
    st.error("❌ Failed to load the model.")

# Input Area
st.subheader("📝 Enter Arabic Text:")
arabic_input = st.text_area("", height=200, placeholder="اكتب النص هنا... (Write Arabic text here)")

# Translate Button
if st.button("✨ Translate to English", use_container_width=True):
    if not arabic_input.strip():
        st.warning("⚠️ Please enter Arabic text to translate!")
    else:
        with st.spinner("Translating..."):
            translation = evaluate(arabic_input, model, source_tokenizer, target_tokenizer)
            st.subheader("💬 Translated English Text:")
            st.code(translation, language=None)

# Footer
st.markdown("""
---
<p class="footer" style="text-align: center;">
    Built with ❤️ by <span>Team CodeRunners</span>
</p>
""", unsafe_allow_html=True)
