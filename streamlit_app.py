import streamlit as st
import tensorflow as tf
import tensorflow_datasets as tfds
import os

# Streamlit Page Config
st.set_page_config(
    page_title="TarjumanAI - Arabic to English Translator",
    page_icon="🌐➡️📝",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ========== Custom CSS for Dark Mode ==========
st.markdown("""
    <style>
    #MainMenu, footer, header {visibility: hidden;}

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

    h3 {
        color: #D4AF37 !important;
        text-align: center;
    }

    .stTextArea textarea {
        background-color: #262730;
        color: #F0EAD6;
        border: 1px solid #4facfe;
        border-radius: 10px;
        font-size: 1rem;
    }

    .stButton > button {
        background: linear-gradient(90deg, #4facfe, #00f2fe);
        color: #0D0D0D;
        font-weight: bold;
        font-size: 1.1rem;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
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

# ========== Tokenizer & Model Loading ==========
@st.cache_resource
def load_tokenizer(filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(
        os.path.join(current_dir, filename)
    )
    return tokenizer

@st.cache_resource
def load_model(source_vocab_size, target_vocab_size):
    transformer = Transformer(
        vocab_size_enc=source_vocab_size + 2,
        vocab_size_dec=target_vocab_size + 2,
        d_model=512,
        n_layers=4,
        ffn_units=512,
        num_heads=8,
        dropout_rate=0.1
    )
    dummy_enc_input = tf.ones((1, 15), dtype=tf.int32)
    dummy_dec_input = tf.ones((1, 15), dtype=tf.int32)
    transformer(dummy_enc_input, dummy_dec_input, training=False)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "model_files", "arabic_to_english_transformer_weights.weights.h5")
    transformer.load_weights(model_path)
    return transformer

# ========== Import Model Classes ==========
# Note: Skipping class definitions here since they remain unchanged.
# You can keep your Transformer, Encoder, Decoder, etc. classes as they are.

# Load Tokenizers & Model
source_tokenizer_path = os.path.join("model_files", "source_tokenizer.subword")
target_tokenizer_path = os.path.join("model_files", "target_tokenizer.subword")

source_tokenizer = load_tokenizer(source_tokenizer_path)
target_tokenizer = load_tokenizer(target_tokenizer_path)

model = load_model(source_tokenizer.vocab_size, target_tokenizer.vocab_size)

# Start and End Tokens
num_words_inputs = source_tokenizer.vocab_size + 2
num_words_output = target_tokenizer.vocab_size + 2
start_token_source = [num_words_inputs - 2]
end_token_source = [num_words_inputs - 1]
start_token_target = [num_words_output - 2]
end_token_target = [num_words_output - 1]

# ========== Translation Functions ==========
def predict(inp_sentence, tokenizer_in, tokenizer_out, target_max_len):
    inp_sentence = start_token_source + tokenizer_in.encode(inp_sentence) + end_token_source
    enc_input = tf.expand_dims(inp_sentence, axis=0)

    out_sentence = start_token_target
    output = tf.expand_dims(out_sentence, axis=0)

    for _ in range(target_max_len):
        predictions = model(enc_input, output, training=False)
        prediction = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(prediction, axis=-1), tf.int32)

        if predicted_id == end_token_target:
            return tf.squeeze(output, axis=0)

        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)

def translate(sentence):
    output = predict(sentence, source_tokenizer, target_tokenizer, 15)
    predicted_sentence = target_tokenizer.decode(
        [i for i in output if i < start_token_target[0]]
    )
    return predicted_sentence

# ========== Streamlit App UI ==========
st.title("🌐➡️📝 TarjumanAI")
st.markdown("##### Translate Arabic to English Seamlessly")

user_input = st.text_area("📝 Enter Arabic Text Below", height=150)

if st.button("✨ Translate to English", use_container_width=True):
    if not user_input.strip():
        st.warning("⚠️ Please enter Arabic text first!")
    else:
        with st.spinner("Translating..."):
            translation = translate(user_input)
            st.subheader("💬 Translated English Text:")
            st.code(translation, language=None)

# Footer
st.markdown("""
---
<p class="footer" style="text-align: center;">
    Built with ❤️ by <span>Team CodeRunners</span>
</p>
""", unsafe_allow_html=True)
