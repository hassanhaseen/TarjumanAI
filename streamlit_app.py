import streamlit as st

def translate_arabic_to_english(arabic_text):
    # Placeholder function, replace with Transformer model later
    return "English translation coming soon!"

st.title("TarjumanAI 📖 - Arabic to English Translator")
arabic_text = st.text_area("Enter Arabic Text:", height=200, placeholder="اكتب النص العربي هنا...")

if st.button("Translate"):
    english_translation = translate_arabic_to_english(arabic_text)
    st.text_area("English Translation:", english_translation, height=200)
