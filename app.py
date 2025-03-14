import streamlit as st
import torch
from model.model import load_resources, translate
import base64

# Custom page icon (a professional translation icon)
def get_custom_icon():
    # Base64 encoded SVG icon for translation
    icon_base64 = """
    <svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512" fill="none">
      <rect width="512" height="512" rx="256" fill="#1A56DB"/>
      <path d="M346 185H256V215H346V185Z" fill="white"/>
      <path d="M346 245H256V275H346V245Z" fill="white"/>
      <path d="M346 305H256V335H346V305Z" fill="white"/>
      <path d="M196 185H166V335H196V185Z" fill="white"/>
      <path d="M226 185H196V215H226V185Z" fill="white"/>
      <path d="M226 305H196V335H226V305Z" fill="white"/>
    </svg>
    """
    return f"data:image/svg+xml;base64,{base64.b64encode(icon_base64.encode()).decode()}"

# Page configuration with custom icon
st.set_page_config(
    page_title="Arabic-English Neural Translator",
    page_icon=get_custom_icon(),
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern UI styling with a professional color scheme
st.markdown("""
<style>
    /* Global styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }
    
    /* Header styling */
    .header {
        text-align: center;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #f0f0f0;
    }
    
    .header h1 {
        color: #1a56db;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .header p {
        color: #6b7280;
        font-size: 1.1rem;
        max-width: 600px;
        margin: 0 auto;
    }
    
    /* Logo styling */
    .logo-container {
        display: flex;
        justify-content: center;
        margin-bottom: 1rem;
    }
    
    .logo {
        width: 80px;
        height: 80px;
        background-color: #1a56db;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 2.5rem;
        box-shadow: 0 4px 6px rgba(26, 86, 219, 0.2);
    }
    
    /* Card styling */
    .card {
        background-color: white;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.1);
        margin-bottom: 24px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .card:hover {
        box-shadow: 0 10px 15px rgba(0, 0, 0, 0.05), 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    .card-header {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }
    
    .card-icon {
        background-color: #e0e7ff;
        color: #1a56db;
        width: 40px;
        height: 40px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 16px;
        font-size: 20px;
    }
    
    .card-title {
        color: #111827;
        font-weight: 600;
        font-size: 1.25rem;
        margin: 0;
    }
    
    /* Language badges */
    .language-switcher {
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 20px 0;
        gap: 15px;
    }
    
    .language-badge {
        padding: 8px 16px;
        border-radius: 30px;
        font-weight: 500;
        font-size: 0.95rem;
        display: flex;
        align-items: center;
    }
    
    .source-lang {
        background-color: #f3f4f6;
        color: #374151;
    }
    
    .target-lang {
        background-color: #1a56db;
        color: white;
    }
    
    .lang-icon {
        margin-right: 8px;
    }
    
    /* Translation area */
    .translation-area {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 24px;
    }
    
    @media (max-width: 768px) {
        .translation-area {
            grid-template-columns: 1fr;
        }
    }
    
    .input-container, .output-container {
        border-radius: 8px;
        overflow: hidden;
    }
    
    .input-header, .output-header {
        padding: 12px 16px;
        font-weight: 500;
        font-size: 0.95rem;
        display: flex;
        align-items: center;
    }
    
    .input-header {
        background-color: #f9fafb;
        color: #374151;
        border: 1px solid #e5e7eb;
        border-bottom: none;
    }
    
    .output-header {
        background-color: #1a56db;
        color: white;
    }
    
    .textarea-container {
        position: relative;
    }
    
    /* Button styling */
    .primary-button {
        background-color: #1a56db;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 6px;
        font-weight: 500;
        cursor: pointer;
        transition: background-color 0.2s;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
    }
    
    .primary-button:hover {
        background-color: #1e429f;
    }
    
    .secondary-button {
        background-color: #f3f4f6;
        color: #374151;
        border: 1px solid #e5e7eb;
        padding: 10px 20px;
        border-radius: 6px;
        font-weight: 500;
        cursor: pointer;
        transition: background-color 0.2s;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
    }
    
    .secondary-button:hover {
        background-color: #e5e7eb;
    }
    
    .button-container {
        display: flex;
        gap: 12px;
        margin-top: 16px;
    }
    
    /* Result container */
    .result-container {
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 20px;
        margin-top: 16px;
    }
    
    .result-header {
        font-weight: 600;
        color: #111827;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .result-content {
        background-color: white;
        padding: 16px;
        border-radius: 6px;
        border: 1px solid #e5e7eb;
        line-height: 1.6;
    }
    
    /* Features section */
    .features-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 20px;
        margin-top: 30px;
    }
    
    .feature-card {
        background-color: white;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
    }
    
    .feature-icon {
        background-color: #e0e7ff;
        color: #1a56db;
        width: 50px;
        height: 50px;
        border-radius: 25px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        margin-bottom: 16px;
    }
    
    .feature-title {
        font-weight: 600;
        margin-bottom: 8px;
        color: #111827;
    }
    
    .feature-description {
        color: #6b7280;
        font-size: 0.95rem;
    }
    
    /* Footer styling */
    .footer {
        margin-top: 40px;
        padding: 24px;
        background-color: #f9fafb;
        border-radius: 12px;
        text-align: center;
    }
    
    .footer-title {
        font-weight: 600;
        color: #111827;
        margin-bottom: 16px;
    }
    
    .contributors {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 24px;
        margin-bottom: 20px;
    }
    
    .contributor {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    
    .contributor-name {
        font-weight: 500;
        margin-bottom: 4px;
    }
    
    .contributor-link {
        color: #1a56db;
        text-decoration: none;
        font-size: 0.9rem;
    }
    
    .contributor-link:hover {
        text-decoration: underline;
    }
    
    .copyright {
        color: #6b7280;
        font-size: 0.9rem;
        margin-top: 16px;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f9fafb;
        padding: 20px;
        border-radius: 8px;
    }
    
    .sidebar-title {
        font-weight: 600;
        color: #111827;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .sidebar-section {
        margin-bottom: 24px;
    }
    
    .sidebar-subtitle {
        font-weight: 500;
        color: #374151;
        margin-bottom: 8px;
        font-size: 0.95rem;
    }
    
    .sidebar-text {
        color: #6b7280;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    
    .model-specs {
        background-color: white;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        padding: 12px;
        margin-top: 12px;
    }
    
    .spec-item {
        display: flex;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px solid #f3f4f6;
    }
    
    .spec-item:last-child {
        border-bottom: none;
    }
    
    .spec-label {
        color: #6b7280;
        font-size: 0.85rem;
    }
    
    .spec-value {
        color: #111827;
        font-weight: 500;
        font-size: 0.85rem;
    }
    
    /* Loading spinner */
    .loading-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 30px;
    }
    
    .loading-text {
        margin-top: 16px;
        color: #6b7280;
    }
    
    /* Utility classes */
    .text-center {
        text-align: center;
    }
    
    .mt-4 {
        margin-top: 16px;
    }
    
    .mb-4 {
        margin-bottom: 16px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #c5c5c5;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #a8a8a8;
    }
    
    /* Streamlit element overrides */
    .stTextArea > div > div > textarea {
        border: 1px solid #e5e7eb !important;
        color: #111827 !important;
        font-size: 1rem !important;
        min-height: 150px !important;
    }
    
    .stTextArea > div {
        border: none !important;
    }
    
    div[data-testid="stVerticalBlock"] {
        gap: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Main app container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Custom logo and header section
st.markdown("""
<div class="header">
    <div class="logo-container">
        <div class="logo">
            <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M5 8l6 6"></path>
                <path d="M4 14l6-6 2-3"></path>
                <path d="M2 5l7 7"></path>
                <path d="M14 11l7 7"></path>
                <path d="M18 7l-4 4 3 3"></path>
                <path d="M22 5l-7 7"></path>
            </svg>
        </div>
    </div>
    <h1>Arabic-English Neural Translator</h1>
    <p>Professional neural machine translation powered by advanced transformer architecture</p>
</div>
""", unsafe_allow_html=True)

# Language switcher
st.markdown("""
<div class="language-switcher">
    <div class="language-badge source-lang">
        <span class="lang-icon">🇦🇪</span> Arabic
    </div>
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M5 12h14"></path>
        <path d="m12 5 7 7-7 7"></path>
    </svg>
    <div class="language-badge target-lang">
        <span class="lang-icon">🇬🇧</span> English
    </div>
</div>
""", unsafe_allow_html=True)

# Load the model and resources (cached to prevent reloading)
@st.cache_resource
def load_translation_resources():
    try:
        return load_resources()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Main translation card
st.markdown("""
<div class="card">
    <div class="card-header">
        <div class="card-icon">📝</div>
        <h2 class="card-title">Translation Interface</h2>
    </div>
""", unsafe_allow_html=True)

# Load resources with a better loading experience
with st.spinner(""):
    resources = load_translation_resources()

if resources:
    model, src_tokenizer, tgt_tokenizer, device = resources
    
    # Create columns for input and output
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        st.markdown('<div class="input-header">🇦🇪 Arabic Input</div>', unsafe_allow_html=True)
        arabic_text = st.text_area(
            "",
            placeholder="اكتب أو الصق النص العربي هنا...", 
            height=200,
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="output-container">', unsafe_allow_html=True)
        st.markdown('<div class="output-header">🇬🇧 English Translation</div>', unsafe_allow_html=True)
        if 'translation' in st.session_state and st.session_state.translation:
            st.text_area(
                "",
                value=st.session_state.translation,
                height=200,
                disabled=True,
                label_visibility="collapsed"
            )
        else:
            st.text_area(
                "",
                placeholder="Translation will appear here...",
                height=200,
                disabled=True,
                label_visibility="collapsed"
            )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Buttons with better styling
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 3])
    
    with col_btn1:
        translate_clicked = st.button("Translate", type="primary", use_container_width=True)
    
    with col_btn2:
        clear_clicked = st.button("Clear", type="secondary", use_container_width=True)
    
    # Handle button actions
    if clear_clicked:
        st.session_state.arabic_text = ""
        st.session_state.translation = None
        st.rerun()
    
    if translate_clicked and arabic_text:
        with st.spinner("Translating..."):
            try:
                english_translation = translate(model, arabic_text, src_tokenizer, tgt_tokenizer, device)
                st.session_state.translation = english_translation
                st.rerun()
            except Exception as e:
                st.error(f"Translation error: {str(e)}")
    
    # Character counter
    if arabic_text:
        st.markdown(f"<div style='text-align: right; color: #6b7280; font-size: 0.85rem;'>{len(arabic_text)} characters</div>", unsafe_allow_html=True)

else:
    # Better error handling with custom styling
    st.markdown("""
    <div style="background-color: #fee2e2; border: 1px solid #fecaca; border-radius: 8px; padding: 16px; margin: 20px 0;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#b91c1c" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="12" y1="8" x2="12" y2="12"></line>
                <line x1="12" y1="16" x2="12.01" y2="16"></line>
            </svg>
            <div>
                <div style="font-weight: 600; color: #b91c1c; margin-bottom: 4px;">Failed to load translation model</div>
                <div style="color: #7f1d1d;">Please check the console for detailed error information.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # Close the card

# Features section
st.markdown("""
<div class="card">
    <div class="card-header">
        <div class="card-icon">✨</div>
        <h2 class="card-title">Key Features</h2>
    </div>
    <div class="features-grid">
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <div class="feature-title">Neural Translation</div>
            <div class="feature-description">Powered by state-of-the-art transformer neural networks for high-quality translations</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Fast Processing</div>
            <div class="feature-description">Optimized for speed with efficient model architecture and caching</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🔄</div>
            <div class="feature-title">Context Awareness</div>
            <div class="feature-description">Understands context and nuances for more accurate translations</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📱</div>
            <div class="feature-title">Responsive Design</div>
            <div class="feature-description">Works seamlessly across desktop and mobile devices</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Footer with contributors
st.markdown("""
<div class="footer" style="background-color: #000000;">
    <div class="footer-title">Contributors</div>
    <div class="contributors">
        <div class="contributor">
            <div class="contributor-name">M Husnain</div>
            <a href="https://www.linkedin.com/in/m-husnain-6b94b6279/" class="contributor-link" target="_blank">LinkedIn Profile</a>
        </div>
        <div class="contributor">
            <div class="contributor-name">Marwa Shahid</div>
            <a href="https://www.linkedin.com/in/marwashahid/" class="contributor-link" target="_blank">LinkedIn Profile</a>
        </div>
    </div>
    <div class="copyright">© 2025 Arabic-English Neural Machine Translation</div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # Close the main container