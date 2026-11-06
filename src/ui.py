import streamlit as st
from src.device import available_devices
from src.summarizer import SummarizerService

@st.cache_resource
def load_model(model, device):
    return SummarizerService(model, device)


def run_app():
    st.set_page_config(page_title="Summarizer")
    st.title("Local 🤗 Summarizer")
    model=st.text_input("Model", "sshleifer/distilbart-cnn-12-6")
    device=st.selectbox("Device", available_devices())
    text=st.text_area("Input text", height=300)
    if st.button("Summarize"):
        svc=load_model(model, device)
        st.write(svc.summarize(text))
