import streamlit as st
import os
import torch
import pdfplumber
import chromadb
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma 
from gtts import gTTS
from io import BytesIO

# --- 1. SYSTEM CONFIGURATION ---
st.set_page_config(page_title="High-Perf RAG", layout="wide")
st.title("🚀 High-Performance RAG (GPU Accelerated)")

# --- GPU DIAGNOSTICS (THE FIX) ---
# This block forces a check. If no GPU, it stops and tells you why.
try:
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        st.sidebar.success(f"✅ GPU DETECTED: {gpu_name}")
        st.sidebar.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        st.error("❌ NO GPU DETECTED!")
        st.warning("You are running on CPU. To fix this, run: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        # Fallback to cpu just so app doesn't crash, but warn user
        device = "cpu"
except Exception as e:
    st.error(f"GPU Check Failed: {e}")
    device = "cpu"

# --- 2. INITIALIZE POWERFUL MODELS ---

api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
if not api_key:
    st.warning("Enter API Key to start.")
    st.stop()

client = Groq(api_key=api_key)

# LOAD STRONG EMBEDDING MODEL ON GPU
@st.cache_resource
def get_embedding_model():
    # We use 'model_kwargs' to force the model onto the GPU (cuda)
    encode_kwargs = {'normalize_embeddings': True} # Best practice for BGE
    
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': device},  # <--- CRITICAL: Sends model to GPU
        encode_kwargs=encode_kwargs
    )

try:
    with st.spinner(f"Loading BGE-Large on {device.upper()}..."):
        embedding_model = get_embedding_model()
except Exception as e:
    st.error(f"Model Load Error: {e}")
    st.stop()

persist_directory = "./chroma_db_high_perf"

# --- 3. ADVANCED PDF PROCESSING ---

def process_documents(uploaded_files):
    documents = []
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    status_text.text("Extracting text...")
    
    for i, file in enumerate(uploaded_files):
        with pdfplumber.open(file) as pdf:
            file_text = ""
            for page in pdf.pages:
                text = page.extract_text(layout=True)
                if text:
                    file_text += text + "\n\n"
            
            doc = Document(page_content=file_text, metadata={"source": file.name})
            documents.append(doc)
        progress_bar.progress((i + 1) / len(uploaded_files))

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, 
        chunk_overlap=500, 
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    status_text.text(f"⚡ Encoding {len(chunks)} chunks on {device.upper()} (Fast)...")
    
    # Store in Chroma
    # Chroma automatically uses the embedding_model which is now on GPU
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    
    status_text.success(f"✅ Indexed {len(chunks)} chunks using {device}!")
    return vector_store

# --- 4. RETRIEVAL & GENERATION ---

def query_rag(question, vector_store):
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7}
    )
    
    docs = retriever.invoke(question)
    context_text = "\n---\n".join([doc.page_content for doc in docs])
    
    # Debug expander
    with st.expander("🔍 Debug: High-Res Context"):
        st.text(context_text)
    
    system_prompt = (
        "You are an expert data analyst. Use the provided context to answer the question. "
        "If the context is a table, read across rows carefully."
    )
    
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"}
        ],
        temperature=0.1
    )
    
    return completion.choices[0].message.content

# --- 5. AUDIO UTILS ---
def transcribe_audio(audio_bytes):
    try:
        return client.audio.transcriptions.create(
            file=("audio.wav", audio_bytes, "audio/wav"),
            model="whisper-large-v3",
            response_format="text"
        )
    except: return None

def text_to_speech(text):
    try:
        mp3_fp = BytesIO()
        gTTS(text=text, lang='en').write_to_fp(mp3_fp)
        return mp3_fp
    except: return None

# --- 6. UI LAYOUT ---

st.sidebar.header("Data Source")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files and st.sidebar.button("Process High-Res"):
    with st.spinner(f"Processing on {device}..."):
        st.session_state.vector_store = process_documents(uploaded_files)

st.header("Analysis Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

audio_val = st.audio_input("Voice Command")
if audio_val:
    txt = transcribe_audio(audio_val)
    if txt:
        st.session_state.messages.append({"role": "user", "content": txt})
        with st.chat_message("user"): st.markdown(txt)
        if "vector_store" in st.session_state:
            ans = query_rag(txt, st.session_state.vector_store)
            st.session_state.messages.append({"role": "assistant", "content": ans})
            with st.chat_message("assistant"): st.markdown(ans)
            audio = text_to_speech(ans)
            if audio: st.audio(audio, format='audio/mp3', start_time=0)

prompt = st.chat_input("Type query...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    
    if "vector_store" in st.session_state:
        with st.spinner("Analyzing..."):
            ans = query_rag(prompt, st.session_state.vector_store)
            st.session_state.messages.append({"role": "assistant", "content": ans})
            with st.chat_message("assistant"): st.markdown(ans)
    else:
        st.error("Upload data first.")