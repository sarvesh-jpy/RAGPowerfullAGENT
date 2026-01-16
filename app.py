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
st.title("🚀 High-Performance RAG (BGE-Large + MMR)")

# Check if GPU is available for the heavy embedding model

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    st.sidebar.success(f"🚀 Running on GPU: {torch.cuda.get_device_name(0)}")
else:
    st.sidebar.warning("⚠️ Running on CPU (Slower)")
# --- 2. INITIALIZE POWERFUL MODELS ---

# API Key Setup
api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
if not api_key:
    st.warning("Enter API Key to start.")
    st.stop()

client = Groq(api_key=api_key)

# LOAD STRONG EMBEDDING MODEL
# We use 'BAAI/bge-large-en-v1.5' -> Top tier retrieval performance
@st.cache_resource
def get_embedding_model():
    # Explicitly tell it to use the device we found
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': device} 
    
    )
try:
    with st.spinner(f"Loading heavy model (BGE-Large)... this happens once."):
        embedding_model = get_embedding_model()
except Exception as e:
    st.error(f"Model Load Error: {e}")
    st.stop()

persist_directory = "./chroma_db_high_perf"

# --- 3. ADVANCED PDF PROCESSING (TABLE OPTIMIZED) ---

def process_documents(uploaded_files):
    documents = []
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    status_text.text(" extracting text with layout analysis...")
    
    for i, file in enumerate(uploaded_files):
        with pdfplumber.open(file) as pdf:
            file_text = ""
            for page in pdf.pages:
                # Extract text preserving layout (crucial for tables)
                text = page.extract_text(layout=True)
                if text:
                    file_text += text + "\n\n"
            
            # Create Document
            doc = Document(page_content=file_text, metadata={"source": file.name})
            documents.append(doc)
        progress_bar.progress((i + 1) / len(uploaded_files))

    # CHUNKING STRATEGY FOR TABLES
    # Tables are large. Small chunks break them. 
    # We use a massive chunk size (2500) to keep full schedules together.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,     # Large chunks for full table context
        chunk_overlap=500,   # High overlap ensures no data is lost at cut points
        separators=["\n\n", "\n", " ", ""] # Prioritize breaking at paragraphs/rows
    )
    
    chunks = text_splitter.split_documents(documents)
    status_text.text(f"Generating Embeddings for {len(chunks)} chunks (This will take time on CPU)...")
    
    # Store in Chroma
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    
    status_text.success(f"✅ Indexed {len(chunks)} chunks with BGE-Large!")
    return vector_store

# --- 4. ADVANCED RETRIEVAL & GENERATION ---

def query_rag(question, vector_store):
    # SEARCH STRATEGY: MMR (Maximal Marginal Relevance)
    # Why? It fetches diverse chunks. If the answer is split across pages, MMR finds both.
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,             # Return top 6 chunks
            "fetch_k": 20,      # Look at top 20 candidates first
            "lambda_mult": 0.7  # Balance between 'relevant' and 'diverse'
        }
    )
    
    docs = retriever.invoke(question)
    context_text = "\n---\n".join([doc.page_content for doc in docs])
    
    # DEBUG: Show what the AI sees
    with st.expander("🔍 Debug: High-Res Context"):
        st.text(context_text)
    
    # SYSTEM PROMPT FOR COMPLEX TASKS
    system_prompt = (
        "You are an expert data analyst specialized in reading complex PDF tables. "
        "The text provided maintains physical layout (rows/cols). "
        "1. Analyze the Context carefully. "
        "2. If looking for a specific value (like a date or subject), scan the columns vertically. "
        "3. If the user asks for a specific code or department (e.g., '243' or 'AI&DS'), locate that section first. "
        "4. Answer precisely based ONLY on the context."
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
    with st.spinner("Processing with BGE-Large..."):
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