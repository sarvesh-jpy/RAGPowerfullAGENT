# 🚀 RAGPowerfullAGENT: High-Performance GPU RAG

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-green?style=for-the-badge&logo=nvidia)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red?style=for-the-badge&logo=streamlit)
![LangChain](https://img.shields.io/badge/Orchestration-LangChain-orange?style=for-the-badge)

**RAGPowerfullAGENT** is a state-of-the-art local RAG (Retrieval-Augmented Generation) system designed for analyzing complex documents, specifically optimized for **PDF tables and structured data**.

Unlike standard RAG implementations, this engine forces **GPU acceleration** for embedding generation, utilizing the massive `BAAI/bge-large-en-v1.5` model to deliver industry-leading retrieval accuracy without the latency.

## ✨ Key Features

* **⚡ GPU-Accelerated Ingestion:** Forces PyTorch to utilize local CUDA cores (NVIDIA) for 50x faster document embedding compared to CPU.
* **🧠 SOTA Embeddings:** Powered by `BAAI/bge-large-en-v1.5` (Top 1 on MTEB Leaderboard) for deep semantic understanding.
* **🔍 MMR Retrieval:** Uses *Maximal Marginal Relevance* to fetch diverse document chunks, ensuring context isn't lost across page breaks.
* **🗣️ Voice-to-Voice Interface:** Integrated **Whisper Large V3** (via Groq) for speech recognition and **gTTS** for spoken responses.
* **📊 Table-Optimized Chunking:** Custom recursive splitting strategy designed to keep complex PDF tables intact during analysis.
* **🤖 Llama 3.3 70B Intelligence:** Uses Groq's LPU inference engine for near-instant answers.

## 🛠️ Tech Stack

* **Core Logic:** Python, LangChain
* **Vector Database:** ChromaDB (Persistent local storage)
* **Inference:** Groq API (Llama 3.3 70B Versatile)
* **Embeddings:** HuggingFace `sentence-transformers` (Running on Local GPU)
* **UI:** Streamlit

## ⚙️ Prerequisites

* **OS:** Windows / Linux (with NVIDIA GPU recommended)
* **Python:** 3.10 or higher
* **API Key:** [Groq Cloud API Key](https://console.groq.com/)
* **Hardware:** NVIDIA GPU with CUDA drivers installed (Optional but recommended).

## 📥 Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/RAGPowerfullAGENT.git](https://github.com/yourusername/RAGPowerfullAGENT.git)
cd RAGPowerfullAGENT
