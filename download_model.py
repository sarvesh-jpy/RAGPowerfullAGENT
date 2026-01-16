from langchain_huggingface import HuggingFaceEmbeddings

print("⏳ Downloading model... this may take a few minutes.")
# This will force the download to cache
model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("✅ Model downloaded successfully! You can now run app.py")