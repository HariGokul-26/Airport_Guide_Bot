import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# ============ CONFIG ============
JSONL_FILE = "minichangi_datas.jsonl"
COLLECTION_NAME = "rag_demo"
CHROMA_PATH = "chroma_db"   # folder for persistent ChromaDB
# =================================

# Load .env
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Local embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Persistent Chroma
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# Get or create collection
if COLLECTION_NAME in [c.name for c in chroma_client.list_collections()]:
    collection = chroma_client.get_collection(COLLECTION_NAME)
else:
    collection = chroma_client.create_collection(COLLECTION_NAME)

# Load JSONL data
print("Loading JSONL data...")
chunks = []
with open(JSONL_FILE, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        text = f"{data.get('name','')} - {data.get('description','')}"
        chunks.append(text)

print(f"Total records loaded: {len(chunks)}")

# Embed + add to Chroma (only if empty)
if collection.count() == 0:
    print("Embedding and adding chunks to Chroma...")
    embeddings = embedder.encode(chunks).tolist()
    ids = [f"id_{i}" for i in range(len(chunks))]
    metadatas = [{"source": f"chunk_{i}"} for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks, metadatas=metadatas, embeddings=embeddings)
    print("Chunks stored in ChromaDB ✅")
else:
    print("Collection already has data in persistent DB. Skipping re-insert ✅")

# ============ RAG QUERY ============
def rag_query(query):
    # Embed query
    query_emb = embedder.encode([query]).tolist()[0]

    # Retrieve from Chroma
    results = collection.query(query_embeddings=[query_emb], n_results=3)

    # Prepare context
    retrieved_docs = results["documents"][0]
    context = "\n".join(retrieved_docs)

    # Build prompt
    prompt = f"""You are a helpful assistant.
Answer the user question using ONLY the context below.
If the context does not contain the answer, say "I don’t know."

Context:
{context}

Question: {query}
Answer:"""

    # Groq API call (updated model)
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()

# ============ CHAT LOOP ============
if __name__ == "__main__":
    print("Chatbot is ready with persistent ChromaDB! Type 'exit' to quit.")
    while True:
        query = input("\nAsk something: ")
        if query.lower() == "exit":
            break
        answer = rag_query(query)
        print("\n🤖 Bot:", answer)
