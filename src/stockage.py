import json
import os

# Les imports ont changé de place dans les dernières versions :
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
 # <--- Changement ici
from langchain_text_splitters import RecursiveCharacterTextSplitter # <--- Et ici
from langchain_core.documents import Document # <--- Standardisation ici

# 1. Chargement des données transcrites
print('pass')
with open("tiktok_rag_docs.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. Préparation des documents
raw_documents = []
for entry in data:
    doc = Document(
        page_content=entry["page_content"],
        metadata=entry["metadata"]
    )
    raw_documents.append(doc)



text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
documents = text_splitter.split_documents(raw_documents)

# 4. Création des Embeddings (Transformation en vecteurs mathématiques)
# On utilise un modèle léger qui tourne sur ton CPU
print("⏳ Génération des vecteurs (Embeddings)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 5. Stockage dans ChromaDB
# Cela va créer un dossier 'chroma_db' dans ton projet
print("📦 Indexation dans la base vectorielle...")
vector_db = Chroma.from_documents(
    documents=documents, 
    embedding=embeddings, 
    persist_directory="./chroma_db"
)

print("✅ Système RAG prêt !")