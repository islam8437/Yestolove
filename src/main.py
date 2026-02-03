import time
import os
import json
from scaraping import get_tiktok_profile_urls # Ton premier script
from audio_to_text import process_videos      # Ton script Faster-Whisper
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
 # <--- Changement ici
from langchain_text_splitters import RecursiveCharacterTextSplitter # <--- Et ici
from langchain_core.documents import Document
from langchain_core.documents import Document

# --- CONFIGURATION ---
PROFILE_URL = "https://www.tiktok.com/@lunivana"
BATCH_SIZE = 5
WAIT_TIME = 150
POINTER_FILE = "last_index.txt"
DB_DIR = "./chroma_db"

# Initialisation des Embeddings (pour rajouter à la DB existante)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
def get_last_index():
    if not os.path.exists(POINTER_FILE): return 1
    with open(POINTER_FILE, "r") as f: return int(f.read().strip())

def save_last_index(index):
    with open(POINTER_FILE, "w") as f: f.write(str(index))

def run_workflow():
    current_start = get_last_index()

        
    while True:
        current_end = current_start + BATCH_SIZE - 1
        print(f"\n--- 🔄 DEBUT DU BATCH : Vidéos {current_start} à {current_end} ---")

        # 1. SCRAPING (On récupère les URLs des 5 prochaines)
        # Note: Modifie ton scraper pour accepter start et end
        new_urls = get_tiktok_profile_urls(PROFILE_URL, start=current_start, end=current_end)

        if not new_urls:
            print("p Pas de nouvelles vidéos trouvées. Fin du compte ou erreur réseau.")
            break

        # 2. VOICE-TO-TEXT (Transcription)
        # On traite les 5 vidéos. On sait que ça marche (ex: 2464 caractères générés)
        new_docs_json = process_videos(new_urls) 

        if new_docs_json:
            # 3. MISE À JOUR DE LA VECTEUR DB (RAG)
            print(f"📦 Indexation de {len(new_docs_json)} nouveaux documents...")
            
            langchain_docs = []
            for item in new_docs_json:
                langchain_docs.append(Document(
                    page_content=item["page_content"],
                    metadata=item["metadata"]
                ))
            
            # On ajoute les nouveaux documents à la base existante
            vector_db = Chroma.from_documents(
                documents=langchain_docs, 
                embedding=embeddings, 
                persist_directory=DB_DIR
            )
            print("✅ Base de données mise à jour.")

        # 4. PASSAGE AU BATCH SUIVANT
        current_start += BATCH_SIZE
        save_last_index(current_start)
        print(f"😴 Pause de {WAIT_TIME} secondes avant le prochain lot...")
        time.sleep(WAIT_TIME)
if __name__ == "__main__":
    run_workflow()