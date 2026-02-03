import json
import yt_dlp
import os
from faster_whisper import WhisperModel
from datetime import datetime

# --- CONFIGURATION ---
INPUT_FILE = "tiktok_urls.json"
DOCS_FOLDER = "documents_rag"  # Dossier qui contiendra chaque JSON
TEMP_AUDIO_FOLDER = "temp_audio"
MODEL_SIZE = "medium"
FFMPEG_PATH = r"C:\Users\12\Downloads\ffmpeg-2026-01-14-git-6c878f8b82-full_build\ffmpeg-2026-01-14-git-6c878f8b82-full_build\bin"

# Création des dossiers nécessaires
for folder in [TEMP_AUDIO_FOLDER, DOCS_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# --- CHARGEMENT DU MODÈLE ---
print(f"⏳ Chargement du modèle Faster-Whisper ({MODEL_SIZE})...")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

def download_audio(video_url, video_id):
    """Télécharge l'audio via yt-dlp"""
    output_path = os.path.join(TEMP_AUDIO_FOLDER, f"{video_id}")
    ydl_opts = {
        'ffmpeg_location': FFMPEG_PATH, 
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_path,
        'quiet': True,
        'no_warnings': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        return f"{output_path}.mp3"
    except Exception as e:
        print(f"   ❌ Erreur de téléchargement pour {video_id}: {e}")
        return None

def transcribe_fast(audio_path):
    """Transcrit l'audio"""
    segments, info = model.transcribe(audio_path, beam_size=5, language="fr")
    full_text = " ".join([segment.text for segment in segments])
    return full_text.strip()

def process_videos(json_data):
    """Transforme les vidéos en documents individuels"""
    rag_documents = []
    print(f"🚀 Démarrage du traitement de {len(json_data)} vidéos...")

    for index, entry in enumerate(json_data):
        video_id = entry['id']
        file_path = os.path.join(DOCS_FOLDER, f"{video_id}.json")
        
        print(f"\n[{index+1}/{len(json_data)}] 📥 Vidéo ID : {video_id}")

        # --- ÉTAPE 1 : VÉRIFICATION DOUBLON ---
        if os.path.exists(file_path):
            print(f"   ℹ️ Document déjà existant. Chargement depuis le dossier...")
            with open(file_path, "r", encoding="utf-8") as f:
                rag_doc = json.load(f)
            rag_documents.append(rag_doc)
            continue

        # --- ÉTAPE 2 : TÉLÉCHARGEMENT ---
        audio_file = download_audio(entry['url'], video_id)
        
        if audio_file and os.path.exists(audio_file):
            try:
                # --- ÉTAPE 3 : TRANSCRIPTION ---
                print("   🎙️ Transcription en cours...")
                transcription_text = transcribe_fast(audio_file)
                
                # --- ÉTAPE 4 : CRÉATION DU DOCUMENT ---
                doc_content = f"TITRE: {entry.get('title', '')}\nCONTENU: {transcription_text}"
                
                rag_doc = {
                    "id": video_id,
                    "page_content": doc_content,
                    "metadata": {
                        "source": "tiktok",
                        "url": entry['url'],
                        "views": entry.get('view_count', 0),
                        "video_id": video_id,
                        "date_traitement": datetime.now().strftime("%Y-%m-%d %H:%M")
                    }
                }
                
                # --- ÉTAPE 5 : SAUVEGARDE INDIVIDUELLE ---
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(rag_doc, f, indent=4, ensure_ascii=False)
                
                rag_documents.append(rag_doc)
                print(f"   ✅ Document sauvegardé ({len(transcription_text)} caractères)")
                
                # Nettoyage MP3
                os.remove(audio_file)
                
            except Exception as e:
                print(f"   ❌ Erreur : {e}")
        else:
            print("   ⚠️ Impossible de récupérer l'audio.")

    return rag_documents

if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Fichier source manquant.")
    else:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        final_list = process_videos(data)
        print(f"\n🎉 Terminés ! {len(final_list)} fichiers sont dans '{DOCS_FOLDER}'")