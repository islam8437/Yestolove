import os
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
 # <--- Changement ici
from langchain_text_splitters import RecursiveCharacterTextSplitter # <--- Et ici
from langchain_core.documents import Document
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

api_key = os.getenv("GROQ_API_KEY")
DB_DIR = "./chroma_db"


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# Le "Cerveau" Groq (LPU)
llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    temperature=0.1 # Basse pour éviter l'invention (hallucination)
)

# --- 3. MANAGEMENT DU CONTEXTE (PROMPT) ---
system_prompt = (
    "Tu es un assistant expert basé sur les conseils des documents auxquels tu as accès. "
    "Utilise les extraits de transcription fournis pour répondre de manière précise et chaleureuse. "
    "Si l'information n'est pas dans le contexte, dis que tu ne sais pas, n'invente rien. "
    "Réponds de manière concise et bienveillante." \
     "sers toi juste que les documents que je t ai donne ."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Voici les extraits des vidéos TikTok :\n\n{context}\n\nQuestion : {input}"),
])

# --- 4. CONSTRUCTION DE LA CHAÎNE LCEL ---
def format_docs(docs):
    """Combine les textes des vidéos pour le LLM."""
    return "\n\n".join(doc.page_content for doc in docs)

# La chaîne de traitement : Recherche -> Formatage -> Prompt -> LLM -> Texte
rag_chain = (
    {
        "context": retriever | format_docs, 
        "input": RunnablePassthrough()
    }
    | prompt 
    | llm 
    | StrOutputParser()
)

# --- 5. INTERFACE DE CHAT ---
def start_agent():
    print("--- 🤖 AGENT YESTOLOVE OPÉRATIONNEL ---")
    print("(Tape 'quitter' pour arrêter)")
    
    while True:
        user_query = input("\n👤 Toi : ")
        if user_query.lower() in ["quitter", "exit", "quit"]:
            print("👋 Au revoir !")
            break
        
        try:
            # Récupération des documents pour afficher les sources TikTok
            source_docs = retriever.invoke(user_query)
            
            # Génération de la réponse via Groq
            print("\n🚀 Réflexion Groq...")
            response = rag_chain.invoke(user_query)
            
            print(f"\n✨ AGENT : {response}")
            
                    
        except Exception as e:
            print(f"❌ Une erreur est survenue : {e}")

if __name__ == "__main__":
    start_agent()