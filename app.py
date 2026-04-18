import os
import gradio as gr
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datasets import load_dataset

load_dotenv()

URGENCES = [
    "douleur thoracique", "douleur poitrine", "infarctus",
    "avc", "paralysie", "perte de connaissance", "inconscient",
    "difficulte a respirer", "difficulté à respirer",
    "hemorragie", "hémorragie", "saignement abondant",
    "overdose", "intoxication", "suicide", "se suicider",
    "crise cardiaque", "arret cardiaque", "arrêt cardiaque", "convulsions"
]

MSG_URGENCE = (
    "URGENCE DÉTECTÉE\n\n"
    "Appelez immédiatement le **15 (SAMU)** ou le **112**.\n"
    "Ne perdez pas de temps — chaque seconde compte.\n\n"
    "Je suis un assistant informatif et ne peux pas gérer les urgences médicales."
)

def detecter_urgence(message: str) -> bool:
    msg = message.lower()
    return any(kw in msg for kw in URGENCES)

def charger_documents():
    print("Chargement du dataset médical...")
    docs = []
    try:
        print("  → Chargement HealthCareMagic-100k...")
        ds1 = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")
        print(f"     Colonnes : {ds1.column_names}")
        for item in ds1.select(range(min(3000, len(ds1)))):
            question = item.get("input", "") or item.get("question", "")
            answer   = item.get("output", "") or item.get("answer", "")
            if question and answer and len(answer) > 30:
                content = f"Question: {question}\nRéponse: {answer}"
                docs.append(Document(page_content=content, metadata={"source": "HealthCareMagic"}))
        print(f"     {len(docs)} documents chargés.")
    except Exception as e:
        print(f"     Erreur : {e}")
    try:
        print("  → Chargement iCliniq...")
        ds2 = load_dataset("lavita/ChatDoctor-iCliniq", split="train")
        print(f"     Colonnes : {ds2.column_names}")
        count_before = len(docs)
        for item in ds2.select(range(min(2000, len(ds2)))):
            question = item.get("input", "") or item.get("question", "")
            answer   = item.get("output", "") or item.get("answer", "")
            if question and answer and len(answer) > 30:
                content = f"Question: {question}\nRéponse: {answer}"
                docs.append(Document(page_content=content, metadata={"source": "iCliniq"}))
        print(f"     {len(docs) - count_before} documents chargés.")
    except Exception as e:
        print(f"     Erreur : {e}")
    print(f"\nTotal : {len(docs)} documents médicaux chargés.")
    if len(docs) == 0:
        raise ValueError("Aucun document chargé ! Vérifiez votre connexion internet.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"{len(chunks)} chunks créés.")
    return chunks

INDEX_PATH = "faiss_medical_index"

def construire_ou_charger_index():
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32}
    )
    if os.path.exists(INDEX_PATH):
        print("Index FAISS existant — chargement rapide...")
        vectorstore = FAISS.load_local(INDEX_PATH, embedder, allow_dangerous_deserialization=True)
    else:
        print("Construction de l'index FAISS (5-10 min)...")
        chunks = charger_documents()
        vectorstore = FAISS.from_documents(chunks, embedder)
        vectorstore.save_local(INDEX_PATH)
        print("Index sauvegardé.")
    return vectorstore, embedder

PROMPT_TEMPLATE = PromptTemplate.from_template(
    """Tu es un assistant médical d'information fiable et bienveillant.
Utilise UNIQUEMENT le contexte médical ci-dessous pour répondre à la question.
Si l'information n'est pas dans le contexte, dis clairement que tu ne sais pas.
Ne pose JAMAIS de diagnostic. Conseille toujours de consulter un médecin.
Réponds en français, de façon claire et accessible.

Contexte médical :
{context}

Question du patient : {question}

Réponse :"""
)

def formater_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def construire_rag(vectorstore):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    rag_chain = (
        {"context": retriever | formater_docs, "question": RunnablePassthrough()}
        | PROMPT_TEMPLATE
        | llm
        | StrOutputParser()
    )
    return rag_chain, retriever

print("Initialisation du chatbot médical...")
vectorstore, embedder = construire_ou_charger_index()
rag_chain, retriever = construire_rag(vectorstore)
print("Chatbot prêt !")

def repondre(message: str, history: list) -> str:
    if not message.strip():
        return "Veuillez poser une question médicale."
    if detecter_urgence(message):
        return MSG_URGENCE
    try:
        source_docs = retriever.invoke(message)
        sources = list(set([doc.metadata.get("source", "MedQuAD") for doc in source_docs]))
        reponse = rag_chain.invoke(message)
        if sources:
            reponse += f"\n\n *Source : {', '.join(sources)}*"
        return reponse
    except Exception as e:
        return f"Erreur : {str(e)}\n\nVérifiez que GROQ_API_KEY est dans le fichier .env"

with gr.Blocks(title="Assistant Médical RAG") as app:
    gr.Markdown("# Assistant Médical ")
    gr.Markdown(
        "> Cet assistant fournit des informations médicales générales uniquement. "
        "Il ne remplace pas l'avis d'un professionnel de santé. "
        "En cas d'urgence, appelez le **15** ou le **112**."
    )
    gr.ChatInterface(
        fn=repondre,
        chatbot=gr.Chatbot(height=450, label="Conversation"),
        textbox=gr.Textbox(placeholder="Posez votre question médicale ici...", label="Votre question"),
        examples=[
            "Quels sont les symptômes du diabète de type 2 ?",
            "Comment traiter une migraine ?",
            "Quelles sont les causes de l'hypertension ?",
            "Qu'est-ce que l'asthme et comment le gérer ?",
            "Quels sont les symptômes d'une grippe ?",
        ],
        submit_btn="Envoyer",
    )
    

if __name__ == "__main__":
    app.launch(share=False, server_port=7860, theme=gr.themes.Soft())