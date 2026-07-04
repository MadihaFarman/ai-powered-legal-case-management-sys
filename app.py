import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


import streamlit as st
import os
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv

# Load environment variables (Groq API, etc.)
load_dotenv(dotenv_path=".env", override=True)

# Page config
st.set_page_config(page_title="Legal Case Management & Precedent Search", layout="wide")

# Custom UI Styling

st.markdown(
    """
    <style>
    .stApp { background-color: #F8F9FA; }
    section[data-testid="stSidebar"] { background-color: #2C2C2C; color: white; }
    section[data-testid="stSidebar"] * { color: white !important; }
    div.stButton > button {
        background-color: #FFFFFF !important;
        border-radius: 10px;
        border: 2px solid #C9A227 !important;
        padding: 0.6em 1.2em;
        font-size: 16px !important;
        font-weight: 700 !important;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #C9A227 !important;
        color: #FFFFFF !important;
        border: 2px solid #1A2B4C !important;
    }
    h1, h2, h3, h4 { color: #1A2B4C; font-family: 'Georgia', serif; }
    .stMarkdown, p, label { color: #000000 !important; font-family: 'Georgia', serif; }
    </style>
    """,
    unsafe_allow_html=True
)

# Ensure NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Helper Functions
def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^a-zA-Z0-9]", " ", text).lower()
    tokens = text.split()
    sw = set(stopwords.words('english'))
    lemm = WordNetLemmatizer()
    return " ".join([lemm.lemmatize(tok) for tok in tokens if tok not in sw])

@st.cache_resource
def load_pickle(path: str):
    if not os.path.exists(path):
        st.warning(f"Missing file: {path}")
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return None

# Sidebar

st.sidebar.title("⚖️ Legal AI Toolkit")
mode = st.sidebar.radio(
    "Choose a tool:",
    ("Home", "Case Classification", "Case Prioritization", "Legal Precedent Search (RAG)")
)

# Home Screen

if mode == "Home":
    st.title("AI Powered Legal Case Management & Precedent Search")
    st.markdown("""
        ### What this project does
        - **Case Classification**: Automatically classify court cases by category (Civil, Criminal, or Constitutional)
        - **Case Prioritization**: Predict the urgency level of cases (High, Medium, Low)
        - **Legal Precedent Search (RAG)**: Retrieve related case precedents using embeddings and similarity search
    """)
    st.info("Select a tool from the sidebar to get started.")
    st.stop()

# Case Classification

if mode == "Case Classification":
    st.title("⚖️ Case Classification")
    pipeline_path = "Case Cateogarization/voting_pipeline.pkl"
    label_path = "Case Cateogarization/label_encoder.pkl"

    with st.spinner("Loading classification model..."):
        pipeline = load_pickle(pipeline_path)
        label_encoder = load_pickle(label_path)

    text_input = st.text_area("Paste case text here:", height=300)

    if st.button("Predict Category"):
        if not text_input.strip():
            st.warning("Please enter some case text.")
        elif pipeline is None:
            st.error("Pipeline not loaded.")
        else:
            cleaned = preprocess_text(text_input)
            pred_enc = pipeline.predict([cleaned])
            pred_label = label_encoder.inverse_transform(pred_enc)[0] if label_encoder else str(pred_enc[0])
            st.success(f"Predicted Case Category: **{pred_label}**")

# Case Prioritization

if mode == "Case Prioritization":
    st.title("⚖️ Case Prioritization")
    pipeline_path = "Case Prioritization/stacking_pipeline.pkl"
    label_path = "Case Prioritization/label_encoder.pkl"

    with st.spinner("Loading prioritization model..."):
        pipeline = load_pickle(pipeline_path)
        label_encoder = load_pickle(label_path)

    text_input = st.text_area("Paste case text here:", height=300)

    if st.button("Predict Priority"):
        if not text_input.strip():
            st.warning("Please enter some case text.")
        elif pipeline is None:
            st.error("Pipeline not loaded.")
        else:
            cleaned = preprocess_text(text_input)
            pred_enc = pipeline.predict([cleaned])
            pred_label = label_encoder.inverse_transform(pred_enc)[0] if label_encoder else str(pred_enc[0])
            st.success(f"Predicted Case Priority: **{pred_label}**")

# Legal Precedent Search (RAG)

if mode == "Legal Precedent Search (RAG)":
    st.title("📚 Legal Precedent Retrieval Engine (RAG)")
    st.markdown("Ask a question like: *What were previous precedents regarding X?*")

    base_dir = "Legal_Precedent_Search"
    PERSIST_DIRECTORY = os.path.join(base_dir, "chroma_db")

    @st.cache_resource
    def load_rag_components(base_dir: str):
        try:
            from langchain_chroma import Chroma
            from langchain_huggingface.embeddings import HuggingFaceEmbeddings
            from langchain_groq import ChatGroq
            from langchain_core.prompts import ChatPromptTemplate
            from langchain.chains.combine_documents import create_stuff_documents_chain
            from langchain.chains import create_retrieval_chain
        except Exception as e:
            return {"error": f"Missing dependencies: {e}"}

        paths = {
            "embeddings": os.path.join(base_dir, "embeddings_config.pkl"),
            "llm": os.path.join(base_dir, "llm_config.pkl"),
            "prompt": os.path.join(base_dir, "prompt_template.pkl")
        }

        # Load embeddings
        try:
            with open(paths["embeddings"], "rb") as f:
                embeddings_config = pickle.load(f)
            embeddings = HuggingFaceEmbeddings(model_name=embeddings_config.get("model_name"))
        except Exception as e:
            return {"error": f"Error loading embeddings: {e}"}

        # Load llm config
        try:
            with open(paths["llm"], "rb") as f:
                llm_config = pickle.load(f)
        except Exception as e:
            return {"error": f"Error loading LLM config: {e}"}

        # Load prompt
        try:
            with open(paths["prompt"], "rb") as f:
                prompt = pickle.load(f)
        except Exception as e:
            return {"error": f"Error loading prompt: {e}"}

        if not os.path.exists(PERSIST_DIRECTORY):
            return {"error": f"ChromaDB folder not found at {PERSIST_DIRECTORY}"}

        vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
        api_key = os.getenv("api_key") or os.getenv("API_KEY")
        if not api_key:
            return {"error": "Groq API key not found in .env file."}

        llm = ChatGroq(model_name=llm_config["model_name"], api_key=api_key, temperature=0.2)
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vectordb.as_retriever(search_kwargs={"k": 5})
        rag_chain = create_retrieval_chain(retriever, document_chain)
        return {"rag_chain": rag_chain}

    with st.spinner("Loading RAG components..."):
        rag_resources = load_rag_components(base_dir)

    if "error" in rag_resources:
        st.error(rag_resources["error"])
        st.stop()

    rag_chain = rag_resources["rag_chain"]
    query = st.text_area("Enter your legal question:", height=150)
    if st.button("Search Precedents"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving precedents..."):
                try:
                    response = rag_chain.invoke({"input": query})
                    answer = response.get("answer") if isinstance(response, dict) else str(response)
                    st.subheader("Answer:")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Error during retrieval: {e}")