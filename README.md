# ⚖️ Pakistan Legal Precedent & Case Management AI System

An intelligent **AI-powered legal assistant** designed to automate key tasks in the legal domain — including **case classification**, **case prioritization**, and **legal precedent retrieval** using **Retrieval-Augmented Generation (RAG)**.  
This tool combines NLP pipelines, machine learning, and LLM-based reasoning to support faster and smarter legal decision-making.

---

## 🚀 Features

### 🧾 Case Classification  
Automatically classifies uploaded or entered legal case text into **Civil**, **Criminal**, or **Constitutional** categories using trained ML pipelines.

### ⏳ Case Prioritization  
Predicts the **urgency level** of a case (**High**, **Medium**, **Low**) to help manage workload efficiently.

### 📚 Legal Precedent Search (RAG)  
Implements a **Retrieval-Augmented Generation (RAG)** pipeline that retrieves relevant past legal precedents using **vector embeddings** and **LLM-powered summarization**.

---

## 🧠 Tech Stack

- **Python 3.13+**
- **Streamlit** – for the user interface  
- **Scikit-learn** – for ML classification and stacking pipelines  
- **LangChain + ChromaDB** – for document retrieval and embeddings  
- **HuggingFace Embeddings** – for text vectorization  
- **ChatGroq API** – for LLM integration  
- **Pickle Pipelines** – for pre-trained ML models (`voting_pipeline.pkl`, `stacking_pipeline.pkl`, etc.)

---

## 📂 Project Structure

├── app.py                                # Main Streamlit app  
├── Case Cateogarization/                 # Classification model + encoder  
│   ├── voting_pipeline.pkl  
│   └── label_encoder.pkl  
├── Case Prioritization/                  # Priority prediction model + encoder  
│   ├── stacking_pipeline.pkl  
│   └── label_encoder.pkl  
├── Legal_Precedent_Search/               # RAG configuration files  
│   ├── embeddings_config.pkl  
│   ├── llm_config.pkl  
│   ├── prompt_template.pkl  
│   └── chroma_db/                        # Vector database  
├── .env                                  # Groq API Key  
└── requirements.txt                      # Dependencies  





