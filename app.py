import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
from collections import Counter

from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.set_page_config(page_title="AI Knowledge Twin")
st.title("🧠 AI Knowledge Twin")

# 🔥 Track user questions
if "question_history" not in st.session_state:
    st.session_state.question_history = []

uploaded_file = st.file_uploader("Upload your PDF notes", type="pdf")

if uploaded_file is not None:
    # Save file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    # Split text
    splitter = CharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    chunks = chunks[:20]  # keep lightweight

    st.success(f"✅ Processed {len(chunks)} chunks")

    try:
        # Embeddings (local, no API)
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        # Vector DB
        db = FAISS.from_documents(chunks, embeddings)

        st.success("🧠 Knowledge ready!")

        # 🔍 User input
        query = st.text_input("💬 Ask a question from your notes")
        ask = st.button("Ask")

        # 🤖 Groq LLM function
        def ask_groq(query, context):
            response = client.chat.completions.create(
                model="Llama-3.3-70B-Versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "Answer only based on the given context. Be clear and concise."
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {query}"
                    }
                ]
            )
            return response.choices[0].message.content

        # 📊 Weak area detection
        def detect_weak_areas():
            words = []
            for q in st.session_state.question_history:
                words.extend(q.lower().split())

            common = Counter(words).most_common(3)
            return [word for word, _ in common if len(word) > 4]

        # 🚀 Q&A Flow
        if ask and query:
            st.session_state.question_history.append(query)

            docs = db.similarity_search(query, k=3)
            context = " ".join([doc.page_content for doc in docs])

            answer = ask_groq(query, context)

            st.write("🧠 Answer:")
            st.write(answer)

            # 🔥 Weak area insight
            weak_topics = detect_weak_areas()
            if weak_topics:
                st.warning(f"⚠️ You may need to revise: {', '.join(weak_topics)}")

        elif ask and not query:
            st.warning("Please enter a question.")

    except Exception as e:
        st.error(f"Error: {e}")