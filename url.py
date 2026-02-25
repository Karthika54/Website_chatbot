import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
import os

st.set_page_config(page_title="Interactive QA App", page_icon="🧙‍♂️", layout="wide")

# Custom Styling
st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App Header
st.title("📚 Interactive QA App with Generative AI")
st.write("Ask detailed questions based on contextual data, and get accurate responses.")

# API Key Input
api_key = st.text_input(
    "Enter your Groq API key:",
    type="password",
    help="Get your free API key from https://console.groq.com",
)

if api_key:
    os.environ["GROQ_API_KEY"] = api_key

# --- Helper Functions ---

@st.cache_data
def get_web_text(url):
    loader = WebBaseLoader(url)
    documents = loader.load()
    return "\n\n".join([doc.page_content for doc in documents])

@st.cache_data
def get_text_chunks(text):
    # REDUCED: 1000 characters is roughly 250-300 tokens, 
    # leaving plenty of room for Groq's 6,000 token limit.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_text(text)

@st.cache_resource
def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return prompt | model

# --- Sidebar UI ---

st.sidebar.header("📝 Configuration")
url = st.sidebar.text_input("Enter the URL:", placeholder="https://en.wikipedia.org/wiki/Harry_Potter")

st.sidebar.header("📝 Ask a Question")
user_question = st.sidebar.text_area("Enter your question:", placeholder="e.g., Who is the protagonist?")

if st.sidebar.button("Get Answer"):
    if url and api_key and user_question:
        try:
            with st.spinner("Processing context and finding answer..."):
                # 1. Fetch and Chunk
                text = get_web_text(url)
                chunks = get_text_chunks(text)
                
                # 2. Create Vector Store
                vectorstore = get_vector_store(chunks)
                
                # 3. Retrieve relevant chunks (k=4 ensures we don't exceed token limits)
                docs = vectorstore.similarity_search(user_question, k=4)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # 4. Generate Answer
                chain = get_conversational_chain()
                response = chain.invoke({"context": context, "question": user_question})

            st.success("Done!")
            st.subheader("Answer:")
            st.write(response.content)

        except Exception as e:
            if "413" in str(e) or "rate_limit_exceeded" in str(e):
                st.error("Error: The context is still too large for the Groq Free Tier. Try a more specific question.")
            else:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please provide the API key, a URL, and a question.")

# Footer
st.markdown("---")
st.caption("🤖 Powered by LangChain, Groq (Llama 3.1) and HuggingFace Embeddings")