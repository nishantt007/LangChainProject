'''RAG with Groq Inference Engine, free Google server's Generative AI Embeddings, Llama model aas LLM with Streamlit UI'''

import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain 
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader   # reads all the PDF files in the specified directory and extract their content for further processing (e.g., text splitting, embedding generation, etc.)
from langchain_google_genai import GoogleGenerativeAIEmbeddings         # vector mbedding technique - text to embeddings 

from dotenv import load_dotenv
import os
load_dotenv()

## load the GROQ And OpenAI API KEY 
groq_api_key=os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")

st.title("Document Q&A with Llama 3.1 and Google Generative AI Embeddings")

llm=ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}
"""
)

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        st.session_state.loader=PyPDFDirectoryLoader("./cricket_docs")      ## Data Ingestion
        st.session_state.docs=st.session_state.loader.load()                ## Document Loading in session state 
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)            ## Chunk Creation
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:20]) ## splitting and embedding creation
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) ## vectorstore for embeddings

prompt1=st.text_input("Enter your question from the provided documents")


if st.button("Create Vector Store for the Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")

import time

if prompt1:
    document_chain=create_stuff_documents_chain(llm, prompt)    
    retriever=st.session_state.vectors.as_retriever()       ## interface to fetch the relevant docs from the vector store based on the user query
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    response=retrieval_chain.invoke({'input':prompt1})      ## invoking the retrieval chain with the user query
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):       # find relevant chunks
            st.write(doc.page_content)
            st.write("--------------------------------")