'''RAG with ObjectBox VectorstoreDB and Llama 3.1 on local documents (PDFs)'''

## ObjectBox
# 1. Fast & Lightweight: 10X faster than any alternative and comes with an incredebly lightweight footprint.
# 2. Vector Search: It is the first on-device offline vector search; 100% cloud optional.
# 3. Sustainable: It reduces resource-use (CPU and memory) and latency, while increasing privacy and security.
# 4. Offline first: For the low-latency “always-on” experience. Develop applications that work on- and offline, unburdened by the need for a constant internet connection. 
# 5. Data Control: Self-host, deploy locally, and run on-premise ensure data sovereignty, compliance, and seamless performance even in low-connectivity environments. 
# 6. Data Sync: Bi-directional, offline-first data sync keeps data flowing across devices on the edge and from edge to cloud, cloud optional.
# It is a high-performance NoSQL database to store and query data on devices. It is designed for mobile and embedded applications, good choice for on-device AI applications that require fast and efficient data storage and retrieval. By using ObjectBox as the vector store in a RAG architecture, we can enable on-device offline vector search, allowing AI models to access relevant information without needing to rely on cloud-based services. This can lead to improved performance, reduced latency, and enhanced privacy and security for users.
## https://objectbox.io/ - refer for more info'''

import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings  
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_objectbox.vectorstores import ObjectBox

import time

from dotenv import load_dotenv
load_dotenv()

## Groq API Key
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key is None:
    raise ValueError("GROQ_API_KEY is not set")
os.environ["GROQ_API_KEY"] = groq_api_key

## Streamlit UI with LLM
st.title("RAG with ObjectBox VectorstoreDB, ChatGroq and Llama3.1")
llm=ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

## Vector Embedding and Objectbox Vectorstore DB - we are creating a function to initialize the vector store and related components, which will be called when the user clicks the "Documents Embedding" button in the Streamlit app. This function checks if the vectors key is not already in the session state, indicating that the vector store has not been initialized yet. If they haven't been initialized, it proceeds to create the embeddings, load the documents from a specified directory (using PyPDFDirectoryLoader), split the documents into chunks, and then create an ObjectBox vector store locally from the processed documents and embeddings.
def vector_embedding():

    if "vectors" not in st.session_state:
        st.session_state.embeddings=OllamaEmbeddings(model="nomic-embed-text-v2-moe")   # need to pass the complete model name for the embeddings to work else by default it uses llama2 embeddings 
        st.session_state.loader=PyPDFDirectoryLoader("./cricket_docs")         # Data Ingestion
        st.session_state.docs=st.session_state.loader.load()               
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:20])     # selecting the first 20 documents for processing with the text splitter - depends on the number of tokens in the documents
        st.session_state.vectors=ObjectBox.from_documents(st.session_state.final_documents,st.session_state.embeddings,embedding_dimensions=768)    # creating an ObjectBox vector store to convert all the processed docs and embeddingsinto a vector locally, specifying the embedding dimensions (768 here - as per our choice)

## creating a prompt template for the LLM, which includes the retrieved context and the user's question
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

## when the user clicks the "Documents Embedding" button on Streamlit app, it calls the vector_embedding function above to initialize the vector store and related components, thus creating the vector embeddings and setting up the ObjectBox vector store for document retrieval based on user queries
if st.button("Documents Embedding"):
    vector_embedding()
    st.write("ObjectBox Database is ready")              # providing feedback to the user on Streamlit app once the vector store is ready for use

input_prompt=st.text_input("Input your query from the documents")     # text input box in Streamlit app

if input_prompt:
    # creating the document chain and retrieval chain inside the if block to ensure that they are created only when the user submits a query, thus avoiding unnecessary initialization of the chains when the app first loads or when the user is just embedding the documents
    document_chain=create_stuff_documents_chain(llm,prompt)     # to get llm context information
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    
    start=time.process_time()                                   # to measure the response time of the retrieval and generation process
    response=retrieval_chain.invoke({'input':input_prompt})     # invoking the retrieval chain with the user's input prompt, which will retrieve relevant docs from the ObjectBox vector store based on the user's query and then use the doc chain to generate a response using the LLM
    st.write(f"Response time: {time.process_time()-start:.2f} seconds")
    st.write(response['answer'])

    # With a streamlit expander (meta data of the retrieved documents)
    with st.expander("Document Similarity Search"):
        # find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")