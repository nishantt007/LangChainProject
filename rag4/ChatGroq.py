'''RAG using Groq, Langchain, FAISS, Ollama and Streamlit - A RAG chatbot leveraging Groq's Mixtral model, LangChain, and FAISS to provide responses based on the content of the documentation.'''

import streamlit as st
from langchain_groq import ChatGroq                                                 # similar to ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader              
from langchain_community.embeddings import OllamaEmbeddings                          
from langchain_text_splitters import RecursiveCharacterTextSplitter                 
from langchain_classic.chains.combine_documents import create_stuff_documents_chain # to combine the retrieved documents into a single context for the LLM to generate a response
from langchain_core.prompts import ChatPromptTemplate                               # to create a prompt template for the LLM, which includes the retrieved context and the user's question
from langchain_classic.chains import create_retrieval_chain                         # to create a retrieval chain that combines the retriever and the document chain, allowing us to retrieve relevant documents and generate a response in a single step
from langchain_community.vectorstores import FAISS              
import time                                                                        
import os

from dotenv import load_dotenv
load_dotenv()

## Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key is None:
    raise ValueError("GROQ_API_KEY is not set")
os.environ["GROQ_API_KEY"] = groq_api_key

## Streamlit UI with LLM
st.title("RAG with ChatGroq using Llama 3.1")
llm=ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")           # initializing the ChatGroq LLM with the Groq API key and the model

## initializing the vector store and document loader in the Streamlit session state to avoid reloading and reprocessing the documents on every interaction, thus ensuring that the embeddings and the vector store are created only once
## Session State - in Streamlit, every time we interact with the app (click a button, type in a box, slide a slider), the entire script runs again from top to bottom. Thus every variable we created is wiped clean and reset to its starting value. Session State acts as a persistent storage container that allows the app to remember data across those re-runs
if "vector" not in st.session_state:    # checking if the "vector" key is not already in the session state, which indicates that the vector store and related components have not been initialized yet. This ensures that the initialization code runs only once when the app is first loaded, rather than on every interaction
    
    st.session_state.embeddings=OllamaEmbeddings(model="nomic-embed-text-v2-moe")   # the OllamaEmbeddings class is used to generate vector embeddings for the documents, which are then stored in the FAISS vector store for similarity search
    st.session_state.loader=WebBaseLoader("https://docs.smith.langchain.com/")      
    st.session_state.docs=st.session_state.loader.load()                            # loading the documents from the session state (here - specified URL using the WebBaseLoader, can use PyPDFLoader for local docs, etc) 
    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)    
    st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:40]) # selecting the first 40 documents for processing with the text splitter - depends on the number of tokens in the documents. If the documents are too long, it may exceed the token limit of the LLM, so we can select a subset of the documents to ensure that we stay within the token limits
    st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

## creating a prompt template for the LLM, which includes the retrieved context and the user's question
prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
</context>
Questions:{input}

"""
)

prompt=st.text_input("Input your query related to LangSmith documentation")  # creating a text input box in the Streamlit app where users can enter their questions or prompts

## document chain & retrieval chain (outside the if prompt block because we want to create the chains only once and reuse them for every user query, rather than creating new chains every time a user submits a query. This avoids unnecessary reinitialization of the chains on every interaction)
document_chain = create_stuff_documents_chain(llm, prompt)              # creating a document chain that combines the LLM and the prompt template (doc chain = llm + prompt template)
retriever = st.session_state.vectors.as_retriever()                     # creating a retriever from the FAISS vector store, which will be used to retrieve relevant documents based on user queries
retrieval_chain = create_retrieval_chain(retriever, document_chain)     # creating a retrieval chain that combines the retriever and the document chain, allowing us to retrieve relevant documents and generate a response in a single step (retrieval chain = retriever + doc chain)

if prompt:
    start=time.process_time()                                           # starting a timer to measure the response time of the retrieval and generation process
    response=retrieval_chain.invoke({"input":prompt})                   # invoking the retrieval chain with the user's input prompt. This will trigger the process of retrieving relevant documents from the FAISS vector store based on the user's query and then generating a response using the LLM based on the retrieved context
    st.write(f"Response time: {time.process_time()-start:.2f} seconds")                  
    st.write(response['answer'])                                        # displaying the generated response from the LLM in the Streamlit app. The response is extracted from the 'answer' key of the response dictionary returned by the retrieval chain

    ## streamlit expander - making the loop go through every document in the retrieved context and display its content in the Streamlit app
    with st.expander("Document Similarity Search"):                     # creating an expander in the Streamlit app that allows users to view the relevant documents that were retrieved based on their query
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):                   # iterating through the retrieved context documents and displaying their content in the Streamlit app
            st.write(doc.page_content)
            st.write("--------------------------------")