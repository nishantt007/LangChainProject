'''Local language model used for chatbot application'''

from langchain_openai import ChatOpenAI                     # used to interact with OpenAI's chat models (need to use one's own API key)
from langchain_core.prompts import ChatPromptTemplate       # initializes prompt template for the model
from langchain_core.output_parsers import StrOutputParser   # can also use custom output parser like split, capitalize, etc
from langchain_community.llms import Ollama                 # all third-party integration is available in community like- Ollama, LMStudio, Vector embeddings, etc.

import streamlit as st                                      
import os
from dotenv import load_dotenv                              

load_dotenv()                                               # initialize env varibles to access them here

## fetching env data

## for LangSmith tracking - tracing and monitoring all the results on LangChain dashboard (generic code for all paid or open-source LLMs)
# where all the monitoring results should be stored and we can see all the results on the dashboard
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
if langchain_api_key is None:
    raise ValueError("LANGCHAIN_API_KEY is not set")
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key

os.environ["LANGCHAIN_TRACING_V2"]="true"                       # for enabling tracing and monitoring


## Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user queries"),
        ("user","Question:{question}")                                                  
    ]
)

## streamlit framework
st.title('Langchain with Local LM Studio API')
input_text=st.text_input("Search any topic you want...")

## LM Studio runs a local OpenAI-compatible model (or we can put in any paid LLM API)
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",                        # LM Studio's local server URL
    api_key="lm-studio",                                        # LM Studio doesn't need a real key, any string works
    model="qwen3.5-4b-claude-4.6-opus-reasoning-distilled",     # API Model identifier in LM Studio
    temperature=0.8,
)

## Output Parser
output_parser=StrOutputParser()

## chaining everything together
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))