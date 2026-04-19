'''Cloud based model like OpenAI used for chatbot application''' 

from langchain_openai import ChatOpenAI                     # used to interact with OpenAI's chat models (need to use one's own API key)
from langchain_core.prompts import ChatPromptTemplate       # initializes prompt template for the model
from langchain_core.output_parsers import StrOutputParser   # can also use custom output parser like split, capitalize, etc

import streamlit as st                                      
import os
from dotenv import load_dotenv                              

load_dotenv()                                               # initialize env varibles to access them here

## fetching env data

# for LangSmith tracking - tracing and monitoring all the results on LangChain dashboard (generic code for all paid or open-source LLMs)
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY is not set")
os.environ["OPENAI_API_KEY"] = openai_api_key

# where all the monitoring results should be stored and we can see all the results on the dashboard
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
if langchain_api_key is None:
    raise ValueError("LANGCHAIN_API_KEY is not set")
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key

os.environ["LANGCHAIN_TRACING_V2"]="true"                   # for enabling tracing and monitoring


## Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user queries"),
        ("user","Question:{question}")
    ]
)

## streamlit framework
st.title('Langchain with OPENAI API')
input_text=st.text_input("Search the topic you want...")

## openAI LLM
llm=ChatOpenAI(model="gpt-3.5-turbo")

## Output Parser
output_parser=StrOutputParser()

## chaining everything together
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))