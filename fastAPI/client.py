'''Dummy frontend application for testing the API (like a mobile app or web app) that interacts with Swagger UI from app.py'''

import requests                                 # used to make HTTP requests to the FastAPI server
import streamlit as st

## function to get response from the OpenAI model
def get_openai_response(input_text):    
    response=requests.post(                     # making POST request to the FastAPI server
    "http://localhost:8000/essay/invoke",       # API URL for OpenAI model with /essay path
    json={'input':{'topic':input_text}})        # input to the OpenAI model as {topic} is the variable in prompt

    return response.json()['output']['content'] # converting output from the OpenAI model from JSON to string which is in key-value pairs, one key is 'output' and other is 'content'

## function to get response from the LM Studio model
def get_lmstudio_response(input_text):    
    response=requests.post(                     # making POST request to the FastAPI server
    "http://localhost:8000/poem/invoke",        # API URL for LM Studio model with /poem path
    json={'input':{'topic':input_text}})        # input to the LM Studio model as {topic} is the variable in prompt

    print("STATUS:", response.status_code)
    print("TEXT:", response.text)

    return response.json()['output']['content']

## streamlit framework
st.title('Langchain Demo With LM Studio API')
input_text1=st.text_input("Write an essay on")
input_text2=st.text_input("Write a poem on")

if input_text1:                                     # if input_text1 is not empty OR anything written in it
    st.write(get_openai_response(input_text1))      # call the function to get response from the OpenAI model

if input_text2:                                     # if input_text2 is not empty OR anything written in it
    st.write(get_lmstudio_response(input_text2))    # call the function to get response from the LM Studio model