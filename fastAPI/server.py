'''For creatng all the APIs and getting Swagger UI for testing in browser'''

from langchain_core.prompts import ChatPromptTemplate    
from langchain_openai import ChatOpenAI             # used to interact with OpenAI's chat models (need to use one's own API key)
## FastAPI is a web framework for building REST APIs. These APIs provide the backend logic for mobile, web, and desktop applications that make API calls to our FastAPI app endpoints.
from fastapi import FastAPI                         # fast (high-performance), web framework for building APIs with Python
from langserve import add_routes                    # used to add routes to the FastAPI app - can use for one route to paid LLM, another to local, etc.
import uvicorn                                      # used to run the FastAPI app                                          
from langchain_community.llms import Ollama         # used to interact with all third-party integration like- Ollama, LMStudio, Vector embeddings, etc.
import os 

from dotenv import load_dotenv                      
load_dotenv()                                       

openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY is not set")
os.environ["OPENAI_API_KEY"] = openai_api_key        

app=FastAPI(                                        # initializing FastAPI app
    title="Langchain Server",                       
    version="1.0",                                  
    decsription="A simple API Server"               
)

### creating routes in a manner so that prompt template also integrated with it
## route for OpenAI model
add_routes(                                         # adding routes to the FastAPI app
    app,                                            # FastAPI app
    ChatOpenAI(),                                   # model in use - OpenAI chat model
    path="/openai"                                  # path to the route of OpenAI model
)

## OpenAI actual chat model initialized
model1=ChatOpenAI() 

## LM Studio chat model (OpenAI-compatible local endpoint)
model2=ChatOpenAI(                                  
    base_url="http://localhost:1234/v1",            # LM Studio's local server URL
    api_key="lm-studio",                            # LM Studio doesn't need a real key, any string works
    model="mistralai/ministral-3-14b-reasoning"     # API Model identifier in LM Studio
)

prompt1=ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")
prompt2=ChatPromptTemplate.from_template("Write me an poem about {topic} for a 5 years child with 100 words")

## route for OpenAI model with prompt
add_routes(                                         # adding routes to the FastAPI app  
    app,                                            # FastAPI app
    prompt1|model1,                                 # prompt and OpenAI model
    path="/essay"                                   # path to the route of OpenAI model - API URL will end with /essay - http://localhost:8000/essay/invoke
)

## route for LM Studio model with prompt
add_routes(                                         # adding routes to the FastAPI app  
    app,                                            # FastAPI app
    prompt2|model2,                                 # prompt and LM Studio model
    path="/poem"                                    # path to the route of LM Studio model - API URL will end with /poem - http://localhost:8000/poem/invoke
)

if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)     # running the FastAPI app
