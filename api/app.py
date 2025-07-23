from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# Initialize FastAPI app
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API server for essay and poem generation using OpenAI and LLaMA2."
)

# ChatOpenAI model for essay generation
openai_model = ChatOpenAI()
essay_prompt = ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words.")
essay_chain = essay_prompt | openai_model

# Ollama model for poem generation
llama_model = Ollama(model="llama2")
poem_prompt = ChatPromptTemplate.from_template("Write me a poem about {topic} with 100 words.")
poem_chain = poem_prompt | llama_model

# Add API routes
add_routes(app, openai_model, path="/openai")
add_routes(app, essay_chain, path="/essay")
add_routes(app, poem_chain, path="/poem")


# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
