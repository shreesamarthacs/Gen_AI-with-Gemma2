import os
from dotenv import load_dotenv

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

#groq_api_key=os.getenv("groq_api_key")

load_dotenv()
# Load secrets using Streamlit's secrets management
langchain_api_key = st.secrets["LANGCHAIN_API_KEY"]
groq_api_key = st.secrets["groq_api_key"]
langchain_project = st.secrets.get("LANGCHAIN_PROJECT", "default_project_name")



## Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a history expert. Please respond to the question asked accordingly"),
        ("user","Question:{question}")
    ]
)

## streamlit framework
st.title("GenAI with LLaMA Model")
input_text=st.text_input("What question you have in mind?")


## Ollama Llama2 model
llm=ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-70b-8192",streaming=True)
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))


