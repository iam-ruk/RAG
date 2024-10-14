from langchain_openai import AzureChatOpenAI
import streamlit as st
import os
import time
import numpy as np
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough



# find .env files
dotenv_path = find_dotenv()
# load environment variables
load_dotenv(dotenv_path)


# index_name = "rag-test"

# pc = Pinecone(api_key=PINECONE_API_KEY)

# llm = AzureChatOpenAI(
#             azure_endpoint = __AZURE_ENDPOINT,
#             api_key = __AZURE_OPENAI_API_KEY,
#             model = __OPENAI_GPT_MODEL,
#             api_version = __OPENAI_API_VERSION,
#             verbose = "True"
#         )

# existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

# if index_name not in existing_indexes:
#     pc.create_index(
#         name=index_name,
#         dimension=1536,
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1")
#     )
#     while not pc.describe_index(index_name).status["ready"]:
#         time.sleep(1)



# loader = TextLoader("virat_interview.txt")
# text_documents = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
# documents = text_splitter.split_documents(text_documents)




# print(setup.invoke("How Virat goes about his diet?"))



# print(chain.invoke("What virat mentions about anushkha?"))



st.markdown(
    r"""
    <style>
    .stDeployButton {
        visibility: hidden;
    }
    .stMainMenu {
        visibility: hidden;
    }
    .stStatusWidget {
        visibility: hidden;
    }
    .stAppToolbar {
        visibility: hidden;
    }
    </style>
    """, unsafe_allow_html=True
)

class RAGAgent:

    def __init__(self):
        #init env variables
        self.__OPENAI_GPT_MODEL = os.getenv("OPENAI_GPT_MODEL")
        self.__AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
        self.__OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
        self.__AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
        self.__AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT")
        self.__PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        self.__PINECONE_INDEX_NAME = os.getenv("INDEX_NAME")

        #instantiate model

        self.__llm = AzureChatOpenAI(
            azure_endpoint = self.__AZURE_ENDPOINT,
            api_key = self.__AZURE_OPENAI_API_KEY,
            model = self.__OPENAI_GPT_MODEL,
            api_version = self.__OPENAI_API_VERSION,
            verbose = "True"
        )

        self.__embeddings = AzureOpenAIEmbeddings(
                    azure_endpoint = self.__AZURE_ENDPOINT,
                    api_key = self.__AZURE_OPENAI_API_KEY,
                    azure_deployment = self.__AZURE_DEPLOYMENT
            )
        
        self.__pinecone = PineconeVectorStore.from_existing_index(
                            index_name = self.__PINECONE_INDEX_NAME, embedding = self.__embeddings
                        )
        
        self.__TEMPLATE = """
                        Answer the question based on the context below. If you can't 
                        answer the question, reply "I don't know".

                        Context: {context}

                        Question: {question}
                        """

        self.__prompt = ChatPromptTemplate.from_template(self.__TEMPLATE)
        self.__setup = RunnableParallel(context = self.__pinecone.as_retriever(), question = RunnablePassthrough())
        self.__parser = StrOutputParser()


        self.__chain = self.__setup | self.__prompt | self.__llm | self.__parser

    def get_agent(self):
        return self.__chain
    
    def stream_data(self, text):
        for word in text.split(" "):
            yield word + " "
            time.sleep(0.1)


rag_agent = RAGAgent()
agent = rag_agent.get_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def handle_user_prompts(prompt):
    with st.chat_message("user"):
        st.write(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})

    agent_response = agent.invoke(prompt)

    with st.chat_message("assistant"):
        st.write_stream(rag_agent.stream_data(agent_response))
    
    st.session_state.messages.append({"role": "assistant", "content": agent_response})



prompt = st.chat_input(placeholder = "Ask Something")
if prompt:
   handle_user_prompts(prompt)