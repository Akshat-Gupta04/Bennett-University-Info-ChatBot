import sys
import os
import torch
import textwrap
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are Bennett InfoBot, an AI assistant trained to answer questions specifically "
        "about Bennett University. Only use the provided context to answer the question.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "If the question is unrelated to Bennett University or cannot be answered with the "
        "provided context, respond with: 'I can only answer questions related to Bennett University. Please refine your query.'"
    ),
)

def create_faiss_database(URLs,db_path):
    loaders=UnstructuredURLLoader(urls=URLs)
    data=loaders.load()
    text_splitter=CharacterTextSplitter(separator='\n',chunk_size=1500,chunk_overlap=200)
    text_chunks=text_splitter.split_documents(data)
    embedding=OpenAIEmbeddings()
    vectordb=FAISS.from_documents(documents=text_chunks,embedding=embedding)
    vectordb.save_local(db_path)
    return vectordb


def load_faiss_database(db_path):

    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.load_local(db_path, embeddings,allow_dangerous_deserialization=True)
    return vectordb




def create_retrieval_qa_chain(faiss_db):
    
    retriever = faiss_db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": custom_prompt},
    )
    return qa_chain