import os
import pickle
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

class PersistentVectorStore:
    def __init__(self, storage_path="vectorstore.pkl"):
        self.storage_path = Path(storage_path)
        self.vectorstore = None
        self.load_vectorstore()

    def get_vectorstore(self, text_chunks, force_refresh=False):
        if self.vectorstore is None or force_refresh:
            embeddings = OpenAIEmbeddings()
            # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
            self.vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
            self.save_vectorstore()
        return self.vectorstore

    def load_vectorstore(self):
        if self.storage_path.exists():
            with open(self.storage_path, "rb") as file:
                self.vectorstore = pickle.load(file)

    def save_vectorstore(self):
        with open(self.storage_path, "wb") as file:
            pickle.dump(self.vectorstore, file)


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
#        pdf_reader = PdfReader(pdf)
        with open(pdf, 'rb') as f:
           pdf_reader = PdfReader(f)

        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


# Function to read multiple PDF files and return them as a list
def read_pdf_files_from_directory(directory):
    pdf_docs = []
    for file in os.listdir(directory):
        if file.endswith('.pdf'):
            file_path = os.path.join(directory, file)
            with open(file_path, 'rb') as f:
                pdf_docs.append(f.read())
    return pdf_docs


def main():
        load_dotenv()

        # Open the file
        file_path = 'mi.txt'
        file = open(file_path, 'r')

        # Read the contents
        raw_text = file.read()
#        raw_text = TextLoader("mi.txt")

        # Get the text chunks
        text_chunks = get_text_chunks(raw_text)

        # Create vector store
#        vectorstore = get_vectorstore(text_chunks)
        vector_store_manager = PersistentVectorStore()
        vectorstore = vector_store_manager.get_vectorstore(text_chunks)


if __name__ == '__main__':
    main()
