import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader 

# Actualización de imports para embeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings

#Memoria de conversacion
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI, HuggingFaceHub

# Actualización de FAISS
import os
import pickle
from langchain_community.vectorstores import FAISS


# Inyeccion de Html
from htmlTemplate import bot_template, css, user_template



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    return text

def get_text_chunks(raw):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(raw)
    return chunks


# Construcción de embeddings mediante servidores de OPEN AI
def get_vector_store_OPENAI(text_chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(texts=text_chunks, embedding = embeddings)
    return vector_store

#Construcción de embedings mediante instructor, hechos con mi cpu/gpu.
def get_vector_store_Instructor(text_chunks, save_path='vectorstore.pkl'):
 
    """ if os.path.exists(save_path):
        print(f"Cargando vector store desde {save_path}...")
        with open(save_path, 'rb') as f:
            vectorstore = pickle.load(f)
        return vectorstore """
    
    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
     # Guardar el vectorstore en un archivo

    """ with open(save_path, 'wb') as f:
        pickle.dump(vectorstore, f) """

    return vectorstore

def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":0.5, "max_length":512})

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
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)



def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    #st.write(user_template.replace("{{MSG}}", "Hellow Robot"), unsafe_allow_html=True)
    #st.write(bot_template.replace("{{MSG}}", "Hellow human"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Documentos")
        pdf_docs = st.file_uploader("Sube tus pdfs ", accept_multiple_files=True)
        if st.button("Procesar"):
            with st.spinner("Procesando: "):
                # Sacar datos de pdf
                raw = get_pdf_text(pdf_docs)
                # Convertilos en particiones
                text_chunks = get_text_chunks(raw)
                #st.write(text_chunks)
                
                # Crear bd vectorial y embeddings
                vector_store = get_vector_store_Instructor(text_chunks)

                #Crear conversacion
                st.session_state.conversation = get_conversation_chain(vector_store)




if __name__ == '__main__':
    main()