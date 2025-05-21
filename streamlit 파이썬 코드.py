import streamlit as st
from dotenv import load_dotenv

from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
import faiss
import os

#load_dotenv()

os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']

@st.cache_resource
def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()

@st.cache_resource
def create_vector_store(_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
    split_docs = text_splitter.split_documents(_docs)

    vectorstore = FAISS.from_documents(
        split_docs,
        OpenAIEmbeddings(model='text-embedding-3-small')
    )
    faiss.write_index(vectorstore.index, "faiss_index.index")
    return vectorstore

def get_vector_store(_docs):
    if os.path.exists("faiss_index"):
        return FAISS.load_local("faiss_index", OpenAIEmbeddings(model='text-embedding-3-small'))
    else:
        return create_vector_store(_docs)
    
@st.cache_resource
def initialize_components(selected_model):
    file_path = "C:/Users/sbin0/Desktop/3-1/인공지능서비스개발/대한민국헌법(헌법)(제00010호)(19880225).pdf"
    pages = load_and_split_pdf(file_path)
    vectorstore = get_vector_store(pages)
    retriever = vectorstore.as_retriever()

    contextualize_q_system_prompt = """Given a chat history and latest user question\
        which might reference context in the chat history, formulate a standalone question\
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    qa_system_prompt = """
    You are an assistant for question-answering tasks.\
    Use the folloeing pieces of retrieved context to answer the question.\
    If you don't know the answer, just say that you don't know.\
    Keep the answer perfect, please use imogi with the answer.\
    대답은 한국어로 하고, 존댓말을 써줘.\
    
    {context}"""

    qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
    )

    contextualize_q_system_prompt = """Given a chat history and latest user question\
    which might reference context in the chat history, formulate a standalone question\
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is"""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
    )

    llm = ChatOpenAI(model=selected_model)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain

st.header("헌법 Q&A 챗봇")
option = st.selectbox("Select GPT Model", ("gpt-4o", "gpt-3.5-turbo-0125"))
rag_chain = initialize_components(option)
chat_history = StreamlitChatMessageHistory(key="chat_messages")

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role":"assistant", "content":"헌법에 대해 무엇이든 물어보세요!"}]

for msg in chat_history.messages:
    st.chat_message(msg.type).write(msg.content)

if prompt_message := st.chat_input("Your question"):
    st.chat_message("human").write(prompt_message)
    with st.chat_message("ai"):
        with st.spinner("Thinking....."):
            config = {"configurable":{"session_id":"any"}}
            response = conversational_rag_chain.invoke(
                {"input":prompt_message},
                config
            )
            answer = response['answer']
            st.write(answer)

            with st.expander("참고 문서 확인"):
                for doc in response['context']:
                    st.markdown(doc.metadata['source'], help=doc.page_content)
