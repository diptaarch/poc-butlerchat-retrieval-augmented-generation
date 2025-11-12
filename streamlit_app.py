import streamlit as st
import os
import dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_community.llms import Replicate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment
dotenv.load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Archipelago Assistant",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load vector database and retriever
@st.cache_resource
def load_retriever():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    archipelago_vector_db = Chroma(
        persist_directory="chroma_data/",
        embedding_function=embeddings
    )
    return archipelago_vector_db.as_retriever(k=3)

# Load LLM
@st.cache_resource
def load_model():
    return Replicate(
        model="ibm-granite/granite-3.3-8b-instruct",
        replicate_api_token=os.environ.get("REPLICATE_API_TOKEN"),
    )

# Set up the chain
@st.cache_resource
def create_chain():
    archipelago_template_str = """You are a warm, welcoming hospitality expert for Archipelago International, 
Southeast Asia's largest privately owned hospitality group with 13 award-winning hotel brands.

IMPORTANT: Answer ONLY the specific question asked. Do not generate additional Q&A pairs or answer other questions.

Your tone should be:
- Warm, friendly, and inviting like a gracious hotel host
- Professional yet relaxed and conversational
- Polite and courteous with a genuine smile in your words

Instructions:
1. Answer ONLY the question provided - nothing more
2. Use context to provide accurate, helpful information
3. Keep responses concise and engaging
4. If information is not in context, warmly suggest contacting Archipelago Customer Services
5. Do NOT list multiple Q&A pairs or answer questions beyond what was asked
6. DO NOT include any signatures, sign-offs, or closing remarks like "Warm Regards" or "[Your Name]"
7. Do NOT include your role or title at the end of the response
8. If user question in another languages than english. apologize that you can't answer it in that languages

Context information: {context}

Guest's Question: {question}

Your Response (answer only this question, nothing else, no signatures or closing remarks):"""

    archipelago_system_prompt = SystemMessagePromptTemplate(
        prompt=PromptTemplate(input_variables=["context"], template=archipelago_template_str)
    )
    messages = [archipelago_system_prompt]
    archipelago_prompt_template = ChatPromptTemplate(
        input_variables=["context", "question"], messages=messages
    )
    
    retriever = load_retriever()
    model = load_model()
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | archipelago_prompt_template
        | model
        | StrOutputParser()
    )
    return chain

# UI Layout
st.title("🏨 Archipelago International Assistant")
st.markdown("*Your friendly guide to Southeast Asia's largest privately owned hospitality group*")

col1, col2 = st.columns([3, 1])
with col2:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me about Archipelago International, our brands, properties, or services..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                chain = create_chain()
                response = chain.invoke(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Please ensure your REPLICATE_API_TOKEN is set correctly in .env")

# Sidebar info
with st.sidebar:
    st.header("📖 About This Assistant")
    st.markdown("""
    **Archipelago International** is Southeast Asia's largest privately owned hospitality group with 13 award-winning hotel brands.
    
    ### Ask About:
    - Hotel brands (ASTON, Huxley, ALANA, etc.)
    - Properties and locations
    - Services and amenities
    - Membership programs
    - Company information
    
    ### Tips:
    - Be specific in your questions
    - Ask about particular brands or properties
    - The assistant uses our knowledge base to provide accurate info
    """)
