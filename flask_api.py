from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import dotenv
from langchain_core.messages import SystemMessage
from langchain_community.llms import Replicate
from langchain_core.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment
dotenv.load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize components
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
archipelago_vector_db = Chroma(
    persist_directory="chroma_data/",
    embedding_function=embeddings
)
archipelago_retriever = archipelago_vector_db.as_retriever(k=3)

chat_model = Replicate(
    model="ibm-granite/granite-3.3-8b-instruct",
    replicate_api_token=os.environ.get("REPLICATE_API_TOKEN"),
)

# Set up the chain
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

archipelago_chain = (
    {"context": archipelago_retriever, "question": RunnablePassthrough()}
    | archipelago_prompt_template
    | chat_model 
    | StrOutputParser()
)


@app.route("/", methods=["GET"])
def home():
    """Health check and API info"""
    return jsonify({
        "status": "online",
        "service": "Archipelago International Assistant",
        "version": "1.0",
        "endpoints": {
            "health": "GET /health",
            "chat": "POST /api/chat",
            "info": "GET /"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Chat endpoint
    
    Request JSON:
    {
        "question": "Your question here"
    }
    
    Response JSON:
    {
        "question": "Your question",
        "answer": "The response from the assistant",
        "status": "success"
    }
    """
    try:
        data = request.get_json()
        
        if not data or "question" not in data:
            return jsonify({
                "error": "Missing 'question' field in request",
                "status": "error"
            }), 400
        
        question = data.get("question", "").strip()
        
        if not question:
            return jsonify({
                "error": "Question cannot be empty",
                "status": "error"
            }), 400
        
        # Get response from chain
        answer = archipelago_chain.invoke(question)
        
        return jsonify({
            "question": question,
            "answer": answer,
            "status": "success"
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
