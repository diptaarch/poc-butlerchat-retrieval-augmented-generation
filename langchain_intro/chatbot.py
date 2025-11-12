import os
import dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.llms import Replicate
from langchain_core.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from tools import get_hotel_information, get_brand_details

## Vector Databases && Embeddings
ARCHIPELAGO_CHROMA_PATH = "chroma_data/"
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
archipelago_vector_db = Chroma(
     persist_directory=ARCHIPELAGO_CHROMA_PATH,
     embedding_function=embeddings
)
archipelago_retriever = archipelago_vector_db.as_retriever(k=3)


## Models 
dotenv.load_dotenv()
chat_model = Replicate(
    model="ibm-granite/granite-3.3-8b-instruct",
    replicate_api_token=os.environ.get("REPLICATE_API_TOKEN"),
)


## Prompting
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
archipelago_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(input_variables=["question"], template="")
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


def call_tool(question: str) -> str:
    """Process user question and return appropriate response about Archipelago International"""
    # question_lower = question.lower().strip()
    
    # Use the archipelago chain to answer questions
    return archipelago_chain.invoke(question)




if __name__ == "__main__":
    print("� Archipelago International Assistant")
    print("=" * 50)
    print("Welcome! I'm here to help you with questions about")
    print("Archipelago International - Southeast Asia's largest")
    print("privately owned hospitality group with 13 award-winning brands.")
    print("\nAsk me about:")
    print("  • Hotel brands (ASTON, Huxley, ALANA, etc.)")
    print("  • Properties and locations")
    print("  • Services and amenities")
    print("  • Membership programs")
    print("  • Company information and history")
    print("  • Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ["quit", "exit", "q"]:
            print("\n✨ Thank you for choosing Archipelago International!")
            print("We look forward to welcoming you soon! 🏨\n")
            break
        
        if not user_input:
            continue
        
        print("\nAssistant: ", end="", flush=True)
        try:
            response = call_tool(user_input)
            print(response)
        except Exception as e:
            print(f"Apologies! We encountered a small issue: {str(e)}")
        
        print()