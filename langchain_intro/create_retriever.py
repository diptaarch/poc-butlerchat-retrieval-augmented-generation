import dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

ARCHIPELAGO_INFO_PATH = "data/archipelago_info.csv"
ARCHIPELAGO_CHROMA_PATH = "chroma_data"

dotenv.load_dotenv()

loader = CSVLoader(file_path=ARCHIPELAGO_INFO_PATH, source_column="content")
archipelago_docs = loader.load()

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

archipelago_vector_db = Chroma.from_documents(
    archipelago_docs, embeddings, persist_directory=ARCHIPELAGO_CHROMA_PATH
)

question = """What brands does Archipelago International operate?"""

relevant_docs = archipelago_vector_db.similarity_search(question, k=3)

print(relevant_docs[0].page_content)