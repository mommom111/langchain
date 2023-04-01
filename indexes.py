from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader

loader = TextLoader('')
documents = loader.load()
text_splitter = Character.TextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()

db = FAISS.from_documents(docs, embeddings)
query = ""
docs = db.similarity_search(query)
docs[0]