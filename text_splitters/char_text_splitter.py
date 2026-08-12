import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("docs/travel_guide.pdf")
documents = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0,
    length_function=len,
)
split_documents = splitter.split_documents(documents)

print(f"Loaded {len(split_documents)} split documents.")
print("First split document content:")
print(split_documents[0].page_content)
print("metadata:")
print(split_documents[0].metadata)