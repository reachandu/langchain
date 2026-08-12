import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("sample_western_life.pdf")
documents = loader.load()

print(f"Loaded {len(documents)} documents.")
print("First document content:")
print(documents[0].page_content)
print("First document metadata:")
print(documents[0].metadata)