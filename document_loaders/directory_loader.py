import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader("docs", glob="*.pdf", loader_cls=PyPDFLoader)

# documents = loader.load()

# print(f"Loaded {len(documents)} documents.")
# print("First document content:")
# print(documents[0].page_content)
# print("First document metadata:")
# print(documents[0].metadata)

docs = loader.lazy_load()
for i, doc in enumerate(docs):
    print(f"Document {i+1} metadata:")
    print(doc.metadata)