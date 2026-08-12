import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.retrievers import WikipediaRetriever

retrievers = WikipediaRetriever(
    top_k_results=2,
    lang="en"
)

docs = retrievers.invoke("Labrador Retriever")
print(f"Retrieved {len(docs)} documents.")
for i, doc in enumerate(docs):
    print(f"Document {i+1} content:")
    print(doc.page_content)
    print(f"Document {i+1} metadata:")
    print(doc.metadata)

