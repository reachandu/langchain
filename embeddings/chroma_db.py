from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

texts = [
    "Investing in stocks can be a good way to build wealth over the long term.",
    "Stocks represent ownership in companies, allowing investors to benefit when those companies grow.",
    "Before investing, it is important to research a company's financial performance and future prospects.",
    "Diversifying investments across different companies and industries can help reduce risk.",
    "Investors should understand that stock prices can rise and fall due to market conditions and economic events.",
    "Long-term investing and regular contributions can help investors take advantage of market growth over time.",
    "It is important to choose investments that match your financial goals, risk tolerance, and investment time horizon."
]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# Create a Chroma vector store from the texts and embeddings
vector_store = Chroma.from_texts(texts, embeddings, collection_name="stock_investing")

query = "What are some tips for investing in stocks?"
# Perform a similarity search in the vector store
results = vector_store.similarity_search(query, k=3)
print("---------------------------------------------------")
print("Query:", query)
print("Top 3 similar texts:")
for i, result in enumerate(results):
    print(f"{i + 1}. {result.page_content}")
    