from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

cricketer_texts = [
    "Cricket is a popular sport played and followed by millions of people around the world.",
    "A cricket match is usually played between two teams, with each team having eleven players.",
    "The main objective is to score more runs than the opposing team.",
    "Players score runs by hitting the ball with a cricket bat and running between the wickets.",
    "The bowler tries to dismiss the batter by delivering the ball toward the wicket.",
    "A cricket team usually has batters, bowlers, and all-rounders who contribute in different ways.",
    "The wicketkeeper stands behind the stumps and tries to catch the ball when the batter misses it.",
    "Cricket matches can be played in different formats, including Test matches, One Day Internationals, and Twenty20 games.",
    "Test cricket can last up to five days, while a T20 match usually lasts only a few hours.",
    "Fielders play an important role by stopping runs and taking catches.",
    "Captains make important decisions about batting orders, bowling changes, and field placements during a match.",
    "Cricket is exciting because matches can change quickly through big partnerships, wickets, and close finishes." 
]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# Create a FAISS vector store from the cricketer_texts and embeddings
vector_store = FAISS.from_texts(cricketer_texts, embeddings)    

query = "What is cricket?"
# Perform a similarity search in the vector store
results = vector_store.similarity_search(query, k=3)
print("---------------------------------------------------")
print("Query:", query)
print("Top 3 similar texts:")
for i, result in enumerate(results):
    print(f"{i + 1}. {result.page_content}")    