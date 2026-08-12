import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
We Humanize Our Pets.
In many ways, we consider our canine companions to be important members of our families. Obviously, we name them just like we do our offspring. As the popular film tells us, we also believe they all go to heaven. Many people refer to themselves as “dog parents,” and bumper stickers proclaim their dogs are their virtual children. We confer notional (if sometimes vague) rights upon our pets, and organizations like PETA are dedicated to enunciating and preserving those rights. Fair enough, but why have these emotional bonds evolved and persisted across the 14 or so millennia since canine domestication began? We know that the presence of dogs benefits us, and vice versa, but how did our relationship evolve to the point of such deep bonds of affection? 
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)
split_documents = splitter.split_text(text)
print(f"Loaded {len(split_documents)} split documents.")
for i, doc in enumerate(split_documents):
    print(f"Split document {i+1} content:")
    print(doc)  

