import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_text_splitters import RecursiveCharacterTextSplitter, PythonCodeTextSplitter, Language

# sample python code to split
code = """
class Book:
    # A simple class to represent a book in a library.
    
    # Class attribute (shared by all instances)
    library_name = "City Central Library"

    # Constructor method to initialize instance attributes
    def __init__(self, title: str, author: str, pages: int):
        self.title = title          # Instance attribute
        self.author = author        # Instance attribute
        self.pages = pages          # Instance attribute
        self.is_checked_out = False # Default state

    # Instance method (action the object can perform)
    def check_out(self) -> str:
        if not self.is_checked_out:
            self.is_checked_out = True
            return f"'{self.title}' has been successfully checked out."
        return f"Sorry, '{self.title}' is already checked out."

    # Instance method using class attributes and data
    def get_description(self) -> str:
        return f"'{self.title}' by {self.author} ({self.pages} pages). Available at {self.library_name}."


# --- Using the Class (Creating Objects) ---

# 1. Create instances (objects) of the Book class
book1 = Book("The Hobbit", "J.R.R. Tolkien", 310)
book2 = Book("1984", "George Orwell", 328)

# 2. Access attributes using dot notation
print(book1.title)  # Output: The Hobbit
print(book2.author) # Output: George Orwell

# 3. Call methods on the objects
print(book1.get_description()) 
# Output: 'The Hobbit' by J.R.R. Tolkien (310 pages). Available at City Central Library.

print(book1.check_out())       
# Output: 'The Hobbit' has been successfully checked out.

print(book1.check_out())       
# Output: Sorry, 'The Hobbit' is already checked out.
"""

splitter = PythonCodeTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)
split_documents = splitter.split_text(code)
print(f"Loaded {len(split_documents)} split documents.")
for i, doc in enumerate(split_documents):
    print(f"Split document {i+1} content:")
    print(doc)

