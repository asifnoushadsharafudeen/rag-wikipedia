import os
import wikipedia
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document

# Constants
DOCS_DIR = "docs"
VECTOR_DIR = "embeddings"

# Ensure folders exist
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

# Step 1 - Fetch Wikipedia article
def fetch_and_save_wikipedia_article():
    topic = input("Enter the Wikipedia topic to search: ")
    try:
        summary = wikipedia.page(topic).content
        filename = topic.lower().replace(" ", "") + ".txt"
        filepath = os.path.join(DOCS_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"📄 Wikipedia content saved to {filepath}")
    except wikipedia.exceptions.DisambiguationError as e:
        print("❌ Multiple pages found. Please refine your topic.")
    except Exception as e:
        print(f"❌ Error: {e}")

# Step 2 - Convert saved text to vector DB
def embed_and_save_vector_store():
    filename = input("Enter filename (e.g., india.txt): ")
    filepath = os.path.join(DOCS_DIR, filename)
    if not os.path.exists(filepath):
        print("❌ File does not exist.")
        return
    
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_text(text)
    docs = [Document(page_content=t) for t in texts]

    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embedding)
    db.save_local(VECTOR_DIR)
    print("📦 Vector store saved successfully to embeddings")

# Step 3 - Ask questions using local LLM and saved vector DB
def ask_question():
    print("🔍 Loading vector store and local model for QA...")

    # Load embedding and vector DB
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    try:
        db = FAISS.load_local(VECTOR_DIR, embedding, allow_dangerous_deserialization=True)
    except Exception as e:
        print("❌ Could not load vector store. Did you run Option 2 first?")
        return

    # Set up the local LLM
    local_pipeline = pipeline(
        "text2text-generation",  # ✅ Changed from 'text-generation' for Flan-T5 models
        model="google/flan-t5-base",
        tokenizer="google/flan-t5-base",
        max_length=256,
        do_sample=False,
    )
    llm = HuggingFacePipeline(pipeline=local_pipeline)

    # ✅ Modern QA chain (retriever + LLM)
    from langchain.chains import RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(),
        return_source_documents=False
    )

    while True:
        query = input("\nEnter your question based on the Wikipedia document (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        result = qa_chain.invoke({"query": query})
        print(f"\n🧠 Answer:\n {result['result']}")


# === Main menu ===
def main():
    while True:
        print("\n=== RAG Wikipedia QA System ===")
        print("1. Fetch Wikipedia article and save to file")
        print("2. Embed text from saved file")
        print("3. Ask question (QA)")
        print("4. Exit")
        choice = input("Enter your choice (1/2/3/4): ")

        if choice == "1":
            fetch_and_save_wikipedia_article()
        elif choice == "2":
            embed_and_save_vector_store()
        elif choice == "3":
            ask_question()
        elif choice == "4":
            break
        else:
            print("❌ Invalid choice. Try again.")

if __name__ == "__main__":
    main()
