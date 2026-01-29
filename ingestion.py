import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from dotenv import load_dotenv

load_dotenv()


if __name__ == "__main__":

    print("Ingestion...")

    loader = TextLoader(
        "C:/dev/practice/udemy/langchain-course/mediumblog1.txt", autodetect_encoding=True)

    document = loader.load()

    print("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    text = text_splitter.split_documents(document)
    print(f"created {len(text)} chunks")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))

    print("Ingesting...")
    PineconeVectorStore.from_documents(
        text, embeddings, index_name=os.getenv("INDEX_NAME"))
    print("finished")
