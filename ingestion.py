import os
import asyncio
import ssl
from typing import Any, Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

import certifi
from logger import *
from dotenv import load_dotenv

load_dotenv()

# Configure SSL context to to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small",
                              show_progress_bar=True, chunk_size=50, retry_min_seconds=10)

# chroma = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
vectorstore = PineconeVectorStore(
    index_name="langchain-doc-index", embedding=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


async def main():
    """Main async function to orchestrate the entire process"""

    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "TavilCrawl: Starting to crawl documentation from https://docs.langchain.com/oss/python/langchain/")

    # Crawl the documentation site
    res = tavily_crawl.invoke({
        "url": "https://docs.langchain.com/oss/python/langchain/",
        "max_depth": 5,
        "extract_depth": "advanced",
        "instructions": "content on ai agents"
    })

    all_docs = [Document(page_content=result["raw_content"], metadata={
                         "source": result["url"]}) for result in res["results"]]
    log_success(
        f"TavilrCrawl: Successfully crawled {len(all_docs)} URLs from documentation website"
    )


if __name__ == "__main__":
    asyncio.run(main())
