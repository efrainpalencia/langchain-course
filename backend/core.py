import os
from typing import Any, Dict
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain_core.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings

load_dotenv()

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize vectorstore
vectorstore = PineconeVectorStore(
    index_name=os.getenv("INDEX_NAME"), embedding=embeddings
)

# initialize chat model
model = init_chat_model(model="gpt-5.2", model_provider="openai")

# Retrieval function tool with context


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to help us answer user quries about LangChain"""

    # Retrieve top 4 most similar documents
    retrieved_docs = vectorstore.as_retriever().invoke(query, k=4)

    # Serialize documents for the model
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )

    # Return both serialized conent and raw documents
    return serialized, retrieved_docs


# Wrapper function to run LLM
def run_llm(query: str) -> Dict[str, Any]:
    """
    Run a RAG pipeline to answerv a query using retrieved documentation

    Args:
        query: The user's question

    Returns:
        Dictionary containing:
            - answer: The generated answer
            - context: List of retrieved documents
    """

    # Create agent with retrieval tool and system prompt
    system_prompt = (
        "You are a help AI assistant that answers questions about Langchan documentation."
        "You have access to a tool that retrieves relevant documentation."
        "Use the tool to access relevant information before answering questions."
        "Always cite the sources you use in your answers."
        "If you cannot find the answer in the retrieved documetation, say so."
    )

    agent = create_agent(model, tools=[
                         retrieve_context], system_prompt=system_prompt)

    # Build message list
    messages = [{"role": "user", "content": query}]

    # Invoke the agent
    response = agent.invoke({"messages": messages})

    # Extract the answer from the last AI message
    answer = response["messages"][-1].content

    # Extract the context documents from ToolMessage artifacts
    context_docs = []
    for message in response["messages"]:
        # Check if this is a ToolMessage with artifact
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            # The message should contain the list of document objects
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)
    return {
        "answer": answer,
        "context": context_docs
    }


if __name__ == '__main__':
    result = run_llm(query="What are deep agents?")
    print(result)
