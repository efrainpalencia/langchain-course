import os
from operator import itemgetter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()

print("Initializing components...")

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()

vectorstore = PineconeVectorStore(
    index_name=os.getenv("INDEX_NAME"), embedding=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the question only based on the following context: {context}

    Question: {question}

    Provide a detailed answer:
    """
)


def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def retrieval_chain_without_lcel(query: str):
    """Simple retriever without LCEL.
    Manually retrieves documents, formats them, and generates a response.
    """

    # Step 1: Retrieve relevant documents
    docs = retriever.invoke(query)

    # Step 2: Format documents into a string
    context = format_docs(docs)

    # Step 3:Format the prompt with context and question
    messages = prompt_template.format_messages(context=context, question=query)

    # Step 4: Invoke LLM with the formatted message
    response = llm.invoke(messages)

    # Step 5: Return the content
    return response.content


def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL (Langchain Expression Language).
    Returns a chain that can be invoked with {"question": "..." }
    """

    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )

        | prompt_template
        | llm
        | StrOutputParser()
    )

    return retrieval_chain


if __name__ == "__main__":
    print("Retrieving...")

    query = "What is Pinecone in machine learning?"

    result_without_lcel = retrieval_chain_without_lcel(query)
    print("====================================")
    print("Result without LCEL")
    print("====================================")
    print("\nAnswer:")
    print(result_without_lcel)

    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("====================================")
    print("Result with LCEL")
    print("====================================")
    print("\nAnswer:")
    print(result_with_lcel)
    print("debug stop")
