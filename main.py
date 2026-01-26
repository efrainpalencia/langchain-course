import os
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)

tools = [TavilySearch()]

system_prompt = """You are a helpful AI agent that uses tools to answer user queries.
BEFORE YOU ANSWER, always verify your answer using the tools available to you."""

agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)


def main():
    print("Create a ReAct agent!")

    response = agent.invoke(
        {"messages": HumanMessage(
            content="Search for 3 job positions on LinkedIn for AI Engineer in Fort Lauderdale, FL that use LangChain. Provide me the LinkedIn URLs")}
    )
    print(response)


if __name__ == "__main__":
    main()
