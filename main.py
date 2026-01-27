import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage, SystemMessage
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

load_dotenv()


def call_agent(input: str):

    class Source(BaseModel):
        """Schema for agents to return source information."""

        url: str = Field(description="The URL of the source")

    class AgentResponse(BaseModel):
        """Schema for agent responses."""

        answer: str = Field(description="The answer to the user's query")
        sources: list[Source] = Field(
            description="A list of sources used to generate the answer"
        )

    tools = [TavilySearch()]

    system_prompt = "You are a helpful assistant that uses Tavily to help generate leads."

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    agent = create_agent(model=llm, tools=tools,
                         system_prompt=system_prompt, response_format=AgentResponse)

    response = agent.invoke({"messages": HumanMessage(content=input)})

    return response


def main():
    user = str(input("Enter your query (press Enter to exit):"))

    if user != "":
        response = call_agent(user)
        print(response)


if __name__ == "__main__":
    main()
