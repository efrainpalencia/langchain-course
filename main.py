from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from langchain_classic.agents.react.agent import create_react_agent
from langchain_classic.agents import AgentExecutor
from langsmith import Client
import os

from dotenv import load_dotenv

load_dotenv()

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4", temperature=0)

client = Client()
react_prompt = client.pull_prompt("hwchase17/react")

agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
chain = agent_executor


def main():
    response = chain.invoke(
        input={
            "input": "Search for 3 job listing on LinkedIn for an AI Engineer in the Broward County, Florida area and list their details."
        }
    )
    print(response)


if __name__ == "__main__":
    main()
