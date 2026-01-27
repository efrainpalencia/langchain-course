
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from schemas import AgentResponse

from dotenv import load_dotenv

load_dotenv()

tools = [TavilySearch()]
model = ChatOpenAI(model="gpt-4-turbo", temperature=0)

agent = create_agent(
    model=model,
    tools=tools,
    response_format=ToolStrategy(AgentResponse),
)


def main():
    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Search for 3 job listing for AI Engineers with Lanchain experience and provide the details.",
                }
            ]
        }
    )
    print(response["structured_response"])


if __name__ == "__main__":
    main()
