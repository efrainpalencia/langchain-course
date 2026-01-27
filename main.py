import os
from langchain_openai import ChatOpenAI
from langchain_classic.agents.react.agent import create_react_agent
from langchain_tavily import TavilySearch
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_classic.agents import AgentExecutor
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langsmith import Client
from schemas import AgentResponse
from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS

from dotenv import load_dotenv

load_dotenv()

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
client = Client()
outputParser = PydanticOutputParser(pydantic_object=AgentResponse)
react_prompt = client.pull_prompt("hwchase17/react")
react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=[
        "tools",
        "tool_names",
        "format_instructions",
        "input",
        "agent_scratchpad",
    ],
).partial(format_instructions=outputParser.get_format_instructions())

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt_with_format_instructions,
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
extract_ouput = RunnableLambda(lambda x: x['output'])
parse_output = RunnableLambda(lambda x: outputParser.parse(x))
chain = agent_executor | extract_ouput | parse_output


def main():
    response = chain.invoke(
        input={
            "input": "Find 3 job listings for AI Engineers in Broward County, Florida and provide the details."
        }
    )
    print(response)


if __name__ == "__main__":
    main()
