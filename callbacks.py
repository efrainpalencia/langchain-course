from langchain_classic.callbacks.base import BaseCallbackHandler
from langchain_classic.schema import LLMResult

class AgentCallbackHandler(BaseCallbackHandler):
    def on_llm_start(
        self,
        serialized: dict,
        prompts: list[str],
        **kwargs,
    ) -> None:
        """Run when llm starts running"""
        print(f"***Prompt to LLM was***\n{prompts[0]}")
        print("********")

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Run whenn llm ends running"""
        print(f"***LLM response was***\n{response.generations[0][0].text}")
        print("********")