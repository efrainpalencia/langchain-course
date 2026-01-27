from typing import List
from pydantic import BaseModel, Field


class Source(BaseModel):
    """Schema for agents to return source information."""
    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent responses."""
    answer: str = Field(description="The answer to the user's query")
    sources: List[Source] = Field(
        default_factory=list,
        description="A list of sources used to generate the answer"
    )
