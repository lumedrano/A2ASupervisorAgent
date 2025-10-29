from typing import Any, Literal, Optional, List, TypedDict, Annotated
from pydantic import BaseModel, Field
import uuid
from operator import add
from a2a.types import AgentCard, Role
from a2a._base import A2ABaseModel

class Part(BaseModel):
    kind: str = Field(default="message", description="The kind of part, always 'text'.")
    text: str = Field(..., description="The text content of the message.")

class A2AMesssage(A2ABaseModel):
    context_id: Optional[str] = None
    extensions: Optional[List[str]] = None
    kind: Literal["message"] = "message"
    message_id: str = Field(default=lambda: uuid.uuid4().hex)
    metadata: Optional[dict[str, Any]] = None
    parts: List[Part] = Field(
        default_factory=lambda: [Part(text="text")],
        description="The message content parts.",
        examples=[[{"kind": "message", "text": "How are you doing today?"}]]
    )
    reference_task_ids: Optional[List[str]] = Field(default_factory=list)
    role: Role = Role.user
    task_id: Optional[str] = None

class Params(BaseModel):
    configuration: Optional[dict[str, Any]]
    message: A2AMesssage

class A2ARequest(BaseModel):
    id: Optional[str]
    jsonrpc: Optional[str]
    method: Optional[str]
    params: Params

class AgentCall(BaseModel):
    agent_id: str = Field(description="The 'id' of the agent's skill to call, taken from the available agents list.")
    query: str = Field(description="The specific query or text to send to that agent")

class Plan(BaseModel):
    thought: str = Field(description="Your reasoning and plan to answer the query. Analyze the available agents and previous results to formulate a step-by-step plan. If the query is answered, explain why.")
    next_action: Literal["call_agent", "finish"] = Field(description="If more steps are needed, choose 'call_agent'. If the query is fully answered, choose 'finish'.")
    agent_call: Optional[AgentCall] = Field(description="The specific agent call to make, if next_action is 'call_agent'.")

class SupervisorState(TypedDict):
    original_query: str
    available_agents: list[AgentCard]
    intermediate_results: Annotated[List[str], add]
    plan: Plan
    step_count: int
    final_answer: str
    retrieval_done: bool

class WorkflowState(dict):
    query: str
    reranked_docs: str
    summary: str
    task_id: str
    context_id: str