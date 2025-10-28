#supervisor.py
#default imports 
import httpx
from dotenv import load_dotenv 
from uuid import uuid4
import json
from typing import List 
import time


#langchain/graph imports
from langchain_core.prompts import ChatPromptTemplate 
from langgraph.graph import StateGraph, END, START


#a2a imports
from a2a.server.agent_execution import AgentExecutor, RequestContext 
from a2a.server.events import EventQueue 
from a2a.types import (
AgentCard, AgentCapabilities, AgentSkill, Message, Role, TextPart)

#custom imports
# from llm.dsx_11m.new import client_fc_mistral #Change this
from utils.protocol_wrappers import (extract_text, send_text_async_url) 
from utils.config import SUPERVISOR_PORT, MAX_STEPS, AGENT_REGISTRY_URL 
from utils.server import run_agent_blocking 
from models.models import Plan, SupervisorState 
from utils.constants import SYSTEM_PROMPT_SUPERVISOR
from utils.logging_utils import Colors, get_logger, configure_logging 

configure_logging()
logger = get_logger(__name__)
load_dotenv()


skills = AgentSkill(
    id="dynamic_planner_orchestrator",
    name="Dynamic Planner & Orchestrator",
    description="A master orchestrator agent that discovers and coordinates specialist agents",
    tags=["supervisor", "orchestrator", "graph", "planning"]
)

card = AgentCard(
    name="Supervisor",
    description=skills.description,
    url=f"http://localhost:8000/agent/message",
    version="0.0.1",
    protocol_version="0.2.5",
    capabilities=AgentCapabilities(streaming=False),
    skills=[skills],
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain"]
)

#TODO: define LLM client here (POSSIBLY LLAMA)


def get_text_from_a2a_message(message: Message | None) -> str:
    """
    Extracts the text content from an A2A Message Object
    
    """
    if not message or not message.parts:
        return ""
    for part_wrapper in message.parts:
        actual_part = getattr(part_wrapper, "root", part_wrapper)
        if isinstance(actual_part, TextPart):
            return actual_part.text
    return ""


async def call_sub_agent(url: str, query: str, docs: list[str] | None = None) -> str:
    try:
        if docs is not None:
            payload = json.dumps({"query": query, "docs": docs})
            resp = await send_text_async_url(url, text=payload)
        else:
            resp = await send_text_async_url(url, text=query)
        return extract_text(resp)
    except Exception as e:
        return f"ERROR: Failed to connect to the agent at {url}."
    
async def discover_agents_node(state: SupervisorState) -> SupervisorState:
    """
    Discovers available sub-agents by querying the agent registry.
    Args:
        state (SupervisorState): Current supervisor workflow state.
    Returns:
        SupervisorState: Updated state with a list of discovered "AgentCard" objects.      
    """

    logger. info(f" {Colors. HEADER}--- Step 1: Discovering Agents from Registry ---{Colors.ENDC}") 
    start_time_dicovery = time.time()
    cards: List[AgentCard] = []


    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{AGENT_REGISTRY_URL}/agents")
            resp.raise_for_status()
            data = resp.json()
            for raw_card in data.get("agents", []):
                try:
                    card - AgentCard.model_validate(raw_card)
                    cards.append(card)
                except Exception as e:
                    print (f"Invalid card skipped: {e}")
        except Exception as e:
            logger.info(f"{Colors.FAIL} ERROR: Could not fetch agents from registry: (e){Colors. ENDC}")

    state["available agents"] = cards
    logger.info(f" {Colors.OKGREEN} OBTAINED AGENT CARDS {Colors.ENDC} ") 
    end_time_dicovery = time.time()
    logger.info(f"{Colors.OKCYAN}[TIME FOR AGENT DISCOVERY]: {end_time_dicovery - start_time_dicovery}{Colors.ENDC}")
    return state


