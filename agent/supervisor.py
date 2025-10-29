#supervisor.py
#default imports 
import httpx
from dotenv import load_dotenv 
from uuid import uuid4
import json
from typing import List 
import time
import traceback


#langchain/graph imports
from langchain_core.prompts import ChatPromptTemplate 
from langgraph.graph import StateGraph, END, START


#a2a imports
from a2a.server.agent_execution import AgentExecutor, RequestContext 
from a2a.server.events import EventQueue 
from a2a.types import (
AgentCard, AgentCapabilities, AgentSkill, Message, Role, TextPart)

#custom imports
from llm.llm import client_llama
from utils.protocol_wrappers import (extract_text, send_text_async_url) 
from utils.config import SUPERVISOR_PORT, MAX_STEPS, SUBAGENT_URLS
from utils.server import run_agent_blocking 
from models.models import Plan, SupervisorState 
from utils.constants import SYSTEM_PROMPT_SUPERVISOR
from utils.logging_utils import Colors, get_logger, configure_logging 

configure_logging()
logger = get_logger(__name__)
load_dotenv()


#TODO: work on why requests are not waiting for a response from the subagents and automatically going to planning again


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

llm = client_llama

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
        resp = await send_text_async_url(url, text=query)
        return extract_text(resp)
    except Exception as e:
        return f"ERROR: Failed to connect to the agent at {url}."
    
# async def discover_agents_node(state: SupervisorState) -> SupervisorState:
#     """
#     Discovers available sub-agents by querying the agent registry.
#     Args:
#         state (SupervisorState): Current supervisor workflow state.
#     Returns:
#         SupervisorState: Updated state with a list of discovered "AgentCard" objects.      
#     """

#     logger. info(f" {Colors. HEADER}--- Step 1: Discovering Agents from Registry ---{Colors.ENDC}") 
#     start_time_dicovery = time.time()
#     cards: List[AgentCard] = []


#     async with httpx.AsyncClient() as client:
#         try:
#             resp = await client.get(f"{AGENT_REGISTRY_URL}/agents")
#             resp.raise_for_status()
#             data = resp.json()
#             for raw_card in data.get("agents", []):
#                 try:
#                     card - AgentCard.model_validate(raw_card)
#                     cards.append(card)
#                 except Exception as e:
#                     print (f"Invalid card skipped: {e}")
#         except Exception as e:
#             logger.info(f"{Colors.FAIL} ERROR: Could not fetch agents from registry: (e){Colors. ENDC}")

#     state["available agents"] = cards
#     logger.info(f" {Colors.OKGREEN} OBTAINED AGENT CARDS {Colors.ENDC} ") 
#     end_time_dicovery = time.time()
#     logger.info(f"{Colors.OKCYAN}[TIME FOR AGENT DISCOVERY]: {end_time_dicovery - start_time_dicovery}{Colors.ENDC}")
#     return state


async def discover_agents_node(state: SupervisorState) -> SupervisorState:
    """
    Discovers available sub-agents by fetching their agent cards from 
    well-known paths instead of a central registry.
    """
    logger.info(f"{Colors.HEADER}--- Step 1: Discovering Agents from .well-known paths ---{Colors.ENDC}")
    start_time_discovery = time.time()
    cards: List[AgentCard] = []

    async with httpx.AsyncClient() as client:
        for url in SUBAGENT_URLS:
            try:
                base_url = url.rstrip("/")
                resp = await client.get(f"{base_url}/.well-known/agent-card.json")
                resp.raise_for_status()
                data = resp.json()
                card = AgentCard.model_validate(data)
                cards.append(card)
            except Exception as e:
                logger.info(f"{Colors.FAIL}ERROR: Could not fetch agent cards from {base_url}{Colors.ENDC}")

    state["available_agents"] = cards
    logger.info(f"{Colors.OKCYAN}Discovered {len(cards)} agent(s).{Colors.ENDC}")
    end_time_discovery = time.time()
    logger.info(f"{Colors.OKBLUE}[TIME FOR DISCOVERY]: {end_time_discovery - start_time_discovery:.2f}s{Colors.ENDC}")
    return state


async def planner_node(state: SupervisorState) -> SupervisorState:
    logger.info(f"{Colors.HEADER}\n --- Step {state['step_count'] + 2}: Planning ---{Colors.ENDC}")
    agent_descriptions = "\n".join(
        f"- Agent Skill ID: '{skill.id}', Description: {skill.description}"
        for card in state['available_agents'] for skill in card.skills
    )

    system_prompt = SYSTEM_PROMPT_SUPERVISOR

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Available Agents:\n{agents}\n\nPrevious Results:\n{results}\n\nUser Query: {query}")
    ])

    planner_runnable = prompt | llm.with_structured_output(Plan)

    plan: Plan = await planner_runnable.ainvoke({
        "agents": agent_descriptions,
        "results": "\n".join(state["intermediate_results"]) or "None",
        "query": state["original_query"],
    })
    
    state["plan"] = plan
    print(f"Thought: {plan.thought}")
    return state

async def execute_agent_call_node(state: SupervisorState) -> SupervisorState:
    plan = state["plan"]
    if not plan.agent_call:
        state["intermediate_results"] = ["Planner decided to call an agent but provided no details"]
        return state
    agent_call = plan.agent_call
    logger.info(f"{Colors.HEADER}--- Step {state['step_count'] + 2}: Executing Call to '{agent_call.agent_id}' ---{Colors.ENDC}")

    agent_url = None
    for card in state["available_agents"]:
        for skill in card.skills:
            if skill.id == agent_call.agent_id:
                agent_url = card.url
                break
            if agent_url:
                break
    if not agent_url:
        result = f"{Colors.FAIL}ERROR: Could not find an agent with skill ID '{agent_call.agent_id}'.{Colors.ENDC}"
    else:
        result = await call_sub_agent(agent_url, agent_call.query)
    state["intermediate_results"] = [result]
    logger.info(state["intermediate_results"])

    
    state["step_count"] += 1
    return state

async def final_answer_node(state: SupervisorState) -> SupervisorState:
    logger.info(f"{Colors.HEADER}--- Final Step: Synthesizing Answer ---{Colors.ENDC}")
    synthesizer_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Synthesize a final, clean answer for the user based on their original query and the collected results from specialist agents. Response directly to the user."),
        ("human", "Original Query: {query}\n\nCollected Information:\n{results}")
    ])

    synthesizer_runnable = synthesizer_prompt | llm

    final_answer = await synthesizer_runnable.ainvoke({
        "query": state["original_query"],
        "results": "\n".join(state["intermediate_results"])
    })

    state["final_answer"] = final_answer.content
    return state
    
def should_continue_edge(state: SupervisorState) -> str:
    if state["step_count"] >= MAX_STEPS:
        return "error_node"
    if state["plan"].next_action == "finish":
        return "final_answer_node"
    if state["plan"].next_action == "call_agent":
        return "execute_agent_call_node"
    return "error_node"

def error_node(state: SupervisorState) -> SupervisorState:
    state["final_answer"] = "I'm sorry, I encountered an error and could not complete your request."
    return state

workflow = StateGraph(SupervisorState)
workflow.add_node("discover_agents_node", discover_agents_node)
workflow.add_node("planner_node", planner_node)
workflow.add_node("execute_agent_call_node", execute_agent_call_node)
workflow.add_node("final_answer_node", final_answer_node)
workflow.add_node("error_node", error_node)

workflow.set_entry_point("discover_agents_node")
workflow.add_edge("discover_agents_node", "planner_node")
workflow.add_conditional_edges("planner_node", should_continue_edge)
workflow.add_edge("execute_agent_call_node", "planner_node")
workflow.add_edge("final_answer_node", END)
workflow.add_edge("error_node", END)

supervisor_agent_graph = workflow.compile()

class SupervisorExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        try:
            if not context.message: raise ValueError("Executor received no message.")
            query = get_text_from_a2a_message(context.message)
            if not query: raise ValueError("No user query")

            logger.info(f"{Colors.HEADER}--- Supervisor Graph starting with query: '{query}'---{Colors.ENDC}")
            initial_state = {"original_query": query, "intermediate_results": [], "step_count":0, "retrieval_done": False}

            final_state = await supervisor_agent_graph.ainvoke(initial_state, {"recursion_limit": 15})

            response_text = final_state.get("final_answer", "")
            logger.info(f"{Colors.HEADER}---Supervisor Graph finished with response: '{response_text}'---{Colors.ENDC}")
            final_message = Message(message_id=f"response-msg-{uuid4().hex}", role=Role.agent, parts=[TextPart(text=response_text)])
            print(response_text)
            await event_queue.enqueue_event(final_message)
        except Exception as e:
            logger.info(f"{Colors.FAIL}Error in Supervisor execution: {e}{Colors.ENDC}")
            traceback.print_exc()
            error_message = Message(message_id=f"error-msg-{uuid4().hex}", role=Role.agent, parts=[TextPart(text=f"An error occured: {e}")])
            await event_queue.enqueue_event(error_message)
        finally:
            await event_queue.close()

    async def cancel(self, context, event_queue):
        pass

if __name__ == "__main__":
    run_agent_blocking(
        name="SupervisorAgent",
        port=SUPERVISOR_PORT,
        agent_card = card,
        executor=SupervisorExecutor()
    )