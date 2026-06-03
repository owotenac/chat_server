from typing import Literal
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from .f1_agent import create_f1_agent
from .foot_agent import create_foot_agent
from dotenv import load_dotenv
from config import get_llm

load_dotenv()

llm = get_llm()

# State custom qui étend MessagesState avec le champ next
class AgentState(TypedDict):
    messages: list[BaseMessage]
    next: str

def make_supervisor_node(members: list[str]):
    system_prompt = f"""Tu es un superviseur qui route les questions vers le bon agent.
    Agents disponibles: {members}
    - f1_agent : questions sur la Formule 1, résultats, courses, pilotes
    - foot_agent : questions sur le football, matchs, résultats
    - general : toute autre question hors sport, ou si tu n'es pas sûr
    Réponds UNIQUEMENT avec exactement l'un de ces mots: {', '.join(members + ['general'])}"""

    def supervisor(state: AgentState):
        last_message = state["messages"][-1].content
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": last_message}
        ])
        next_agent = ""
        if isinstance(response.content, list):
            next_agent = response.content[0].get("text", "").strip()
        else:
            next_agent = response.content.strip()

        # Cas 1 : le LLM a répondu un nom valide
        if next_agent in members:
            print(f"✅ Routing to: {next_agent}")
            return {"messages": state["messages"], "next": next_agent}

        # Cas 2 : hors scope explicite → répondre directement sans agent
        if next_agent == "general":
            print("💬 General question, answering directly")
            return {"messages": state["messages"], "next": "general"}
                
        print(f"🔀 Supervisor decision: '{next_agent}'")
        # Sécurité : si le LLM répond n'importe quoi, on fallback
        # Cas 3 : réponse inattendue du LLM → log + general
        print(f"⚠️ Unexpected response '{next_agent}', fallback to general")
        return {"messages": state["messages"], "next": "general"}

    return supervisor


    return supervisor

async def create_graph():
    f1_agent = await create_f1_agent()
    foot_agent = await create_foot_agent()

    async def f1_node(state: AgentState):
        result = await f1_agent.ainvoke({"messages": state["messages"]})
        return {"messages": result["messages"], "next": state["next"]}

    async def foot_node(state: AgentState):
        result = await foot_agent.ainvoke({"messages": state["messages"]})
        return {"messages": result["messages"], "next": state["next"]}

    async def general_node(state: AgentState):
        response = llm.invoke(state["messages"])
        from langchain_core.messages import AIMessage
        return {
            "messages": state["messages"] + [AIMessage(content=response.content)],
            "next": state["next"]
        }

    supervisor = make_supervisor_node(["f1_agent", "foot_agent"])

    def route(state: AgentState) -> Literal["f1_agent", "foot_agent", 'general']:
        print(f"🗺️ route() called with next='{state['next']}'")
        return state["next"]

    builder = StateGraph(AgentState)
    builder.add_node("supervisor", supervisor)
    builder.add_node("f1_agent", f1_node)
    builder.add_node("foot_agent", foot_node)
    builder.add_node("general", general_node)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges("supervisor", route)
    builder.add_edge("f1_agent", END)
    builder.add_edge("foot_agent", END)
    builder.add_edge("general", END)

    return builder.compile()