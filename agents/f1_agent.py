from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from config import get_llm


async def create_f1_agent():
    client = MultiServerMCPClient({
        "f1": {
            "command": "c:/Git/f1/server/.venv/Scripts/python.exe",
            "args": ["c:/Git/f1/server/f1_mcp.py"],
            "transport": "stdio"
        }
    })
    tools = await client.get_tools()
    
    llm = get_llm()
    
    return create_react_agent(
        model=llm,
        tools=tools,
        prompt="Tu es un expert Formule 1. Utilise tes outils pour répondre précisément sur les résultats de courses. tu pourras aussi donner les horaires des prochaines courses, les classements pilotes et constructeurs, et toute autre question liée à la F1."
    )