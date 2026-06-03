from langchain_mcp_adapters.client import MultiServerMCPClient
from config import get_llm
from langgraph.prebuilt import create_react_agent

async def create_foot_agent():
    client = MultiServerMCPClient({
        "foot": {
            "command": "c:/Git/district_reader/.venv/Scripts/python.exe",
            "args": ["c:/Git/district_reader/district_reader_mcp.py"],
            "transport": "stdio"
        }
    })
    tools = await client.get_tools()
    
    llm = get_llm()
    
    return create_react_agent(
        model=llm,
        tools=tools,
        prompt="Tu es un expert football de l'AS Canet. Utilise tes outils pour répondre précisément sur les résultats des matchs, les classements, les horaires des prochaines rencontres, et toute autre question liée au club et au football en général."
    )