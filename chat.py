# import os
# from pydantic import BaseModel
# from google import genai
# from fastapi import HTTPException
# #list of session
# active_sessions = {}

# client = genai.Client()

# class ChatMessage(BaseModel):
#     session_id: str
#     message: str

# def chat(message: ChatMessage):
#     try:
#         if message.session_id not in active_sessions:
#             print(f"Création d'une nouvelle session de chat : {message.session_id}")
#             active_sessions[message.session_id] = client.chats.create(
#                 model="gemini-2.5-flash"
#             )        
        
#         current_session = active_sessions[message.session_id]
#         response = current_session.send_message(message.message)
        
#         return {"message": response.text}
#     except Exception as e:
#         return HTTPException(status_code=500, detail=str(e))


from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

graph = None
sessions: dict[str, list] = {}

class ChatMessage(BaseModel):
    session_id: str
    message: str

async def chat(message: ChatMessage):
    if graph is None:
        raise HTTPException(status_code=503, detail="Graph not initialized")

    if not message.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    if message.session_id not in sessions:
        sessions[message.session_id] = []

    sessions[message.session_id].append(
        HumanMessage(content=message.message)
    )

    try:
        result = await graph.ainvoke({
            "messages": sessions[message.session_id]
        })
    except Exception as e:
        # On retire le message qu'on vient d'ajouter pour ne pas corrompre l'historique
        sessions[message.session_id].pop()
        logger.error(f"Graph invocation failed for session {message.session_id}: {e}")
        raise HTTPException(status_code=500, detail="Agent error")

    sessions[message.session_id] = result["messages"]

    last = result["messages"][-1]

    # LangGraph renvoie le contenu directement comme string sur AIMessage
    if not isinstance(last, AIMessage) or not last.content:
        logger.warning(f"Unexpected last message type: {type(last)}")
        raise HTTPException(status_code=500, detail="No response from agent")

    #print(f"Session {message.session_id} - User: {message.message} | Agent: {last.content}")
    return {
        "message": [{"type": "text", "text": last.content}]
    }