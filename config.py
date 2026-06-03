import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from static import LLM_PROVIDER, LLM_MODEL

def get_llm():
    if LLM_PROVIDER == "ollama":
        return ChatOllama(model=LLM_MODEL, base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"))
    # elif LLM_PROVIDER == "openai":
    #     from langchain_openai import ChatOpenAI
    #     return ChatOpenAI(model=LLM_MODEL)
    # elif LLM_PROVIDER == "anthropic":
    #     from langchain_anthropic import ChatAnthropic
    #     return ChatAnthropic(model=LLM_MODEL)
    else:  # gemini par défaut
        return ChatGoogleGenerativeAI(model=LLM_MODEL)