from langchain_openai import ChatOpenAI
from app.core.config import get_settings

settings = get_settings()

def create_chat_llm(model_name: str, max_tokens: int, temperature: float = 0):
    return ChatOpenAI(
        model=model_name,
        api_key=settings.api_key,
        base_url=settings.api_base,
        temperature=temperature,
        max_tokens=max_tokens,
    )

sentiment_chat_llm = create_chat_llm("vsf-lora", 16)
medqa_chat_llm = create_chat_llm("medqa-lora", 64)  