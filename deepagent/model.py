from langchain_deepseek import ChatDeepSeek


def deepseek_model(model: str = "deepseek-chat"):
    return ChatDeepSeek(
        model=model,
        temperature=0.7,
        max_tokens=1024,
    )
