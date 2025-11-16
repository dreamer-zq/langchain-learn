from __future__ import annotations

import json
from typing import Dict

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import Tool
from langchain.agents.middleware.types import wrap_model_call, ModelResponse
from deepagents import create_deep_agent
from model import deepseek_model
from langchain_core.runnables import RunnableConfig, ensure_config


MOCK_USERS: Dict[str, Dict[str, str]] = {
    "user1": {"password": "pass1", "role": "user"},
    "user2": {"password": "pass2", "role": "user"},
    "admin": {"password": "admin", "role": "admin"},
}


def _verify_credentials(username: str, password: str) -> dict | None:
    """Verify credentials against mocked user store.

    Returns a payload with username and role on success, otherwise None.
    """
    record = MOCK_USERS.get(username)
    if not record or record.get("password") != password:
        return None
    return {"user": username, "role": record["role"]}


def _login_func(username: str, password: str) -> str:
    """Authenticate user and return a JSON payload with status and role.

    This function uses mocked credentials and returns a JSON string:
    {"status": "ok", "user": "...", "role": "..."} or
    {"status": "error", "message": "..."}.
    """
    payload = _verify_credentials(username, password)
    if not payload:
        return json.dumps({"status": "error", "message": "Invalid credentials"})
    return json.dumps({"status": "ok", **payload})


login_tool = Tool(
    name="login",
    description="Authenticate with username/password (mock). Returns JSON status.",
    func=_login_func,
)


def _augment_ai_message(ai: AIMessage, prefix: str) -> AIMessage:
    """Prepend user info prefix to AIMessage content, handling str or list content."""
    content = ai.content
    if isinstance(content, str):
        return AIMessage(content=f"{prefix}{content}")
    if isinstance(content, list):
        return AIMessage(content=[prefix] + content)
    return AIMessage(content=f"{prefix}{str(content)}")


@wrap_model_call(name="LoginRequiredMiddleware")
def login_required_middleware(request, handler):
    """Block model execution if user is not authenticated.

    Reads authentication from runtime context keys: `auth_user` and `auth_role`.
    If absent, short-circuits with an AIMessage prompting for login.
    """
    
    ctx = request.runtime.context or {}
    conf = ensure_config()
    configurable = conf.get("configurable", {}) if isinstance(conf, dict) else {}
    user = ctx.get("auth_user") or configurable.get("auth_user")
    password = ctx.get("auth_pass") or configurable.get("auth_pass")

    if user and password:
        payload = _verify_credentials(user, password)
        if not payload:
            return AIMessage(content="Authentication failed. Please provide valid credentials via config.configurable.")
        ctx["auth_role"] = payload.get("role")
        response = handler(request)
        try:
            if isinstance(response, ModelResponse):
                msgs = list(response.result)
                if msgs and isinstance(msgs[-1], AIMessage):
                    msgs[-1] = _augment_ai_message(
                        msgs[-1], f"[user:{user}, role:{payload.get('role')}] "
                    )
                return ModelResponse(result=msgs, structured_response=response.structured_response)
            if isinstance(response, AIMessage):
                return _augment_ai_message(
                    response, f"[user:{user}, role:{payload.get('role')}] "
                )
        except Exception:
            pass
        return response

    return AIMessage(content="Authentication required. Provide `auth_user` and `auth_pass` via config.configurable or call the `login` tool.")


def build_demo_agent():
    """Build a deep agent configured with login tool and middlewares.

    Returns a compiled agent graph ready for invocation.
    """
    llm = deepseek_model(model="deepseek-chat")
    agent = create_deep_agent(
        model=llm,
        tools=[login_tool],
        middleware=[login_required_middleware],
        system_prompt="You are a helpful assistant that requires authentication before responding.",
    )
    return agent


def demo_login_example():
    """Demonstrate login gating: before login blocked, after login allowed.

    1) Invoke agent without authentication and capture the auth-required message.
    2) Call `login` tool with mocked credentials and attach auth to agent context.
    3) Invoke agent again to show normal operation after login.
    """
    agent = build_demo_agent()

    pre = agent.invoke({
        "messages": [{"role": "user", "content": "Say hello"}]
    })
    print(pre["messages"][-1].content)

    config: RunnableConfig = {"configurable": {"auth_user": "user1", "auth_pass": "pass1"}}
    post = agent.invoke(
        {"messages": [{"role": "user", "content": "Hello world"}]}, 
        config=config
    )
    print(post["messages"][-1].content)
