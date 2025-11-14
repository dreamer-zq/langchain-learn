from langchain_core.tools import tool
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import RunnableConfig
from langgraph.types import Command
from model import deepseek_model

import uuid

@tool
def delete_file(path: str) -> str:
    """Delete a file from the filesystem."""
    return f"Deleted {path}"


@tool
def read_file(path: str) -> str:
    """Read a file from the filesystem."""
    return f"Contents of {path}"

@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file on the filesystem."""
    return f"Written {content} to {path}"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    return f"Sent email to {to}"


def human_in_the_loop():
    """Run the agent with human-in-the-loop."""

    llm = deepseek_model(
        model="deepseek-reasoner",
    )

    # Checkpointer is REQUIRED for human-in-the-loop
    checkpointer = MemorySaver()

    agent = create_deep_agent(
        model=llm,
        tools=[delete_file, read_file, send_email],
        interrupt_on={
            "delete_file": {
                "allowed_decisions": ["approve", "reject"]
            },
            "read_file": False,  # No interrupts needed
            "write_file": False,  # No interrupts needed
            "send_email": {"allowed_decisions": ["approve", "reject"]},  # No editing
        },
        checkpointer=checkpointer,  # Required!
    )

    config: RunnableConfig = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Please write the content 'Hello, World!' to temp.txt, then read the content of temp.txt and send an email to admin@example.com, last, delete temp.txt",
                }
            ]
        },
        config=config,
    )

    if result.get("__interrupt__"):
        interrupts = result["__interrupt__"][0].value
        action_requests = interrupts["action_requests"]

        # Two tools need approval
        # assert len(action_requests) == 2
        # Provide decisions in the same order as action_requests
        decisions = [
            {"type": "approve"},  # First tool: send_email
            {"type": "reject"},  # Second tool: delete_file
        ]

        result = agent.invoke(Command(resume={"decisions": decisions}), config=config)

        print(result["messages"][-1].content)
