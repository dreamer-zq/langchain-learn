from deepagents import create_deep_agent
from model import deepseek_model
from tools import internet_search
from store import get_long_term_store, get_short_term_store
from langchain_core.runnables import RunnableConfig


# Initialize the DeepSeek chat model
llm = deepseek_model(
    model="deepseek-reasoner",
)


# System prompt to steer the agent behavior
system_prompt = "You are an expert researcher. Conduct thorough research and write a polished report."


def process_long_term_memory():
    """Exercise long-term memory persistence across threads using StoreBackend.

    Steps:
    1) In thread A, write to `/memories/longterm.txt` using filesystem tools.
    2) In thread A, read back `/memories/longterm.txt` to verify content.
    3) In thread B, read `/memories/longterm.txt` again to verify cross-thread persistence.
    """

    store, checkpointer, backend = get_long_term_store()
    agent = create_deep_agent(
        tools=[internet_search],
        system_prompt=system_prompt,
        model=llm,
        store=store,
        checkpointer=checkpointer,
        backend=backend,
    )

    config_a: RunnableConfig = {"configurable": {"thread_id": "persistent-thread-A"}}
    write_res = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Write the text 'long-term memo' to /memories/longterm.txt using filesystem tools.",
                }
            ]
        },
        config=config_a,
    )
    print("[A] write result:", write_res["messages"][ -1 ].content)

    read_res_a = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Read /memories/longterm.txt and return its content exactly.",
                }
            ]
        },
        config=config_a,
    )
    print("[A] read result:", read_res_a["messages"][ -1 ].content)

    config_b: RunnableConfig = {"configurable": {"thread_id": "persistent-thread-B"}}
    read_res_b = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Read /memories/longterm.txt and return its content exactly.",
                }
            ]
        },
        config=config_b,
    )
    print("[B] read result:", read_res_b["messages"][ -1 ].content)


def process_short_term_memory():
    """Exercise short-term (ephemeral) memory behavior using StateBackend.

    Steps:
    1) In thread A, write to a non-/memories/ path (e.g., /notes.txt) using filesystem tools.
    2) In thread A, read back /notes.txt to verify persistence across turns within the same thread.
    3) In thread B, attempt to read /notes.txt; expected to be missing since StateBackend is ephemeral.
    """

    # Create the deep agent with tools, prompt, and hybrid memory backend
    store, checkpointer, backend = get_short_term_store()
    agent = create_deep_agent(
        tools=[internet_search],
        system_prompt=system_prompt,
        model=llm,
        store=store,
        checkpointer=checkpointer,
        backend=backend,
    )

    # Thread A: write ephemeral content and read it back
    config_a: RunnableConfig = {"configurable": {"thread_id": "ephemeral-thread-A"}}
    write_res = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Write the text 'short-term memo' to /notes.txt using filesystem tools.",
                }
            ]
        },
        config=config_a,
    )
    print("[A] write result:", write_res["messages"][ -1 ].content)

    read_res_a = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Read /notes.txt and return its content exactly.",
                }
            ]
        },
        config=config_a,
    )
    print("[A] read result:", read_res_a["messages"][ -1 ].content)

    # Thread B: attempt to read the same path; should not exist for StateBackend
    config_b: RunnableConfig = {"configurable": {"thread_id": "ephemeral-thread-B"}}
    read_res_b = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Read /notes.txt and if not found say 'NOT_FOUND'.",
                }
            ]
        },
        config=config_b,
    )
    print("[B] read result:", read_res_b["messages"][ -1 ].content)
