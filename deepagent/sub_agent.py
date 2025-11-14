import uuid
from tools import internet_search
from model import deepseek_model
from deepagents import create_deep_agent
from langgraph.graph.state import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient


async def example_multi_agent():
    """Fetch MCP-provided tools as LangChain-compatible tools.

    Returns a list of tools retrieved from configured MCP servers.
    """
    mcp_client = MultiServerMCPClient(
        {
            "docs-langchain": {
                "transport": "streamable_http",
                "url": "https://docs.langchain.com/mcp",
            }
        }
    )
    tools = await mcp_client.get_tools()

    """Create and run a multi-agent workflow.

    This function builds sub-agents for data collection, verification, and report writing,
    then invokes the deep agent to execute the workflow.
    """
    subagents = [
        {
            "name": "data-collector",
            "description": "Gathers raw data from the internet",
            "system_prompt": (
                "You are an expert researcher. Use `internet_search` to collect data. limit the search to 3 results."
                "Output only raw sources with URLs; do not summarize."
            ),
            "tools": [internet_search],
        },
        {
            "name": "data-checker",
            "description": "Responsible for proofreading the collected LangChain-related information and refining the proofread documents for users to read",
            "system_prompt": (
                "You are a LangChain expert. Strictly verify the collected content with `docs-langchain` tools. "
                "Reject unverifiable claims. Provide citations and mark any discrepancies."
            ),
            "tools": tools,
        },
        {
            "name": "report-writer",
            "description": "Writes professional reports from the proofread documents",
            "system_prompt": (
                "Write a professional Markdown report strictly based on verified content from data-checker. "
                "Include sources; do not introduce new information. Limit to 2000 words."
            ),
        },
    ]

    llm = deepseek_model(
        model="deepseek-reasoner",
    )

    agent = create_deep_agent(
        model=llm,
        system_prompt=(
            "Follow a strict three-step pipeline: (1) data-collector → (2) data-checker → (3) report-writer. "
            "You must run data-checker after data-collector and before report-writer. If verification fails, do not proceed to reporting."
        ),
        subagents=subagents,
    )

    config: RunnableConfig = {"configurable": {"thread_id": str(uuid.uuid4())}}
    # Stream the agent
    async for chunk in agent.astream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Collect the latest Deepagents-related technical articles from the internet (step 1). "
                        "Verify all information with docs-langchain (step 2). Then write a professional report (step 3). "
                        "Do NOT skip step 2."
                    ),
                }
            ]
        },
        config=config,
        stream_mode="values",
    ):
        if "messages" in chunk:
            chunk["messages"][-1].pretty_print()
