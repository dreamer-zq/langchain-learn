import os
import atexit
import contextlib

from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import PostgresStore
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend

def get_long_term_store():
    """Return (store, checkpointer, backend) for long-term memory.

    Requires `DATABASE_URL` to be set; otherwise raises an exception.
    Initializes a Postgres-backed Store and Checkpointer, registers cleanup on exit,
    runs setup to ensure required tables exist, and configures a Composite backend
    that routes `/memories/` to the persistent store while keeping other paths ephemeral.
    """
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is required for long-term store")

    try:
        store_cm = PostgresStore.from_conn_string(db_url, ttl={"default_ttl": 60 * 5})
        check_cm = PostgresSaver.from_conn_string(db_url)
        store = store_cm.__enter__()
        checkpointer = check_cm.__enter__()
        atexit.register(
            lambda: contextlib.suppress(Exception)
            or store_cm.__exit__(None, None, None)
        )
        atexit.register(
            lambda: contextlib.suppress(Exception)
            or check_cm.__exit__(None, None, None)
        )
        try:
            store.setup()
        except Exception:
            pass
        try:
            checkpointer.setup()
        except Exception:
            pass
        backend = lambda rt: CompositeBackend(
            default=StateBackend(rt),
            routes={"/memories/": StoreBackend(rt)},
        )
        return store, checkpointer, backend
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Postgres-backed long-term store: {e}")


def get_short_term_store():
    """Return (store, checkpointer, backend) for short-term (ephemeral) memory.

    Uses `InMemoryStore` and `InMemorySaver` for non-persistent usage. Configures a
    Composite backend that keeps paths ephemeral; `/memories/` is still routed via
    `StoreBackend(rt)` but backed by the provided in-memory store.
    """
    store = InMemoryStore()
    checkpointer = InMemorySaver()
    backend = lambda rt: CompositeBackend(
        default=StateBackend(rt),
        routes={"/memories/": StoreBackend(rt)},
    )
    return store, checkpointer, backend
