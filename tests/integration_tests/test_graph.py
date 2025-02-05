import os
from contextlib import contextmanager
from typing import Generator

import pytest
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStore
from langsmith import expect, unit

from index_graph import graph as index_graph
from retrieval_graph import graph
from shared.configuration import BaseConfiguration
from shared.retrieval import make_text_encoder


@contextmanager
def make_milvus_vectorstore(
    configuration: BaseConfiguration,
) -> Generator[VectorStore, None, None]:
    """Configure this agent to connect to a specific milvus index."""
     
    from langchain_milvus.vectorstores import Milvus as LangchainMilvus
    embedding_model = make_text_encoder(configuration.embedding_model)
     
    connection_args={
        "uri":os.environ.get("MILVUS_URL","http://localhost:19530"),
        #"token":os.environ.get("MILVUS_TOKEN",""),
        "user":os.environ.get("MILVUS_USER",""),
        "password":os.environ.get("MILVUS_PASSWORD",""),
    }
    vstore = LangchainMilvus(
            embedding_function=embedding_model,
            collection_name="langchain_docs",
            connection_args=connection_args, 
            auto_id = True
        )
    yield vstore


@pytest.mark.asyncio
@unit
async def test_retrieval_graph() -> None:
    simple_doc = 'In LangGraph, nodes are typically python functions (sync or async) where the first positional argument is the state, and (optionally), the second positional argument is a "config", containing optional configurable parameters (such as a thread_id).'
    config = RunnableConfig(
        configurable={
            "retriever_provider": "milvus",
            "embedding_model": "openai/text-embedding-3-small",
        }
    )
    configuration = BaseConfiguration.from_runnable_config(config)

    doc_id = "test_id"
    result = await index_graph.ainvoke(
        {"docs": [{"page_content": simple_doc, "id": doc_id}]}, config
    )
    expect(result["docs"]).against(lambda x: not x)  # we delete after the end
    # test general query
    res = await graph.ainvoke(
        {"messages": [("user", "Hi! How are you?")]},
        config,
    )
    expect(res["router"]["type"]).to_contain("general")

    # test query that needs more info
    res = await graph.ainvoke(
        {"messages": [("user", "I am having issues with the tools")]},
        config,
    )
    expect(res["router"]["type"]).to_contain("more-info")

    # test LangChain-related query
    res = await graph.ainvoke(
        {"messages": [("user", "What is a node in LangGraph?")]},
        config,
    )
    expect(res["router"]["type"]).to_contain("langchain")
    response = str(res["messages"][-1].content)
    expect(response.lower()).to_contain("function")

    # clean up after test
    with make_milvus_vectorstore(configuration) as vstore:
        await vstore.adelete([doc_id])
