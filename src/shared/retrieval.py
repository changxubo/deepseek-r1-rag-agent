"""Manage the configuration of various retrievers.

This module provides functionality to create and manage retrievers for different
vector store backends, specifically Elasticsearch, Pinecone, and MongoDB.
"""

import os
from contextlib import contextmanager
from typing import Generator

from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStoreRetriever

from shared.configuration import BaseConfiguration

## Encoder constructors


def make_text_encoder(model: str) -> Embeddings:
    """Connect to the configured text encoder."""
    if model is None:
        model = "nvidia/nvidia/nv-embedqa-e5-v5"
    provider, model = model.split("/", maxsplit=1)
    match provider:
        case "openai":
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(model=model)
        case "nvidia":
            from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

            return NVIDIAEmbeddings(model=model )
        case _:
            raise ValueError(f"Unsupported embedding provider: {provider}")


## Retriever constructors

@contextmanager
def make_milvus_retriever(
    configuration: BaseConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific pinecone index."""
    from langchain_milvus.vectorstores import Milvus as LangchainMilvus

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
    search_kwargs = configuration.search_kwargs
    if search_kwargs is None:
        search_kwargs= {}
    yield vstore.as_retriever(search_kwargs=search_kwargs)


@contextmanager
def make_retriever(
    config: RunnableConfig,
) -> Generator[VectorStoreRetriever, None, None]:
    """Create a retriever for the agent, based on the current configuration."""
    configuration = BaseConfiguration.from_runnable_config(config)
    embedding_model = make_text_encoder(configuration.embedding_model)
    provider = configuration.retriever_provider
    if provider is None:
        provider = "milvus"
    match provider:
        case "milvus":
            with make_milvus_retriever(configuration, embedding_model) as retriever:
                yield retriever
        case _:
            raise ValueError(
                "Unrecognized retriever_provider in configuration. "
                f"Expected one of: {', '.join(BaseConfiguration.__annotations__['retriever_provider'].__args__)}\n"
                f"Got: {provider}"
            )
