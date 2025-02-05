# RAG Research Agent with NVIDIA NIM Deepseek-R1

This is a starter project to help you get started with developing a RAG research agent using Langchain and NIVDIA NIM Deepseek-R1.

![Graph view in LangGraph studio UI](./static/studio_ui.png)

## Streamlit App
```bash
streamlit run src/app.py
```
![Streamlit UI](./static/streamlit-run1.png)
![Streamlit UI](./static/streamlit-run2.png)
![Streamlit UI](./static/streamlit-run3.png)
## Graphs

This project has three graphs:

* an "index" graph (`src/index_graph/graph.py`)
* a "retrieval" graph (`src/retrieval_graph/graph.py`)
* a "researcher" subgraph (part of the retrieval graph) (`src/retrieval_graph/researcher_graph/graph.py`)

The index graph takes in document objects indexes them.

```json
[{ "page_content": "LangGraph is a library for building stateful, multi-actor applications with LLMs, used to create agent and multi-agent workflows." }]
```

If an empty list is provided (default), a list of sample documents from `src/sample_docs.json` is indexed instead. Those sample documents are based on the conceptual guides for LangChain and LangGraph.

The retrieval graph manages a chat history and responds based on the fetched documents. Specifically, it:

1. Takes a user **query** as input
2. Analyzes the query and determines how to route it:
- if the query is about "LangChain", it creates a research plan based on the user's query and passes the plan to the researcher subgraph
- if the query is ambiguous, it asks for more information
- if the query is general (unrelated to LangChain), it lets the user know
3. If the query is about "LangChain", the researcher subgraph runs for each step in the research plan, until no more steps are left:
- it first generates a list of queries based on the step
- it then retrieves the relevant documents in parallel for all queries and return the documents to the retrieval graph
4. Finally, the retrieval graph generates a response based on the retrieved documents and the conversation context

## Getting Started

Assuming you have already [installed LangGraph Studio](https://github.com/langchain-ai/langgraph-studio?tab=readme-ov-file#download), to set up:

1. Create a `.env` file.

```bash
cp .env.example .env
```

2. Select your retriever & index, and save the access instructions to your `.env` file.
 
3. Set up your environment:
- In the Atlas dashboard, click on "Connect" for your cluster.
- Choose "Connect your application" and copy the provided connection string.
- Create a `.env` file in your project root if you haven't already.
- Add your MongoDB Atlas connection string to the `.env` file:

```
MILVUS_URL=...
MILVUS_USER=...
MILVUS_PASSWORD=...
```

You can access milvus cloud to start up free instance [here](https://milvus.io/)
 
### Setup Model

The defaults values for `response_model`, `query_model` are shown below:

```yaml
response_model: nvidia/meta/llama-3.3-70b-instruct
query_model: nvidia/meta/llama-3.1-8b-instruct
```

Follow the instructions below to get set up, or pick one of the additional options.

#### NVIDIA NIM

To use NVIDIA NIM's models:

1. Sign up for an [NVIDIA API key](https://build.nvidia.com/) if you haven't already.
2. Once you have your API key, add it to your `.env` file:

```
NVIDIA_API_KEY=your-api-key
```
#### OpenAI

To use OpenAI's chat models:

1. Sign up for an [OpenAI API key](https://platform.openai.com/signup).
2. Once you have your API key, add it to your `.env` file:
```
OPENAI_API_KEY=your-api-key
```



### Setup Embedding Model

The defaults values for `embedding_model` are shown below:

```yaml
embedding_model: nvidia/nvidia/nv-embedqa-e5-v5
```

Follow the instructions below to get set up, or pick one of the additional options.

#### OpenAI

To use OpenAI's embeddings:

1. Sign up for an [OpenAI API key](https://platform.openai.com/signup).
2. Once you have your API key, add it to your `.env` file:
```
OPENAI_API_KEY=your-api-key
```

<!--
End setup instructions
-->

## Using

Once you've set up your retriever and saved your model secrets, it's time to try it out! First, let's add some information to the index. Open studio, select the "indexer" graph from the dropdown in the top-left, and then add some content to chat over. You can just invoke it with an empty list (default) to index sample documents from LangChain and LangGraph documentation.

You'll know that the indexing is complete when the indexer "delete"'s the content from its graph memory (since it's been persisted in your configured storage provider).

Next, open the "retrieval_graph" using the dropdown in the top-left. Ask it questions about LangChain to confirm it can fetch the required information!

## How to customize

You can customize this retrieval agent template in several ways:

1. **Change the retriever**: You can switch between different vector stores (Elasticsearch, MongoDB, Pinecone) by modifying the `retriever_provider` in the configuration. Each provider has its own setup instructions in the "Getting Started" section above.

2. **Modify the embedding model**: You can change the embedding model used for document indexing and query embedding by updating the `embedding_model` in the configuration. Options include various OpenAI and Cohere models.

3. **Adjust search parameters**: Fine-tune the retrieval process by modifying the `search_kwargs` in the configuration. This allows you to control aspects like the number of documents retrieved or similarity thresholds.

4. **Customize the response generation**: You can modify the `response_system_prompt` to change how the agent formulates its responses. This allows you to adjust the agent's personality or add specific instructions for answer generation.

5. **Modify prompts**: Update the prompts used for user query routing, research planning, query generation and more in `src/retrieval_graph/prompts.py` to better suit your specific use case or to improve the agent's performance. You can also modify these directly in LangGraph Studio. For example, you can:

  * Modify system prompt for creating research plan (`research_plan_system_prompt`)
  * Modify system prompt for generating search queries based on the research plan (`generate_queries_system_prompt`)

6. **Change the language model**: Update the `response_model` in the configuration to use different language models for response generation. Options include various Claude models from Anthropic, as well as models from other providers like Fireworks AI.

7. **Extend the graph**: You can add new nodes or modify existing ones in the `src/retrieval_graph/graph.py` file to introduce additional processing steps or decision points in the agent's workflow.

8. **Add tools**: Implement tools to expand the researcher agent's capabilities beyond simple retrieval generation.

Remember to test your changes thoroughly to ensure they improve the agent's performance for your specific use case.

## Development

While iterating on your graph, you can edit past state and rerun your app from past states to debug specific nodes. Local changes will be automatically applied via hot reload. Try adding an interrupt before the agent calls the researcher subgraph, updating the default system message in `src/retrieval_graph/prompts.py` to take on a persona, or adding additional nodes and edges!

Follow up requests will be appended to the same thread. You can create an entirely new thread, clearing previous history, using the `+` button in the top right.

You can find the latest (under construction) docs on [LangGraph](https://github.com/langchain-ai/langgraph) here, including examples and other references. Using those guides can help you pick the right patterns to adapt here for your use case.

LangGraph Studio also integrates with [LangSmith](https://smith.langchain.com/) for more in-depth tracing and collaboration with teammates.

 