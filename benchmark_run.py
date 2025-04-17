from typing import TypedDict, List
from pydantic.v1 import BaseModel

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langgraph.graph import StateGraph

from langchain_text_splitters import RecursiveCharacterTextSplitter
from semantic_chunker import SemanticChunker

# Define the graph state schema
class GraphState(TypedDict):
    question: str
    retrieved_docs: List[Document]
    answer: str

# Embeddings and chunkers
embedding_model = OllamaEmbeddings(model="nomic-embed-text")
semantic_chunker = SemanticChunker()

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=512,
    chunk_overlap=50,
)

# RAG nodes
def make_retrieve_node(retriever):
    def retrieve_node(state: GraphState) -> GraphState:
        question = state["question"]
        docs = retriever.invoke(question)
        return {**state, "retrieved_docs": docs}
    return retrieve_node

def generate_node(state: GraphState) -> GraphState:
    question = state["question"]
    docs = state["retrieved_docs"]
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    You are a particle physicist.

    Answer the question using only the context provided. Output your response as a JSON object with this format:

    {{
      "answer": "<your numeric or symbolic answer here>",
      "reasoning": "<your short chain-of-thought reasoning>"
    }}

    - Only include the JSON object, no extra text.
    - The "answer" field must be a number or a string like "N/A".
    - The "reasoning" field must be a one-line explanation.

    Context:
    {context}

    Question:
    {state['question']}

    Answer JSON:
    """
    llm = ChatOllama(model="mistral", temperature=0.0)
    response = llm.invoke(prompt)

    return {
        **state,
        "answer": response.content.strip()
    }

# LangGraph wiring
def build_graph(retriever):
    builder = StateGraph(GraphState)  # <- Legacy compatible constructor
    builder.add_node("retrieve", make_retrieve_node(retriever))
    builder.add_node("generate", generate_node)
    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "generate")
    builder.set_finish_point("generate")
    return builder.compile()

# Vectorstore builder with semantic chunking
def build_vectorstore(contexts: List[str]):
    documents = []
    for context in contexts:
        chunks = semantic_chunker.split_by_tokens(context) if len(context.split()) > 200 else [context]
        for chunk in chunks:
            if chunk.strip():
                documents.append(Document(page_content=chunk))

    print(f"📚 Built semantic vectorstore with {len(documents)} chunks")
    return Chroma.from_documents(documents, embedding=embedding_model)
(planner-ai-patched) MacBookPro:rag michaelsigamani$ cat benchmark_run.py 
from langchain_community.vectorstores import Chroma
from rag_config import build_graph, build_vectorstore
from judge import judge_answer
from typing import List
import json

def format_context(pre_text, table, post_text):
    table_str = "\n".join(["\t".join(row) for row in table])
    context = []
    if isinstance(pre_text, list):
        context.append("\n".join(pre_text))
    else:
        context.append(pre_text)
    context.append(table_str)
    if isinstance(post_text, list):
        context.append("\n".join(post_text))
    else:
        context.append(post_text)
    return "\n\n".join([part for part in context if part.strip()])

def extract_examples(raw_data, max_examples):
    examples = []
    for entry in raw_data:
        try:
            question = entry["qa"]["question"]
            answer = entry["qa"]["answer"] if "answer" in entry["qa"] else str(entry["qa"].get("exe_ans", "N/A"))
            context = format_context(entry.get("pre_text", ""), entry.get("table", []), entry.get("post_text", ""))
            examples.append({
                "question": question,
                "answer": answer,
                "context": context
            })
        except Exception as e:
            continue
        if len(examples) >= max_examples:
            break
    return examples

def run_benchmark(examples: List[dict]):
    retriever = build_vectorstore([ex["context"] for ex in examples]).as_retriever(search_kwargs={"k": 2})
    graph = build_graph(retriever)

    correct = 0
    print("\n🔍 Running terminal benchmark on", len(examples), "examples\n")

    for i, example in enumerate(examples):
        question = example["question"]
        expected = example["answer"]

        print(f"--- Example {i+1} ---")
        print("Q:", question)
        state = graph.invoke({"question": question})
        answer = state.get("answer", "").strip()
        retrieved = state.get("retrieved_docs", [])
        eval_result = judge_answer(question, retrieved, answer, expected)

        print("Predicted:", answer)
        print("Expected:", expected)
        print(eval_result)
        print()

        if "✔️ Correct" in eval_result:
            correct += 1

    print(f"✅ Accuracy: {correct}/{len(examples)} = {100 * correct / len(examples):.1f}%\n")

if __name__ == "__main__":
    with open("data/dev.json") as f:
        raw_data = json.load(f)
    examples = extract_examples(raw_data, max_examples=50)
    run_benchmark(examples)
