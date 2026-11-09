from typing import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

parent_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
subgraph_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")


# ---------- ONE shared state ----------
class ParentState(TypedDict):
    question: str
    english_answer: str
    hindi_answer: str


# ---------- SUBGRAPH (uses the SAME state keys) ----------
def translate_text(state: ParentState):
    prompt = (
        "Translate the following text into Hindi, natural and clear, nothing extra:\n\n"
        f"{state['english_answer']}"
    )
    result = subgraph_llm.invoke(prompt)
    return {"hindi_answer": result.content}


sub_builder = StateGraph(ParentState)
sub_builder.add_node("translate_text", translate_text)
sub_builder.add_edge(START, "translate_text")
sub_builder.add_edge("translate_text", END)
subgraph = sub_builder.compile()


# ---------- PARENT ----------
def generate_answer(state: ParentState):
    result = parent_llm.invoke(
        f"You are a helpful assistant. Answer this:\n\n{state['question']}"
    )
    return {"english_answer": result.content}


parent_builder = StateGraph(ParentState)
parent_builder.add_node("generate", generate_answer)
parent_builder.add_node("translate", subgraph)      # ← subgraph AS a node
parent_builder.add_edge(START, "generate")
parent_builder.add_edge("generate", "translate")
parent_builder.add_edge("translate", END)
parent_graph = parent_builder.compile()


if __name__ == "__main__":
    out = parent_graph.invoke({"question": "Explain gradient descent in very simple terms"})
    print("HI:", out["hindi_answer"])