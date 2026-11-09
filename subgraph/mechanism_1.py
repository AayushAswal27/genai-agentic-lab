from typing import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

parent_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
subgraph_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")


# ---------- SUBGRAPH (its own state) ----------
class SubState(TypedDict):
    input_text: str
    translated_text: str


def translate_text(state: SubState):
    prompt = (
        "Translate the following text into Hindi. Keep it natural and clear, "
        f"add nothing extra:\n\n{state['input_text']}"
    )
    result = subgraph_llm.invoke(prompt)
    return {"translated_text": result.content}


sub_builder = StateGraph(SubState)
sub_builder.add_node("translate_text", translate_text)
sub_builder.add_edge(START, "translate_text")
sub_builder.add_edge("translate_text", END)
subgraph = sub_builder.compile()


# ---------- PARENT (its own, different state) ----------
class ParentState(TypedDict):
    question: str
    english_answer: str
    hindi_answer: str


def generate_answer(state: ParentState):
    result = parent_llm.invoke(
        f"You are a helpful assistant. Answer this question:\n\n{state['question']}"
    )
    return {"english_answer": result.content}


def translate_answer(state: ParentState):
    sub_result = subgraph.invoke({"input_text": state["english_answer"]})
    return {"hindi_answer": sub_result["translated_text"]}


parent_builder = StateGraph(ParentState)
parent_builder.add_node("generate", generate_answer)
parent_builder.add_node("translate", translate_answer)
parent_builder.add_edge(START, "generate")
parent_builder.add_edge("generate", "translate")
parent_builder.add_edge("translate", END)
parent_graph = parent_builder.compile()


if __name__ == "__main__":
    out = parent_graph.invoke({"question": "Explain gradient descent in very simple terms"})
    print("EN:", out["english_answer"])
    print("HI:", out["hindi_answer"])