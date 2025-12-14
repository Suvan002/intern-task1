from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import json
import Streamlit as st

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-chat-v1.o",
    task="text_generation"
)

model = ChatHuggingFace(llm=llm)

chat_history = [SystemMessage(content="You are an expert startup consultant. Analyze the given startup idea
and return a structured JSON object with the fields: problem,
customer, market, competitor, tech_stack, risk_level,
profitability_score, justification. Rules: - Keep answers concise and
realistic. - 'competitor' should contain exactly 3 competitors with
one-line differentiation each. - 'tech_stack' should be 4–6 practical
technologies for MVP. - 'profitability_score' must be an integer
between 0–100. Return ONLY JSON. Input: { "title": "", "description":
"" }")]

st.header('AI Startup Idea Validator')

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    chat_history.append(HumanMessage(content=user_input))

    try:
        result = model.invoke(chat_history)
        chat_history.append(AIMessage(content=result.content))

a=json.dumps([{"role": m.type, "content": m.content} for m in chat_history], indent=2))
if st.button("\nFinal conversation history:"):
    st.write(a)
