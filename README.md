# Chatbot
Simple working python chatbot using langchain

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseOutputParser


api_key="Your API Key"

llm = ChatOpenAI(
    temperature=0.7,
    openai_api_key=api_key,
    model="gpt-4o"  
)

memory = ConversationBufferMemory()

template = """
You are a helpful AI assistant. You will answer the user's questions conversationally and helpfully.

{history}
Human: {input}
AI:"""

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=template,
)

conversation = ConversationChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True  
)


if __name__ == "__main__":
    print("Chatbot is running. Type 'exit' to stop.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = conversation.predict(input=user_input)
        print(f"Bot: {response}")
