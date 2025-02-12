from pydantic import BaseModel, Field
from pydantic_ai import Agent


class ResultMessage(BaseModel):
    message: str = Field(description="The message to be sent to the user")
    is_request: bool = Field(description="Whether the message is a request for user")


agent = Agent(
    model="openai:gpt-4o",
    result_type=ResultMessage,
    system_prompt=(
        "You are a helpful assistant"
        " that can answer questions."
        " Ask the user for information if you need it."
    ),
)


@agent.tool_plain
def sum_tool(values: list[float]) -> float:
    print(values)
    return sum(values)


def main():
    questions = [
        "I am Hoge",
        "I like to play games",
        "what is my name?",
        "what is my favorite game?",
        "what is sum of 1, 2, 3, 4, 5?",
        "what is 1243 + 44",
    ]
    message_history = []
    for question in questions:
        print("User: " + question)
        result = agent.run_sync(question, message_history=message_history)
        message_history = result.all_messages()
        print("Assistant: ", result.data)
    print(result)


main()
