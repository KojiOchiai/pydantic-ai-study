from dataclasses import dataclass, field

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext


@dataclass
class Note:
    pages: dict[str, str] = field(default_factory=dict)

    def add(self, title: str, content: str) -> None:
        if len(title) == 0:
            raise ValueError("Title cannot be empty")
        if len(content) == 0:
            raise ValueError("Content cannot be empty")
        if len(self.pages) >= 10:
            raise ValueError("You can only have up to 10 pages")
        if len(title) > 100:
            raise ValueError("Title cannot be longer than 100 characters")
        if len(content) > 1000:
            raise ValueError("Content cannot be longer than 1000 characters")
        self.pages[title] = content

    def remove(self, title: str) -> None:
        if title not in self.pages:
            raise ValueError("Title not found")
        del self.pages[title]

    def get_all_titles(self) -> list[str]:
        return list(self.pages.keys())

    def get_content(self, title: str) -> str:
        if title not in self.pages:
            raise ValueError("Title not found")
        return self.pages[title]


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
def plus_tool(values: list[float]) -> float:
    """
    Add all the values together.
    """
    return sum(values)


@agent.tool_plain
def minus_tool(values: list[float]) -> float:
    """
    Subtract all the values from the first value.
    """
    return values[0] - sum(values[1:])


@agent.tool
async def add_note_tool(ctx: RunContext[Note], title: str, content: str) -> None:
    """
    Add a note with the given title and content. Returns nothing.
    """
    ctx.deps.add(title, content)


@agent.tool
async def remove_note_tool(ctx: RunContext[Note], title: str) -> None:
    """
    Remove a note with the given title. Returns nothing.
    """
    ctx.deps.remove(title)


@agent.tool
async def get_all_titles_tool(ctx: RunContext[Note]) -> list[str]:
    """
    Get all the titles of the notes. Returns a list of strings.
    """
    return ctx.deps.get_all_titles()


@agent.tool
async def get_content_tool(ctx: RunContext[Note], title: str) -> str:
    """
    Get the content of the note with the given title. Returns a string.
    """
    return ctx.deps.get_content(title)


def main():
    questions = [
        "I am Hoge",
        "I like to play games",
        "what is my name?",
        "what is my favorite game?",
        "what is sum of 1, 2, 3, 4, 5?",
        "what is 1243 + 44",
        "what is 1243 - 44",
        "add a note with title 'test' and content 'test content'",
        "get all titles",
        "get the content of the note with title 'test'",
        "remove the note with title 'test'",
        "get all titles",
    ]
    message_history = []
    deps = Note()
    for question in questions:
        print("User: " + question)
        result = agent.run_sync(question, message_history=message_history, deps=deps)
        message_history = result.all_messages()
        print("Assistant: ", result.data)


if __name__ == "__main__":
    main()
