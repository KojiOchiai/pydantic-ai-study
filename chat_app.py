import fastapi

from agent import ResultMessage, agent

app = fastapi.FastAPI()


@app.post("/chat")
async def post_chat(message: str) -> ResultMessage:
    result = await agent.run(message)
    return result.data
