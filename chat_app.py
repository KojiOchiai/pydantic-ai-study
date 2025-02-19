import fastapi

from agent import agent

app = fastapi.FastAPI()


@app.post("/chat")
async def post_chat(message: str):
    result = await agent.run(message)
    return result.data.message
