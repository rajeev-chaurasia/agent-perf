import asyncio
import json

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "mock-model", "object": "model"}
        ],
    }


async def _chat_stream():
    content_chunks = [
        {"choices": [{"delta": {"content": "Hello"}, "index": 0, "finish_reason": None}], "object": "chat.completion.chunk"},
        {"choices": [{"delta": {"content": " world"}, "index": 0, "finish_reason": None}], "object": "chat.completion.chunk"},
        {"choices": [{"delta": {"content": "!"}, "index": 0, "finish_reason": None}], "object": "chat.completion.chunk"},
        {"choices": [{"delta": {"content": " How"}, "index": 0, "finish_reason": None}], "object": "chat.completion.chunk"},
        {"choices": [{"delta": {"content": " are you?"}, "index": 0, "finish_reason": None}], "object": "chat.completion.chunk"},
    ]

    for chunk in content_chunks:
        yield "data: " + json.dumps(chunk) + "\n\n"
        await asyncio.sleep(0)

    usage_chunk = {
        "choices": [],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        "object": "chat.completion.chunk",
    }
    yield "data: " + json.dumps(usage_chunk) + "\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions():
    return StreamingResponse(_chat_stream(), media_type="text/event-stream")


@app.get("/metrics")
async def metrics():
    body = (
        "vllm:gpu_cache_usage_perc 0.42\n"
        "vllm:num_requests_running 1\n"
        "vllm:cache_hit_rate 0.65\n"
    )
    return StreamingResponse(iter([body]), media_type="text/plain")


def main():
    uvicorn.run(app, host="0.0.0.0", port=8765)


if __name__ == "__main__":
    main()
