from typing import Any
import asyncio

from a2a.types import Task, TaskState
from utils.config import SUPERVISOR_PORT
from utils.protocol_wrappers import (
    extract_text,
    send_text_async
)

async def handle_request(query: str) -> str:
    resp_obj = await send_text_async(SUPERVISOR_PORT, query)
    feedback = extract_text(resp_obj)
    print(resp_obj.model_dump_json(indent=2, exclude_none=True))
    print(feedback)
    return feedback

if __name__ == "__main__":
    asyncio.run(handle_request("Hello how are you?"))