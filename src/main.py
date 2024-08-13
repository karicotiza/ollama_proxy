"""Main app. Redirects requests to local ollama, adds simple authentication."""

import json
from os import environ
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from httpx import AsyncClient
from pydantic import BaseModel
from uvicorn import run


class Conf:
    """App Config."""

    token: str = environ['OLLAMA_PROXY_TOKEN']
    external: str = environ['OLLAMA_PROXY_EXTERNAL_ADDRESS']
    local: str = environ['OLLAMA_PROXY_LOCAL_ADDRESS']

    model: str = 'llama3.1:70b'
    endpoint: str = ''.join(('http://', local, '/api/generate'))

    code: int = 401
    detail: str = 'token is not valid'

    media: str = 'application/json'

    timeout: int = 120


class APIResponse(BaseModel):
    """API Response model."""

    token: str
    done: bool = False


class APIRequest(BaseModel):
    """API Request model."""

    question: str
    token: str

    async def is_token_valid(self) -> bool:
        """Token validation.

        Returns:
            bool: True if valid
        """
        return self.token == Conf.token

    async def re_stream(self) -> AsyncGenerator[str, None]:
        """Re-stream data from local ollama.

        Yields:
            Iterator[AsyncGenerator[str, None]]: json-like string of data.
        """
        load: dict[str, str] = {'model': Conf.model, 'prompt': self.question}

        args: tuple[str, ...] = 'post', Conf.endpoint
        kwargs: dict = {'json': load, 'timeout': Conf.timeout}

        async with client.stream(*args, **kwargs) as response:
            async for line in response.aiter_lines():
                yield await self._process_line(line)

    async def _process_line(self, line: str):
        response: dict = json.loads(line)

        api_response: APIResponse = APIResponse(
            token=response['response'],
            done=response['done'],
        )

        return await self._to_string(api_response)

    async def _to_string(self, api_response: APIResponse) -> str:
        return ''.join((api_response.model_dump_json(), '\n'))


app: FastAPI = FastAPI()
client: AsyncClient = AsyncClient()


@app.post('/')
async def generate(request: APIRequest) -> StreamingResponse:
    """Question endpoint.

    Args:
        request (APIRequest): API request.

    Raises:
        HTTPException: if token is not valid.

    Returns:
        StreamingResponse: stream of data.
    """
    if await request.is_token_valid():
        return StreamingResponse(request.re_stream(), media_type=Conf.media)

    raise HTTPException(Conf.code, detail=Conf.detail)

if __name__ == '__main__':
    host, port = Conf.external.split(':')
    run('main:app', host=host, port=int(port))
