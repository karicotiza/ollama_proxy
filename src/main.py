"""Main app. Redirects requests to local ollama, adds simple authentication."""

from os import environ
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from httpx import AsyncClient
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


app: FastAPI = FastAPI()
client: AsyncClient = AsyncClient()


async def token_valid(payload: dict) -> bool:
    """Token validation.

    Args:
        payload (dict): json dict.

    Returns:
        bool: True if valid.
    """
    model: str = payload.get('model', '')
    token: str = model.split(' ')[0]
    return token == Conf.token


async def re_stream(payload: dict) -> AsyncGenerator[str, None]:
    """Re-stream data from local ollama.

    Args:
        payload (dict): json payload.

    Yields:
        Iterator[AsyncGenerator[str, None]]: json-like string of data.
    """
    args: tuple[str, ...] = 'post', Conf.endpoint
    kwargs: dict = {'json': payload, 'timeout': Conf.timeout}

    async with client.stream(*args, **kwargs) as response:
        async for line in response.aiter_lines():
            yield ''.join((line, '\n'))


@app.post('/api/generate')
async def generate(request: Request) -> StreamingResponse:
    """Question endpoint.

    Args:
        request (Request): API request.

    Raises:
        HTTPException: if token is not valid.

    Returns:
        StreamingResponse: stream of data.
    """
    payload: dict = await request.json()

    if await token_valid(payload):
        payload['model'] = payload['model'].split(' ')[1]
        return StreamingResponse(re_stream(payload), media_type=Conf.media)

    raise HTTPException(Conf.code, detail=Conf.detail)

if __name__ == '__main__':
    host, port = Conf.external.split(':')
    run('main:app', host=host, port=int(port))
