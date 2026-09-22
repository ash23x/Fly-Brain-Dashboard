"""Send one JSON message to a live server's WebSocket, e.g.
    python scripts/send.py http://localhost:8766 "{\"odour\": \"A\"}"
    python scripts/send.py http://localhost:8765 "{\"cue\": 90}"
"""
import asyncio, json, sys
import aiohttp

async def main(url: str, msg: str) -> None:
    async with aiohttp.ClientSession() as s, s.ws_connect(url.rstrip("/") + "/ws") as ws:
        await ws.send_json(json.loads(msg))
        await asyncio.sleep(0.3)
        print("sent", msg)

asyncio.run(main(sys.argv[1], sys.argv[2]))
