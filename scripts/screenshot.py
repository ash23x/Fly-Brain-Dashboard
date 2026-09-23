"""Screenshot a live dashboard page with headless Chrome, waiting for the
WebSocket to connect and the canvases to draw (plain --screenshot doesn't).

    python scripts/screenshot.py http://localhost:8765/ docs/compass.png [wait_seconds] [css-selector]

Set SHOT_HEIGHT (default 1180) for a taller capture.
"""
import asyncio, base64, json, os, pathlib, subprocess, sys, time
import aiohttp

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PORT = 9333
WIDTH, HEIGHT = 1280, int(os.environ.get("SHOT_HEIGHT", "1180"))

async def main(url: str, out: str, wait: float, selector: str | None = None) -> None:
    proc = subprocess.Popen([CHROME, "--headless=new", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--hide-scrollbars",
                             f"--window-size={WIDTH},{HEIGHT}", f"--remote-debugging-port={PORT}",
                             "--user-data-dir=" + str(pathlib.Path.home() / ".fly-shot-profile"), "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        async with aiohttp.ClientSession() as s:
            for _ in range(50):
                try:
                    async with s.get(f"http://localhost:{PORT}/json") as r:
                        tabs = await r.json()
                    break
                except Exception:
                    await asyncio.sleep(0.2)
            page = next(t for t in tabs if t["type"] == "page")
            async with s.ws_connect(page["webSocketDebuggerUrl"], max_msg_size=64 * 1024 * 1024) as ws:
                msg_id = 0
                async def cmd(method, **params):
                    nonlocal msg_id
                    msg_id += 1
                    await ws.send_json({"id": msg_id, "method": method, "params": params})
                    while True:
                        m = json.loads((await ws.receive()).data)
                        if m.get("id") == msg_id:
                            return m.get("result", {})
                await cmd("Emulation.setDeviceMetricsOverride", width=WIDTH, height=HEIGHT, deviceScaleFactor=1, mobile=False)
                await cmd("Network.enable")
                await cmd("Network.setCacheDisabled", cacheDisabled=True)
                await cmd("Page.navigate", url=url)
                await asyncio.sleep(wait)
                if selector:
                    await cmd("Runtime.evaluate", expression=f"document.querySelector({selector!r}).scrollIntoView({{block:'start'}})")
                    await asyncio.sleep(1.5)
                shot = await cmd("Page.captureScreenshot", format="png")
                pathlib.Path(out).write_bytes(base64.b64decode(shot["data"]))
                print(f"wrote {out} ({pathlib.Path(out).stat().st_size:,} bytes)")
    finally:
        proc.kill()

if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 8.0,
                     sys.argv[4] if len(sys.argv) > 4 else None))
