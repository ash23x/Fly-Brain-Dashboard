"""Drive the live compass server over its WebSocket and report what the needle does."""
import asyncio, json, time, sys
import aiohttp

URL = "http://localhost:8765"

async def main():
    async with aiohttp.ClientSession() as s:
        async with s.get(URL + "/neurons") as r:
            meta = await r.json()
        print("mode", meta.get("mode"), "n", meta["n"], "edges", meta["n_edges"], "step_hz", meta.get("step_hz"), "params", meta.get("params"))
        async with s.ws_connect(URL + "/ws") as ws:
            async def watch(seconds, label):
                t0 = time.time(); last = None; n = 0; hs = []; per_sec = []
                nxt = t0 + 1
                while time.time() - t0 < seconds:
                    msg = await ws.receive(timeout=5)
                    d = json.loads(msg.data); n += 1; last = d; hs.append(d["heading"])
                    if time.time() >= nxt:
                        per_sec.append(f"{d['heading']:5.0f}"); nxt += 1
                print(f"{label:22s} {hs[0]:6.1f} -> {hs[-1]:6.1f}  per-second: {' '.join(per_sec)}   R={last['strength']:.2f} spk/f={len(last['spikes'])} re={last.get('reignitions')}")
                return hs
            plan = [("cue", 90, 3, "landmark at 90"), ("turn", 1.0, 3, "turn +1 (right)"), ("turn", 0.0, 8, "released"),
                    ("turn", -1.0, 3, "turn -1 (left)"), ("turn", 0.0, 8, "released"),
                    ("turn", 0.6, 4, "turn +0.6"), ("turn", 0.0, 8, "released"),
                    ("reset", True, 5, "killed -> re-forms?")]
            for key, val, secs, label in plan:
                await ws.send_json({key: val})
                await watch(secs, label)

asyncio.run(main())
