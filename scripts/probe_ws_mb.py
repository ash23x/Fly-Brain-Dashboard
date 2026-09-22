"""Drive the live mushroom-body server over its WebSocket: Pavlov in 40 seconds."""
import asyncio, json, time, sys
import aiohttp

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8766"

async def main():
    async with aiohttp.ClientSession() as s:
        async with s.get(URL + "/neurons") as r:
            meta = await r.json()
        print("mode", meta.get("mode"), "n", meta["n"], "plastic", meta.get("n_plastic"), "step_hz", meta.get("step_hz"))
        async with s.ws_connect(URL + "/ws") as ws:
            async def watch(seconds, label):
                t0 = time.time(); last = None; n = 0; prefs = []
                while time.time() - t0 < seconds:
                    msg = await ws.receive(timeout=5)
                    d = json.loads(msg.data); n += 1; last = d
                    if time.time() - t0 > seconds / 2: prefs.append(d["pref"])
                p = sum(prefs) / max(1, len(prefs))
                print(f"{label:26s} frames={n:3d} KC={last['kc_sparsity'] * 100:4.1f}%  pref={p:+.2f}  "
                      f"depressed={last['n_depressed']:,}  naive={last['naive']} learned={last['learned']}")
            plan = [({"odour": "A"}, 3, "smell A (naive)"), ({"odour": None}, 1, "air"),
                    ({"odour": "B"}, 3, "smell B (naive)"), ({"odour": None}, 1, "air")]
            for _ in range(3):
                plan += [({"odour": "A", "reward": 1}, 2.5, "A + sugar"), ({"odour": None, "reward": 0}, 1, "air")]
            plan += [({"odour": "A"}, 3, "test A"), ({"odour": None}, 1, "air"), ({"odour": "B"}, 3, "test B (control)"),
                     ({"odour": None}, 1, "air")]
            for _ in range(3):
                plan += [({"odour": "B", "punish": 1}, 2.5, "B + shock"), ({"odour": None, "punish": 0}, 1, "air")]
            plan += [({"odour": "B"}, 3, "test B"), ({"odour": None}, 1, "air"), ({"odour": "A"}, 3, "test A (retained?)")]
            for msg, secs, label in plan:
                await ws.send_json(msg)
                await watch(secs, label)

asyncio.run(main())
