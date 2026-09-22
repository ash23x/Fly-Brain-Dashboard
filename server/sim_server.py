"""
Runs one of the fly circuits and serves its dashboard on localhost.

    python server/sim_server.py --mode compass                  # http://localhost:8765/
    python server/sim_server.py --mode mushroom --port 8766     # http://localhost:8766/

Options: --host (default localhost; 0.0.0.0 exposes it to your LAN, which
also disables the webcam button on non-localhost pages -- browsers only
allow camera access on localhost or HTTPS), --hz frames per second sent
to the browser (30), --substeps simulation steps per frame (3).

Routes:  /            the mode's page (dashboard/compass.html or mushroom.html)
         /neurons     circuit metadata as JSON
         /ws          WebSocket: one JSON message per frame from the server;
                      the page sends small JSON commands back
Commands the page may send:
    compass:   {"turn": -1..1}  {"cue": degrees}  {"reset": true}
    mushroom:  {"odour": "A"|"B"|"C"|null}  {"reward": 0..1}  {"punish": 0..1}
               {"forget": true}  {"reset": true}
Nothing here talks to anything but your own browser.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys
import time

from aiohttp import WSMsgType, web

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from compass_sim import CompassSim  # noqa: E402
from mushroom_sim import MushroomSim  # noqa: E402

MODES = {
    "compass": {"dir": ROOT / "data" / "compass", "page": "compass.html",
                "build": "scripts/03_build_compass.py", "cls": CompassSim},
    "mushroom": {"dir": ROOT / "data" / "mushroom", "page": "mushroom.html",
                 "build": "scripts/05_build_mushroom_body.py", "cls": MushroomSim},
}
PAGES = ROOT / "dashboard"


def read_params(circuit_dir: pathlib.Path) -> dict:
    """The tuned gains, if a params.json is present (keys starting with _ are notes)."""
    path = circuit_dir / "params.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return {k: v for k, v in json.load(f).items() if not str(k).startswith("_")}


SKELETON_FILES = {"skeletons.json": "application/json", "skeleton_pos.bin": "application/octet-stream",
                  "skeleton_idx.bin": "application/octet-stream"}


class Server:
    def __init__(self, mode: str, hz: float, substeps: int):
        spec = MODES[mode]
        self.mode = mode
        self.circuit_dir = spec["dir"]
        self.page = PAGES / spec["page"]
        self.hz = hz
        self.substeps = substeps
        self.sim = spec["cls"](spec["dir"], **read_params(spec["dir"]))
        self.clients: set[web.WebSocketResponse] = set()
        self.frame = 0

    # ---- HTTP ----------------------------------------------------------------
    async def index(self, request: web.Request) -> web.FileResponse:
        return web.FileResponse(self.page, headers={"Cache-Control": "no-store"})

    async def skeleton(self, request: web.Request) -> web.StreamResponse:
        """The packed 3D skeletons, if scripts/06 and 07 have been run for this circuit."""
        name = request.match_info["name"]
        if name not in SKELETON_FILES:
            raise web.HTTPNotFound()
        path = self.circuit_dir / name
        if not path.exists():
            raise web.HTTPNotFound(text="no skeletons built: run scripts/06_fetch_skeletons.py and 07_build_skeletons.py")
        return web.FileResponse(path, headers={"Content-Type": SKELETON_FILES[name], "Cache-Control": "no-store"})

    async def neurons(self, request: web.Request) -> web.Response:
        sim = self.sim
        meta = {"mode": self.mode, "neurons": sim.neurons, "n": sim.n, "n_edges": sim.nnz,
                "step_hz": self.hz * self.substeps, "frame_hz": self.hz}
        if self.mode == "compass":
            meta.update(n_epg=len(sim.epg_idx), n_pen=len(sim.pen_idx), n_peg=len(sim.peg_idx),
                        n_delta7=len(sim.d7_idx), n_inhibitory=sim.n_inhibitory,
                        params={k: getattr(sim, k) for k in ("gE", "gI", "epg_drive", "pen_drive",
                                                             "turn_gain", "turn_tau", "noise_std")})
        else:
            meta.update(counts={r: int(len(sim.idx[r])) for r in sim.idx}, n_plastic=sim.n_plastic,
                        glomeruli=sim.glomeruli, odours=sim.odours,
                        mbon_valence=[sim.neurons[i]["valence"] for i in sim.idx["mbon"]],
                        mbon_types=[sim.neurons[i]["type"] for i in sim.idx["mbon"]])
        return web.json_response(meta)

    # ---- WebSocket ------------------------------------------------------------
    async def ws(self, request: web.Request) -> web.WebSocketResponse:
        sock = web.WebSocketResponse(heartbeat=20)
        await sock.prepare(request)
        self.clients.add(sock)
        try:
            async for msg in sock:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    cmd = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                if isinstance(cmd, dict):
                    self.apply(cmd)
        finally:
            self.clients.discard(sock)
        return sock

    def apply(self, cmd: dict) -> None:
        sim = self.sim

        def num(key, lo, hi):
            try:
                return max(lo, min(hi, float(cmd[key])))
            except (TypeError, ValueError):
                return None

        if self.mode == "compass":
            if "turn" in cmd and (v := num("turn", -1.0, 1.0)) is not None:
                sim.set_turn(v)
            if "cue" in cmd and (v := num("cue", -1e9, 1e9)) is not None:
                sim.cue(v % 360.0, strength=2.0, steps=60)
        else:
            if "odour" in cmd and (cmd["odour"] is None or cmd["odour"] in sim.odours):
                sim.set_odour(cmd["odour"])
            if "reward" in cmd and (v := num("reward", 0.0, 1.0)) is not None:
                sim.set_reward(v)
            if "punish" in cmd and (v := num("punish", 0.0, 1.0)) is not None:
                sim.set_punish(v)
            if cmd.get("forget"):
                sim.forget()
        if cmd.get("reset"):
            sim.reset()

    # ---- the loop ---------------------------------------------------------------
    async def run(self, app: web.Application) -> None:
        period = 1.0 / self.hz
        due = time.perf_counter()
        while True:
            fired: set[int] = set()
            for _ in range(self.substeps):
                out = self.sim.step()
                fired.update(out["spikes"])
            out["spikes"] = sorted(fired)
            out["t"] = time.time()
            if self.mode == "compass":
                out["reignitions"] = self.sim.reignitions
            elif self.frame % 6 == 0:
                out["mbon_trace"] = self.sim.mbon_trace()
            self.frame += 1
            payload = json.dumps(out)
            for sock in list(self.clients):
                if sock.closed:
                    self.clients.discard(sock)
                    continue
                try:
                    await sock.send_str(payload)
                except Exception:
                    self.clients.discard(sock)
            due += period
            delay = due - time.perf_counter()
            if delay < -period:           # fell far behind (sleep, debugger): resynchronise
                due = time.perf_counter()
                delay = 0.0
            await asyncio.sleep(max(0.0, delay))


def make_app(mode: str, hz: float = 30.0, substeps: int = 3) -> web.Application:
    server = Server(mode, hz, substeps)
    app = web.Application()
    app.router.add_get("/", server.index)
    app.router.add_get("/neurons", server.neurons)
    app.router.add_get("/ws", server.ws)
    app.router.add_get("/skeletons/{name}", server.skeleton)
    app.router.add_static("/static/", PAGES, show_index=False)

    async def start(app: web.Application) -> None:
        app["loop_task"] = asyncio.create_task(server.run(app))
        sim = server.sim
        print(f"{mode}: {sim.n} neurons, {sim.nnz:,} synaptic connections, "
              f"{substeps} steps per frame at {hz:g} frames/s")

    async def stop(app: web.Application) -> None:
        task = app.get("loop_task")
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    app.on_startup.append(start)
    app.on_cleanup.append(stop)
    return app


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Serve one of the fly circuits on localhost.")
    p.add_argument("--mode", choices=list(MODES), default="compass")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--hz", type=float, default=30.0)
    p.add_argument("--substeps", type=int, default=3)
    a = p.parse_args(argv)
    spec = MODES[a.mode]
    if not (spec["dir"] / "adjacency.npz").exists():
        raise SystemExit(f"No circuit in {spec['dir']}. Run scripts/01_download_data.py then {spec['build']}.")
    print(f"http://{a.host}:{a.port}/   (Ctrl+C to stop)")
    web.run_app(make_app(a.mode, a.hz, a.substeps), host=a.host, port=a.port, print=None)


if __name__ == "__main__":
    main()
