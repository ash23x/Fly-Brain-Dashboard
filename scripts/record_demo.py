"""Record the live compass: a GIF for the README and an MP4 for social posts.

    python scripts/record_demo.py [url] [gif_out] [mp4_out]

Drives headless Chrome over the DevTools protocol, as screenshot.py does: waits
for the simulation and the 3D skeletons, drops a landmark on the ring, walks the
bump round with a held turn and lets go just short of this fly's favourite
heading, while a screencast keeps every frame with its time.
ffmpeg then crops the ring and the brain side by side. Needs Chrome, aiohttp
and ffmpeg on PATH. Budget, set before recording: GIF under 5 MB and 900 px
wide, about 10 s; it steps down until it fits and prints what it made.
Software GL by default (what screenshot.py proved); set DEMO_GL=gpu to use
the graphics card for a smoother capture.
"""
import asyncio, base64, json, math, os, pathlib, shutil, subprocess, sys, tempfile
import aiohttp

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PORT = 9334
W, H = 1280, 800
URL = sys.argv[1] if len(sys.argv) > 1 else "https://fly-brain-ivory.vercel.app/compass"
GIF = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "docs/compass.gif").resolve()
MP4 = pathlib.Path(sys.argv[3] if len(sys.argv) > 3 else "compass_demo.mp4").resolve()
GIF_MAX = 5_000_000
GIF_STEPS = [(860, 12, 192), (800, 10, 160), (720, 10, 128), (640, 9, 112)]   # width, fps, colours
# software GL is what screenshot.py proved; DEMO_GL=gpu asks Chrome for the graphics card instead
GL_FLAGS = (["--use-angle=d3d11", "--enable-gpu", "--ignore-gpu-blocklist"] if os.environ.get("DEMO_GL") == "gpu"
            else ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])


class CDP:
    def __init__(self, ws):
        self.ws, self.n, self.pending, self.handlers = ws, 0, {}, {}
        self.reader = asyncio.ensure_future(self._read())

    async def _read(self):
        async for msg in self.ws:
            m = json.loads(msg.data)
            if "id" in m:
                f = self.pending.pop(m["id"], None)
                if f and not f.done():
                    f.set_result(m.get("result", {}))
            elif m.get("method") in self.handlers:
                self.handlers[m["method"]](m.get("params", {}))

    async def send(self, method, **params):
        self.n += 1
        f = asyncio.get_running_loop().create_future()
        self.pending[self.n] = f
        await self.ws.send_json({"id": self.n, "method": method, "params": params})
        return await asyncio.wait_for(f, 30)

    async def js(self, expr):
        r = await self.send("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        return r.get("result", {}).get("value")


async def record(tmp: pathlib.Path):
    frames = []
    prof = pathlib.Path.home() / ".fly-shot-profile"
    proc = subprocess.Popen([CHROME, "--headless=new", *GL_FLAGS, "--hide-scrollbars", "--mute-audio", "--no-first-run", f"--window-size={W},{H}",
                             f"--remote-debugging-port={PORT}", f"--user-data-dir={prof}", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        async with aiohttp.ClientSession() as s:
            tabs = []
            for _ in range(75):
                try:
                    async with s.get(f"http://127.0.0.1:{PORT}/json") as r:
                        tabs = await r.json()
                    break
                except Exception:
                    await asyncio.sleep(0.2)
            page = next(t for t in tabs if t["type"] == "page")
            async with s.ws_connect(page["webSocketDebuggerUrl"], max_msg_size=256 * 1024 * 1024) as ws:
                cdp = CDP(ws)

                def on_frame(p):
                    path = tmp / f"f{len(frames):05d}.jpg"
                    path.write_bytes(base64.b64decode(p["data"]))
                    frames.append((p["metadata"]["timestamp"], path))
                    asyncio.ensure_future(cdp.send("Page.screencastFrameAck", sessionId=p["sessionId"]))
                cdp.handlers["Page.screencastFrame"] = on_frame

                await cdp.send("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=1, mobile=False)
                await cdp.send("Page.enable")
                await cdp.send("Page.navigate", url=URL)
                st = None
                for _ in range(90):
                    await asyncio.sleep(0.5)
                    st = await cdp.js("[document.getElementById('statusText')?.textContent || '',"
                                      " document.getElementById('brainStatus')?.textContent || '']")
                    if st and "running" in st[0] and "segments from" in st[1]:
                        break
                print("page:", st)
                await cdp.js("(() => { const s = document.getElementById('brainBright'); s.value = 70;"
                             " s.dispatchEvent(new Event('input')); s.dispatchEvent(new Event('change')); })()")
                await asyncio.sleep(4)                      # the bump settles, the skeletons light
                geo = await cdp.js("(() => { const g = id => { const r = document.getElementById(id).getBoundingClientRect();"
                                   " return [r.left, r.top, r.width, r.height]; }; return {ring: g('ring'), brain: g('brain')}; })()")
                gl = await cdp.js("(() => { const c = document.createElement('canvas').getContext('webgl');"
                                  " const e = c && c.getExtension('WEBGL_debug_renderer_info');"
                                  " return e ? c.getParameter(e.UNMASKED_RENDERER_WEBGL) : 'unknown'; })()")
                fps = await cdp.js("new Promise(res => { let n = 0; const t0 = performance.now(); const f = () => {"
                                   " n++; if (performance.now() - t0 < 1000) requestAnimationFrame(f); else res(n); };"
                                   " requestAnimationFrame(f); })") or 0
                print(f"webgl: {gl}; the page draws {fps} frames/s")
                rx, ry, rw, rh = geo["ring"]
                cx, cy, R = rx + rw / 2, ry + rh / 2, 0.36 * rw

                heading = lambda: cdp.js("parseFloat(document.getElementById('headingValue').textContent) || 0")
                gap = lambda p, q: abs((p - q + 540) % 360 - 180)
                # Released bumps drift toward this fly's favourite headings (measured 7 Oct 2026, six releases:
                # median ~26 deg over 10 s, up to 80); ~300 is the strongest. The turn is let go just before it,
                # so the clip shows a hold that is real - the README says plainly that elsewhere it drifts.
                HOME = 300
                await cdp.send("Page.startScreencast", format="jpeg", quality=88, everyNthFrame=2 if fps > 40 else 1)
                await asyncio.sleep(1.6)                    # it holds a heading
                h0 = await heading()
                cue = 120 if gap(h0, 120) >= 70 else 180    # well away from the bump, half a lap before HOME
                a = math.radians(cue)                       # 0 at the top, clockwise, as the page reads it
                x, y = cx + R * math.sin(a), cy - R * math.cos(a)
                await cdp.send("Input.dispatchMouseEvent", type="mouseMoved", x=x, y=y, button="none")
                for t in ("mousePressed", "mouseReleased"):
                    await cdp.send("Input.dispatchMouseEvent", type=t, x=x, y=y, button="left", clickCount=1)
                await asyncio.sleep(3.2)                    # a landmark: the bump jumps to it
                ev = dict(key="ArrowRight", code="ArrowRight", windowsVirtualKeyCode=39, nativeVirtualKeyCode=39)
                await cdp.send("Input.dispatchKeyEvent", type="keyDown", **ev)   # the fly turns: the bump walks round
                loop = asyncio.get_running_loop()
                t0 = loop.time()
                while loop.time() - t0 < 4.5:
                    await asyncio.sleep(0.04)
                    if loop.time() - t0 > 1.0 and (HOME - await heading()) % 360 < 14:
                        break
                await cdp.send("Input.dispatchKeyEvent", type="keyUp", **ev)
                h_rel = await heading()
                await asyncio.sleep(3.2)                    # let go: it holds, on nothing
                h_end = await heading()
                await cdp.send("Page.stopScreencast")
                await asyncio.sleep(0.3)
                print(f"heading at start {h0:.0f}, landmark at {cue}, released at {h_rel:.0f}, 3.2 s later {h_end:.0f}")
                return frames, geo
    finally:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)


def encode(frames, geo, tmp):
    lines = ["ffconcat version 1.0"]
    for i, (ts, p) in enumerate(frames):
        d = frames[i + 1][0] - ts if i + 1 < len(frames) else 0.1
        lines += [f"file '{p.name}'", f"duration {max(d, 0.001):.4f}"]
    lines.append(f"file '{frames[-1][1].name}'")
    (tmp / "frames.txt").write_text("\n".join(lines))
    fw = int(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width",
                             "-of", "csv=p=0", str(frames[0][1])], capture_output=True, text=True).stdout.strip())
    k = fw / W                                           # frame pixels per CSS pixel
    (rx, ry, rw, rh), (bx, by, bw, bh) = geo["ring"], geo["brain"]
    x0, y0 = max(0, min(rx, bx) - 14), max(0, min(ry, by) - 40)     # keep both card titles
    x1, y1 = min(W, max(rx + rw, bx + bw) + 14), min(H, min(ry + rh, by + bh) + 6)   # stop above the legends
    cw, ch = int((x1 - x0) * k) // 2 * 2, int((y1 - y0) * k) // 2 * 2
    crop = f"crop={cw}:{ch}:{int(x0 * k)}:{int(y0 * k)}"
    src = ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", "frames.txt"]
    for gw, fps, cols in GIF_STEPS:
        lavfi = (f"{crop},fps={fps},scale={gw}:-2:flags=lanczos,split[a][b];[a]palettegen=max_colors={cols}"
                 f":stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
        subprocess.run(src + ["-filter_complex", lavfi, "-loop", "0", str(GIF)], cwd=tmp, check=True)
        size = GIF.stat().st_size
        print(f"gif {gw} px, {fps} fps, {cols} colours: {size / 1e6:.2f} MB")
        if size <= GIF_MAX:
            break
    pad = ("scale=1280:720:force_original_aspect_ratio=decrease:flags=lanczos,"
           "pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=0x0b0e14")
    subprocess.run(src + ["-vf", f"{crop},fps=30,{pad},format=yuv420p", "-c:v", "libx264", "-crf", "20",
                          "-preset", "slow", "-movflags", "+faststart", str(MP4)], cwd=tmp, check=True)
    print(f"mp4 {MP4.stat().st_size / 1e6:.2f} MB -> {MP4}")
    still = frames[int(len(frames) * 0.55)][1]
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(still), "-vf", crop, str(MP4.with_suffix(".png"))], check=True)
    print(f"crop {crop} from {fw} px frames")


if __name__ == "__main__":
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="flydemo_"))
    frames, geo = asyncio.run(record(tmp))
    span = frames[-1][0] - frames[0][0] if frames else 0
    print(f"{len(frames)} frames over {span:.1f} s; ring {geo['ring']}, brain {geo['brain']}")
    if len(frames) < 50:
        sys.exit(f"too few frames - the page probably never drew; frames kept in {tmp}")
    GIF.parent.mkdir(parents=True, exist_ok=True)
    MP4.parent.mkdir(parents=True, exist_ok=True)
    encode(frames, geo, tmp)
    shutil.rmtree(tmp, ignore_errors=True)
