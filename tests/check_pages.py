"""Drive the published (static) pages in a real browser and check the fly runs in the tab.
    python tests/check_pages.py http://localhost:8090 [chromium|firefox]
Chromium runs headless with SwiftShader; Firefox needs a display (xvfb-run) for WebGL.
"""
import json, sys, time
from playwright.sync_api import sync_playwright

base = sys.argv[1].rstrip('/') if len(sys.argv) > 1 else 'http://localhost:8090'
browser_name = sys.argv[2] if len(sys.argv) > 2 else 'chromium'

def launch(p):
    if browser_name == 'firefox':
        return p.firefox.launch(headless=False, firefox_user_prefs={"webgl.force-enabled": True, "gfx.webrender.software": True,
                                                                      "webgl.disable-fail-if-major-performance-caveat": True})
    return p.chromium.launch(headless=True, args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])

def wait_for(page, expr, timeout=60):
    for _ in range(int(timeout * 2)):
        if page.evaluate(expr):
            return True
        time.sleep(0.5)
    return False

with sync_playwright() as p:
    b = launch(p)
    page = b.new_page(viewport={"width": 1280, "height": 900})
    logs = []
    page.on("console", lambda m: logs.append(f"[{m.type}] {m.text}") if m.type in ("error", "warning") else None)
    page.on("pageerror", lambda e: logs.append(f"[pageerror] {e}"))

    # ---- compass ----
    page.goto(base + '/compass.html')
    ok = wait_for(page, "() => document.getElementById('statusText').textContent.startsWith('running in this tab')", 60)
    print('compass status:', page.evaluate("() => document.getElementById('statusText').textContent"), '| ok' if ok else '| FAILED')
    wait_for(page, "() => typeof brain !== 'undefined' && brain && brain.state.ready", 90)
    time.sleep(4)
    h0 = page.evaluate("() => latest && latest.heading")
    st = page.evaluate("() => ({hz: document.getElementById('statHz').textContent, heading: latest.heading, strength: latest.strength, brain: document.getElementById('brainStatus').textContent, w: brain.state.gl.canvas.width})")
    print('compass frames:', json.dumps(st))
    page.dispatch_event('#rightBtn', 'pointerdown'); time.sleep(2.0); page.dispatch_event('#rightBtn', 'pointerup'); time.sleep(0.5)
    h1 = page.evaluate("() => latest.heading")
    print(f'compass turn right 2 s: {h0:.0f} -> {h1:.0f} deg')
    page.screenshot(path='check_compass.png', full_page=True)

    # ---- learning centre ----
    page.goto(base + '/learning.html')
    ok = wait_for(page, "() => document.getElementById('statusText').textContent.startsWith('running in this tab')", 90)
    print('learning status:', page.evaluate("() => document.getElementById('statusText').textContent"), '| ok' if ok else '| FAILED')
    wait_for(page, "() => typeof brain !== 'undefined' && brain && brain.state.ready", 120)
    page.click('#odA'); time.sleep(3)
    print('learning frames:', page.evaluate("() => ({hz: document.getElementById('statHz').textContent, kc: document.getElementById('statKC').textContent, smelling: document.getElementById('smelling').textContent, brain: document.getElementById('brainStatus').textContent})"))
    page.click('#pairReward')
    time.sleep(14)
    print('after pairing:', page.evaluate("() => ({A: document.getElementById('valA').textContent, depressed: document.getElementById('statDepressed').textContent, pref: latest.pref})"))
    page.screenshot(path='check_learning.png', full_page=True)
    print('console:', '\n'.join(logs[-10:]) if logs else '(clean)')
    b.close()
