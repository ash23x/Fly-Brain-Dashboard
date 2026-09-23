// link.js -- how a page talks to its fly. Two ways, same messages:
//
//   1. the local Python server (server/sim_server.py): /neurons answers, frames
//      arrive over a WebSocket, commands go back as small JSON objects;
//   2. no server (the published site): the circuit runs in a Web Worker in this
//      tab (sim/worker.js), fed from data/<circuit>/, and nothing leaves the browser.
//
//   FlyLink.open('compass' | 'mushroom', {
//     onStatus(ok, text),           // connection / loading line
//     onMeta(meta, link),           // circuit metadata (the /neurons payload), before the first frame
//     onFrame(frame),               // one per frame, 30/s
//   }) -> Promise<link>;   link.send({...})   link.assets = base URL of the skeleton files
//                          link.other = {compass, learning}: where the sibling page lives
'use strict';
const FlyLink = (() => {
  const CIRCUIT_DIR = { compass: 'compass', mushroom: 'mushroom' };
  const PAGE = { compass: 'compass', learning: 'learning' };   // clean URLs on the published site

  async function probeServer() {
    try {
      const r = await fetch('/neurons', { cache: 'no-store' });
      if (!r.ok) return null;
      const meta = await r.json();
      return meta && typeof meta.mode === 'string' ? meta : null;
    } catch (e) { return null; }
  }

  function serverLink(mode, meta, h) {
    const link = { kind: 'server', assets: '/skeletons/', ws: null,
                   other: { compass: 'http://localhost:8765/', learning: 'http://localhost:8766/' },
                   send(obj) { if (link.ws && link.ws.readyState === 1) link.ws.send(JSON.stringify(obj)); } };
    const connect = (m) => {
      h.onMeta(m, link);
      const proto = location.protocol === 'https:' ? 'wss' : 'ws';
      const ws = new WebSocket(`${proto}://${location.host}/ws`);
      link.ws = ws;
      ws.onopen = () => h.onStatus(true, `connected — ${mode === 'compass' ? 'compass circuit' : 'mushroom body'} running on the local server`);
      ws.onmessage = (evt) => h.onFrame(JSON.parse(evt.data));
      ws.onclose = () => {
        link.ws = null;
        h.onStatus(false, 'disconnected — retrying…');
        const retry = async () => {
          const m2 = await probeServer();
          if (m2 && m2.mode === mode) connect(m2); else setTimeout(retry, 1500);
        };
        setTimeout(retry, 1500);
      };
    };
    connect(meta);
    return link;
  }

  function workerLink(mode, h) {
    const dir = CIRCUIT_DIR[mode];
    const base = new URL(`data/${dir}/`, location.href).href;
    const link = { kind: 'browser', assets: `data/${dir}/`, worker: null, other: PAGE,
                   send(obj) { if (link.worker) link.worker.postMessage({ type: 'cmd', cmd: obj }); } };
    return new Promise((resolve) => {
      let w;
      try { w = new Worker('sim/worker.js'); }
      catch (e) { h.onStatus(false, 'this browser cannot run the simulation (no Web Worker): ' + e.message); resolve(link); return; }
      link.worker = w;
      w.onmessage = (e) => {
        const msg = e.data || {};
        if (msg.type === 'progress') h.onStatus(false, msg.text);
        else if (msg.type === 'meta') {
          h.onMeta(msg.meta, link);
          h.onStatus(true, `running in this tab — ${msg.meta.n.toLocaleString()} neurons, ${msg.meta.n_edges.toLocaleString()} synaptic connections, nothing leaves your browser`);
          resolve(link);
        }
        else if (msg.type === 'frame') h.onFrame(msg.frame);
        else if (msg.type === 'error') h.onStatus(false, 'could not start the simulation: ' + msg.text);
      };
      w.onerror = (e) => h.onStatus(false, 'simulation worker error: ' + (e.message || e));
      h.onStatus(false, 'starting the fly…');
      w.postMessage({ type: 'init', mode, base });
    });
  }

  async function open(mode, h) {
    const meta = await probeServer();
    if (meta) {
      if (meta.mode !== mode) {
        h.onStatus(false, `the local server is running the ${meta.mode} circuit — start this one with: python server/sim_server.py --mode ${mode}`);
        await new Promise(r => setTimeout(r, 3000));
        return open(mode, h);
      }
      return serverLink(mode, meta, h);
    }
    return workerLink(mode, h);
  }

  return { open };
})();
