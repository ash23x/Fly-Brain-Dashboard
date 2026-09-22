// brain3d.js -- the circuit's real 3D skeletons, drawn with raw WebGL and lit by
// the live simulation. No libraries. One draw call: every neuron is a bundle of
// line segments carrying its neuron index; two 1-D textures hold each neuron's
// base colour and its current glow, and the glow texture is rewritten every frame
// from the spikes the server sends.
//
//   const brain = Brain3D.attach(canvasElement, colourForNeuron /* (meta, i) -> [r,g,b] 0..1 */);
//   brain.load(neuronsList);          // fetches /skeletons/* from the same server
//   brain.spike(listOfNeuronIndices); // call on every WebSocket frame
'use strict';
const Brain3D = (() => {
  const VS = `
    attribute vec3 a_pos; attribute float a_idx;
    uniform mat4 u_mvp; uniform float u_n;
    varying float v_u;
    void main() { gl_Position = u_mvp * vec4(a_pos, 1.0); v_u = (a_idx + 0.5) / u_n; }`;
  const FS = `
    precision mediump float;
    varying float v_u; uniform sampler2D u_col; uniform sampler2D u_glow; uniform float u_base; uniform float u_bright;
    void main() {
      vec4 bc = texture2D(u_col, vec2(v_u, 0.5));
      vec3 base = bc.rgb; float rest = bc.a;      // alpha channel = how visible the cell is when quiet
      float g = texture2D(u_glow, vec2(v_u, 0.5)).r;
      // "over" compositing with premultiplied colour: dense bundles converge on the
      // cell's colour instead of burning out to white; a firing cell goes bright.
      // u_bright is the user's dimmer: it scales everything, the white flare most of all.
      vec3 c = (base * (u_base * rest + 0.7 * g) + vec3(1.0, 0.98, 0.9) * g * 0.6 * u_bright) * u_bright;
      float a = (0.15 * rest + 0.65 * g) * (0.4 + 0.6 * u_bright);
      gl_FragColor = vec4(c * a, a);
    }`;

  // ---- tiny matrix helpers (column-major, like WebGL wants) ----
  function perspective(fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
    return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
  }
  function lookAt(eye, target, up) {
    const z = norm(sub(eye, target)), x = norm(cross(up, z)), y = cross(z, x);
    return new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0,
      -dot(x, eye), -dot(y, eye), -dot(z, eye), 1]);
  }
  function mul(a, b) {
    const o = new Float32Array(16);
    for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
      let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      o[c * 4 + r] = s;
    }
    return o;
  }
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

  function compile(gl, type, src) {
    const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  }

  function attach(canvas, colourFor) {
    const gl = canvas.getContext('webgl', { antialias: true, alpha: false, premultipliedAlpha: false });
    if (!gl) { canvas.replaceWith(Object.assign(document.createElement('div'), { textContent: 'WebGL is not available in this browser.', className: 'hint' })); return null; }
    const prog = gl.createProgram();
    gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VS));
    gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
    gl.useProgram(prog);
    const loc = {
      pos: gl.getAttribLocation(prog, 'a_pos'), idx: gl.getAttribLocation(prog, 'a_idx'),
      mvp: gl.getUniformLocation(prog, 'u_mvp'), n: gl.getUniformLocation(prog, 'u_n'),
      col: gl.getUniformLocation(prog, 'u_col'), glow: gl.getUniformLocation(prog, 'u_glow'),
      base: gl.getUniformLocation(prog, 'u_base'), bright: gl.getUniformLocation(prog, 'u_bright'),
    };
    const state = {
      gl, prog, loc, n: 0, count: 0, glow: null, glowTex: null, colTex: null, bounds: null,
      theta: 0.6, phi: 0.35, dist: 1.25, target: [0, 0, 0], dragging: false, lastX: 0, lastY: 0,
      lastInteract: 0, spikesPending: new Set(), ready: false, base: 0.4, bright: 0.5,
    };

    // ---- orbit controls: drag to turn, wheel to zoom, idle = slow auto-rotate ----
    canvas.addEventListener('pointerdown', (e) => { state.dragging = true; state.lastX = e.clientX; state.lastY = e.clientY; state.lastInteract = performance.now(); canvas.setPointerCapture(e.pointerId); });
    canvas.addEventListener('pointerup', () => { state.dragging = false; });
    canvas.addEventListener('pointercancel', () => { state.dragging = false; });
    canvas.addEventListener('pointermove', (e) => {
      if (!state.dragging) return;
      state.theta -= (e.clientX - state.lastX) * 0.008;
      state.phi = Math.max(-1.4, Math.min(1.4, state.phi + (e.clientY - state.lastY) * 0.008));
      state.lastX = e.clientX; state.lastY = e.clientY; state.lastInteract = performance.now();
    });
    canvas.addEventListener('wheel', (e) => { e.preventDefault(); state.dist *= Math.exp(e.deltaY * 0.001); state.dist = Math.max(0.3, Math.min(4, state.dist)); state.lastInteract = performance.now(); }, { passive: false });

    async function load(neurons) {
      const [meta, posBuf, idxBuf] = await Promise.all([
        fetch('/skeletons/skeletons.json').then(r => { if (!r.ok) throw new Error('no skeletons'); return r.json(); }),
        fetch('/skeletons/skeleton_pos.bin').then(r => r.arrayBuffer()),
        fetch('/skeletons/skeleton_idx.bin').then(r => r.arrayBuffer()),
      ]);
      const pos = new Float32Array(posBuf);
      const idx16 = new Uint16Array(idxBuf);
      const idx = new Float32Array(idx16.length);
      for (let i = 0; i < idx16.length; i++) idx[i] = idx16[i];
      state.n = meta.n_neurons; state.count = meta.n_vertices; state.bounds = meta.bounds;
      const ext = Math.max(meta.bounds.max[0] - meta.bounds.min[0], meta.bounds.max[1] - meta.bounds.min[1], meta.bounds.max[2] - meta.bounds.min[2]);
      state.radius = ext * 1.1;

      const posB = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, posB); gl.bufferData(gl.ARRAY_BUFFER, pos, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(loc.pos); gl.vertexAttribPointer(loc.pos, 3, gl.FLOAT, false, 0, 0);
      const idxB = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, idxB); gl.bufferData(gl.ARRAY_BUFFER, idx, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(loc.idx); gl.vertexAttribPointer(loc.idx, 1, gl.FLOAT, false, 0, 0);

      // base colour per neuron, as a 1-D RGB texture
      const col = new Uint8Array(state.n * 4);
      neurons.forEach((m, i) => { const c = colourFor(m, i); col[i * 4] = c[0] * 255; col[i * 4 + 1] = c[1] * 255; col[i * 4 + 2] = c[2] * 255; col[i * 4 + 3] = (c.length > 3 ? c[3] : 1) * 255; });
      state.colTex = makeTex(gl, state.n, col, gl.RGBA);
      state.glow = new Uint8Array(state.n);
      state.glowTex = makeTex(gl, state.n, state.glow, gl.LUMINANCE);
      gl.uniform1f(loc.n, state.n);
      gl.uniform1i(loc.col, 0); gl.uniform1i(loc.glow, 1);
      gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA); gl.disable(gl.DEPTH_TEST);
      state.ready = true;
      requestAnimationFrame(frame);
      return meta;
    }

    function makeTex(gl, n, data, fmt) {
      const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, fmt, n, 1, 0, fmt, gl.UNSIGNED_BYTE, data);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      return t;
    }

    function spike(ids) { for (const i of ids) state.spikesPending.add(i); }

    // Optional dimmer: pass a <input type="range" min="0" max="100">. Remembered per browser.
    function bindBrightness(slider) {
      let saved = null;
      try { saved = localStorage.getItem('brain3d.bright'); } catch (e) {}
      if (saved !== null && !isNaN(parseFloat(saved))) state.bright = parseFloat(saved);
      slider.value = Math.round(state.bright * 100);
      slider.addEventListener('input', () => {
        state.bright = slider.value / 100;
        try { localStorage.setItem('brain3d.bright', String(state.bright)); } catch (e) {}
      });
    }

    let lastT = performance.now();
    function frame(now) {
      if (!state.ready) return;
      const dt = Math.min(0.1, (now - lastT) / 1000); lastT = now;
      if (!state.dragging && now - state.lastInteract > 2500) state.theta += 0.12 * dt;
      // glow: decay, then add a flicker for each cell that fired since the last frame.
      // A single spike is a flicker; only a cell firing steadily (the bump, a dopamine
      // neuron under sugar) climbs to full brightness -- rate coding, made visible.
      const g = state.glow;
      for (let i = 0; i < g.length; i++) g[i] = g[i] * 0.84;
      for (const i of state.spikesPending) if (i < g.length) g[i] = Math.min(255, g[i] + 120);
      state.spikesPending.clear();
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, state.glowTex);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, state.n, 1, gl.LUMINANCE, gl.UNSIGNED_BYTE, g);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, state.colTex);

      const dpr = window.devicePixelRatio || 1;
      const w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
      if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
      gl.viewport(0, 0, w, h);
      gl.clearColor(0.02, 0.027, 0.047, 1); gl.clear(gl.COLOR_BUFFER_BIT);
      const r = state.radius * state.dist;
      const eye = [r * Math.cos(state.phi) * Math.sin(state.theta), r * Math.sin(state.phi), r * Math.cos(state.phi) * Math.cos(state.theta)];
      const mvp = mul(perspective(0.9, w / h, r * 0.05, r * 10), lookAt(eye, state.target, [0, 1, 0]));
      gl.uniformMatrix4fv(loc.mvp, false, mvp);
      gl.uniform1f(loc.base, state.base);
      gl.uniform1f(loc.bright, state.bright);
      gl.drawArrays(gl.LINES, 0, state.count);
      requestAnimationFrame(frame);
    }

    return { load, spike, bindBrightness, state };
  }
  return { attach };
})();
