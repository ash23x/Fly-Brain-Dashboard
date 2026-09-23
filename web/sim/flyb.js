// flyb.js -- reads the circuit.bin container written by scripts/08_export_web.py:
//   "FLYB" | uint32 header length | JSON header | arrays (4-byte aligned)
// Returns { constants, arrays } with each array as a typed-array view on the buffer.
'use strict';
(function (root) {
  const TYPES = { float32: Float32Array, int32: Int32Array, uint16: Uint16Array, int16: Int16Array, uint8: Uint8Array };
  function parseFlyb(buffer) {
    const u8 = new Uint8Array(buffer);
    if (String.fromCharCode(u8[0], u8[1], u8[2], u8[3]) !== 'FLYB') throw new Error('not a FLYB container');
    const hlen = new DataView(buffer).getUint32(4, true);
    const header = JSON.parse(new TextDecoder().decode(u8.subarray(8, 8 + hlen)));
    const base = 8 + hlen;
    const arrays = {};
    for (const [name, a] of Object.entries(header.arrays)) {
      const T = TYPES[a.dtype];
      if (!T) throw new Error(`${name}: unknown dtype ${a.dtype}`);
      arrays[name] = new T(buffer, base + a.offset, a.length);
      arrays[name].shape = a.shape;
    }
    return { constants: header.constants, arrays };
  }
  // Small seeded PRNG (mulberry32) with a Box-Muller gaussian: the fly must not be
  // deterministic across visitors' tabs, but a tab should be reproducible if asked.
  function makeRng(seed) {
    let a = (seed >>> 0) || 0x9e3779b9;
    const uniform = () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    let spare = null;
    const normal = () => {
      if (spare !== null) { const v = spare; spare = null; return v; }
      let u, v, s;
      do { u = uniform() * 2 - 1; v = uniform() * 2 - 1; s = u * u + v * v; } while (s >= 1 || s === 0);
      const m = Math.sqrt(-2 * Math.log(s) / s);
      spare = v * m; return u * m;
    };
    return { uniform, normal };
  }
  root.Flyb = { parse: parseFlyb, rng: makeRng };
})(typeof self !== 'undefined' ? self : globalThis);
if (typeof module !== 'undefined') module.exports = globalThis.Flyb;
