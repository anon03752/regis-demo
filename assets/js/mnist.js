/* The in-browser cycling-MNIST rule.
 *
 * The graph is one update of the trained rule, (state, z, fire) -> next_state,
 * so the rollout is driven from here. Two of those inputs are deliberate and
 * the demo is wrong without them:
 *
 *   z     a style code, constant along a trajectory by construction. Held for
 *         a whole run and redrawn only when a new run starts, or on request.
 *   fire  the per-cell Bernoulli firing mask, drawn here rather than inside
 *         the graph so the randomness does not depend on which onnxruntime
 *         build the page happens to load.
 *
 * The digit lives in the LAST channel at 22x22 inside a 32x32 grid, which is
 * what the training transform produces. Anything drawn has to land in the same
 * box -- outside it the rule is being asked about input it never saw -- so
 * strokes are clipped to INK_BOX from the manifest.
 */
(function () {
  'use strict';

  // onnxruntime-web 1.18.0, vendored. The page is meant to work from any
  // static host with nothing fetched off it, so the runtime lives here rather
  // than on a CDN: no third party gets a request, and it cannot break because
  // someone else's URL changed. ort.wasm.min.js is the wasm-only build (140 KB
  // against 520 KB for the full one); we never touch webgl or webgpu.
  var ORT_DIR = 'assets/vendor/onnxruntime/';
  var MODEL = 'assets/models/mnist.onnx';
  var MANIFEST = 'assets/models/mnist.json';
  var BANK = 'assets/models/mnist.bin';

  // Ink on a light ground, the same two colours the cycle figure above uses,
  // so the live panel and the drawn one read as the same object.
  var GROUND = [247, 247, 245];
  var INK = [33, 34, 40];

  var SPS_DEFAULT = 30;      // one update per frame at 30fps: one digit a second
  var BRUSH = 1.45;          // bins; a real digit's stroke is ~1.7 bins wide here

  var widget = document.getElementById('mnist-widget');
  if (!widget) return;

  var el = {
    gate: document.getElementById('m-gate'),
    load: document.getElementById('m-load'),
    status: document.getElementById('m-status'),
    canvas: document.getElementById('m-canvas'),
    guide: document.getElementById('m-guide'),
    cycles: document.getElementById('m-cycles'),
    sub: document.getElementById('m-sub'),
    ring: document.getElementById('m-ring'),
    digit: document.getElementById('m-digit'),
    sample: document.getElementById('m-sample'),
    draw: document.getElementById('m-draw'),
    play: document.getElementById('m-play'),
    step: document.getElementById('m-step'),
    style: document.getElementById('m-style'),
    sps: document.getElementById('m-sps'),
    spsOut: document.getElementById('m-sps-out')
  };

  var S = {
    playing: false, stepping: false, timer: null,
    steps: 0, origin: 'sample', sps: SPS_DEFAULT
  };

  function setStatus(t) { el.status.textContent = t; }

  function loadScript(src) {
    // The heart widget loads the same runtime; whichever gets there first wins
    // and the other reuses it rather than evaluating a second copy.
    if (window.ort) return Promise.resolve();
    return new Promise(function (res, rej) {
      var prior = document.querySelector('script[src="' + src + '"]');
      if (prior) { prior.addEventListener('load', res); prior.addEventListener('error', rej); return; }
      var s = document.createElement('script');
      s.src = src;
      s.onload = res;
      s.onerror = function () { rej(new Error('could not load ' + src)); };
      document.head.appendChild(s);
    });
  }

  function boot() {
    el.load.disabled = true;
    setStatus('fetching runtime…');
    return loadScript(ORT_DIR + 'ort.wasm.min.js').then(function () {
      // The runtime arrives as a data: URL built from assets/embed/wasm.js.
      // fetch() accepts data: URLs even on a file:// page -- which is exactly
      // what fetching the .wasm sitting beside it cannot do. wasmPaths takes
      // a per-file map; the non-SIMD name is deliberately left unmapped so it
      // still resolves to the file on disk on the rare browser without SIMD.
      // Single-threaded: threaded wasm needs cross-origin isolation headers
      // that a plain static host does not send.
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.simd = true;
      return EMBED.wasmUrl();
    }).then(function (wasmUrl) {
      ort.env.wasm.wasmPaths = { 'ort-wasm-simd.wasm': wasmUrl };
      setStatus('loading rule and digits…');
      return Promise.all([
        EMBED.json('mnistManifest'),
        EMBED.binary('mnistBank'),
        EMBED.binary('mnist').then(function (bytes) {
          EMBED.release('mnist');
          return ort.InferenceSession.create(bytes, {
            executionProviders: ['wasm'], graphOptimizationLevel: 'all'
          });
        })
      ]);
    }).then(function (parts) {
      var m = parts[0];
      S.man = m;
      S.C = m.shape[0]; S.H = m.shape[1]; S.W = m.shape[2];
      S.HW = S.H * S.W;
      S.vis = m.visible_index * S.HW;
      S.bank = parts[1];
      EMBED.release('mnistBank');
      S.session = parts[2];
      S.perCycle = m.steps_per_transition * m.cycle_n;
      buildDigitList();
      widget.classList.add('ready');
      placeGuide();
      wire();
      showSpeed();
      sampleDigit();           // open on a real digit, not an empty grid
    }).catch(function (e) {
      setStatus('failed: ' + e.message);
      el.load.disabled = false;
      el.load.hidden = false;
      el.gate.querySelector('p').textContent = 'The rule could not be loaded ('
        + e.message + '). The figure above shows the same cycle.';
    });
  }

  /* ---------------------------------------------------------------- state --- */

  function newZ() {
    // Training draws z from a standard normal, whose norm concentrates at
    // sqrt(z_dim); projecting exactly onto that sphere avoids the occasional
    // freak draw giving handwriting the rule never had to handle.
    var n = S.man.z_dim, z = new Float32Array(n), s = 0, i;
    for (i = 0; i < n; i++) {
      var u = Math.max(Math.random(), 1e-9), v = Math.random();
      z[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
      s += z[i] * z[i];
    }
    s = Math.sqrt(n) / Math.sqrt(s || 1);
    for (i = 0; i < n; i++) z[i] *= s;
    S.z = z;
  }

  /* A run starts from a blank state with the digit written into the visible
   * channel and every hidden channel at zero -- the seeding the rule was
   * trained from. The cycle counter counts updates since this point, so
   * anything that sets a new starting state resets it. */
  function newRun(origin, img) {
    pause();
    strokeEnd();
    S.state = new Float32Array(S.C * S.HW);
    if (img) S.state.set(img, S.vis);
    S.steps = 0;
    S.origin = origin;
    newZ();
    render();
    readout();
  }

  function digitImage(idx) {
    var img = new Float32Array(S.HW);
    for (var i = 0; i < S.HW; i++) img[i] = S.bank[idx * S.HW + i] / 255;
    return img;
  }

  function sampleDigit() {
    var want = el.digit.value, pool = [], i;
    for (i = 0; i < S.man.n_digits; i++) {
      if (want === 'any' || S.man.labels[i] === +want) pool.push(i);
    }
    // Avoid handing back the digit already on screen when there is a choice.
    var pick = pool[(Math.random() * pool.length) | 0];
    if (pool.length > 1 && pick === S.lastPick) pick = pool[(pool.indexOf(pick) + 1) % pool.length];
    S.lastPick = pick;
    newRun('sample', digitImage(pick));
    setStatus('seeded with a real ' + S.man.labels[pick]);
  }

  function blankForDrawing() {
    newRun('drawn', null);
    setStatus('draw a digit inside the box');
  }

  /* ------------------------------------------------------------- drawing --- */

  function placeGuide() {
    // The guide is the 22x22 box the training transform resizes a digit into,
    // drawn in page pixels over the canvas so it tracks any canvas size.
    var b = S.man.ink_box;
    el.guide.style.left = (100 * b[1] / S.W) + '%';
    el.guide.style.top = (100 * b[0] / S.H) + '%';
    el.guide.style.width = (100 * b[3] / S.W) + '%';
    el.guide.style.height = (100 * b[2] / S.H) + '%';
  }

  function binAt(ev) {
    var r = el.canvas.getBoundingClientRect();
    return [(ev.clientY - r.top) / r.height * S.H,
            (ev.clientX - r.left) / r.width * S.W];
  }

  function stamp(py, px) {
    var b = S.man.ink_box, changed = false;
    var y0 = b[0], x0 = b[1], y1 = b[0] + b[2], x1 = b[1] + b[3];
    for (var r = Math.floor(py - BRUSH); r <= Math.ceil(py + BRUSH); r++) {
      for (var c = Math.floor(px - BRUSH); c <= Math.ceil(px + BRUSH); c++) {
        if (r < y0 || r >= y1 || c < x0 || c >= x1) continue;
        var d = Math.hypot(r + 0.5 - py, c + 0.5 - px);
        if (d > BRUSH) continue;
        // Soft edge: real digits are antialiased by the resize, and a hard
        // 0/1 brush gives the rule a stroke sharper than anything in training.
        var v = Math.min(1, 1.25 * (1 - d / BRUSH));
        var i = S.vis + r * S.W + c;
        if (v > S.state[i]) { S.state[i] = v; changed = true; }
      }
    }
    return changed;
  }

  /* Stamp along the segment since the last sample, not just at the current
   * point: pointermove fires far less often than the pointer moves, so a
   * quick stroke on a 32-wide grid otherwise lands as a row of dots. */
  function paint(ev) {
    var p = binAt(ev), changed = false;
    var a = S.lastPt || p;
    var n = Math.max(1, Math.ceil(Math.hypot(p[0] - a[0], p[1] - a[1]) / (BRUSH * 0.5)));
    for (var k = 1; k <= n; k++) {
      var t = k / n;
      if (stamp(a[0] + (p[0] - a[0]) * t, a[1] + (p[1] - a[1]) * t)) changed = true;
    }
    S.lastPt = p;
    return changed;
  }

  function strokeStart(ev) {
    var p = binAt(ev), b = S.man.ink_box;
    if (p[0] < b[0] || p[0] >= b[0] + b[2] || p[1] < b[1] || p[1] >= b[1] + b[3]) return;
    el.canvas.setPointerCapture(ev.pointerId);
    S.drawing = true;
    S.lastPt = null;
    // Keep the visible image, but wipe every hidden channel across the grid.
    // A fresh buffer also invalidates any model update already in flight.
    var state = new Float32Array(S.C * S.HW);
    state.set(S.state.subarray(S.vis, S.vis + S.HW), S.vis);
    S.state = state;
    S.steps = 0;
    S.origin = 'drawn';
    paint(ev);
    render();
    readout();
  }

  function strokeMove(ev) {
    if (!S.drawing) return;
    if (paint(ev)) { render(); }
  }

  function strokeEnd() { S.drawing = false; S.lastPt = null; }

  /* ----------------------------------------------------------------- run --- */

  function runStep() {
    // Hold the rule while drawing so hidden state stays zero for the stroke.
    if (S.stepping || S.drawing || !S.session) return Promise.resolve();
    S.stepping = true;
    var state = S.state;
    var fire = new Float32Array(S.HW);
    for (var i = 0; i < S.HW; i++) fire[i] = Math.random() < S.man.fire_rate ? 1 : 0;
    var feeds = {
      state: new ort.Tensor('float32', state, [1, S.C, S.H, S.W]),
      z: new ort.Tensor('float32', S.z, [1, S.man.z_dim]),
      fire: new ort.Tensor('float32', fire, [1, 1, S.H, S.W])
    };
    return S.session.run(feeds).then(function (out) {
      // Drawing or reseeding supersedes results computed from the old state.
      if (S.state !== state) return;
      state = new Float32Array(out.next_state.data);
      S.state = state;
      S.steps += 1;
      render();
      readout();
    }).catch(function (e) {
      if (S.state !== state) return;
      setStatus('step failed: ' + e.message);
      pause();
    }).then(function () { S.stepping = false; });
  }

  /* A timer, not requestAnimationFrame: rAF caps the rollout at the display
   * refresh rate and stops it outright in a throttled tab. Each step schedules
   * the next relative to its own start, so the rate holds without drifting. */
  function pump() {
    if (!S.playing) return;
    var started = performance.now();
    runStep().then(function () {
      if (!S.playing) return;
      S.timer = setTimeout(pump, Math.max(0, 1000 / S.sps - (performance.now() - started)));
    });
  }

  function play() {
    if (S.playing) return;
    S.playing = true;
    el.play.innerHTML = '&#10073;&#10073;&ensp;Pause';
    pump();
  }

  function pause() {
    S.playing = false;
    if (S.timer) { clearTimeout(S.timer); S.timer = null; }
    el.play.innerHTML = '&#9654;&ensp;Play';
  }

  /* ----------------------------------------------------------- rendering --- */

  function render() {
    var cv = el.canvas, g = cv.getContext('2d');
    var im = g.createImageData(S.W, S.H), d = im.data;
    for (var i = 0; i < S.HW; i++) {
      var v = Math.min(1, Math.max(0, S.state[S.vis + i])), j = i * 4;
      d[j] = GROUND[0] + (INK[0] - GROUND[0]) * v;
      d[j + 1] = GROUND[1] + (INK[1] - GROUND[1]) * v;
      d[j + 2] = GROUND[2] + (INK[2] - GROUND[2]) * v;
      d[j + 3] = 255;
    }
    g.putImageData(im, 0, 0);
  }

  // Tabular digits alone still shift the following text at 9 -> 10, etc.
  // Reserve number columns and the plural suffix to keep each label steady.
  function clockNumber(value, digits) {
    return '<span class="clock-number" style="min-width:' + digits + 'ch">' + value + '</span>';
  }

  function clockCount(value, label, digits) {
    return '<span class="clock-metric">' + clockNumber(value, digits) + ' ' + label
      + (value === 1 ? '<span class="clock-padding" aria-hidden="true">s</span>' : 's') + '</span>';
  }

  function readout() {
    var per = S.perCycle, spt = S.man.steps_per_transition;
    var cycles = Math.floor(S.steps / per);
    var within = S.steps % per;
    el.cycles.innerHTML = clockCount(cycles, 'full cycle', 3);
    if (S.steps === 0) {
      el.sub.textContent = S.origin === 'drawn'
        ? 'your digit, nothing run yet' : 'a real test digit, nothing run yet';
    } else {
      el.sub.innerHTML = clockCount(Math.floor(S.steps / spt), 'digit transition', 5)
        + ' · ' + clockCount(S.steps, 'update', 6)
        + ' · <span class="clock-metric">' + clockNumber(within, String(per - 1).length)
        + '/' + per + ' through the current cycle</span>';
    }
    // Ten pips, one per digit position, filling as the cycle progresses.
    var done = Math.floor(within / spt);
    for (var i = 0; i < el.ring.children.length; i++) {
      el.ring.children[i].className = 'pip' + (i < done ? ' on' : (i === done ? ' at' : ''));
    }
  }

  /* --------------------------------------------------------------- wiring --- */

  function buildDigitList() {
    var o = ['<option value="any" selected>Random digit</option>'];
    for (var d = 0; d < S.man.cycle_n; d++) o.push('<option value="' + d + '">' + d + '</option>');
    el.digit.innerHTML = o.join('');
    var pips = [];
    for (var i = 0; i < S.man.cycle_n; i++) pips.push('<span class="pip"></span>');
    el.ring.innerHTML = pips.join('');
  }

  function showSpeed() {
    // Seconds per full turn is the number that means something here, so give
    // it alongside the rate rather than making the reader divide by 300.
    el.spsOut.textContent = S.sps + ' updates/s \u00b7 '
      + (Math.round(S.perCycle / S.sps * 10) / 10) + ' s per turn';
  }

  function wire() {
    el.sps.addEventListener('input', function () {
      S.sps = Math.max(1, Number(el.sps.value) || SPS_DEFAULT);
      showSpeed();
    });
    el.play.addEventListener('click', function () { S.playing ? pause() : play(); });
    el.step.addEventListener('click', function () { pause(); runStep(); });
    el.sample.addEventListener('click', sampleDigit);
    el.digit.addEventListener('change', sampleDigit);
    el.draw.addEventListener('click', blankForDrawing);
    el.style.addEventListener('click', function () {
      newZ();
      setStatus('new style vector, same state');
    });
    el.canvas.addEventListener('pointerdown', strokeStart);
    el.canvas.addEventListener('pointermove', strokeMove);
    el.canvas.addEventListener('pointerup', strokeEnd);
    el.canvas.addEventListener('pointercancel', strokeEnd);
    el.canvas.addEventListener('lostpointercapture', strokeEnd);
  }

  // Preloaded rather than gated behind a click. The button stays as a
  // retry path when loading fails; it is just not the only way in.
  el.load.addEventListener('click', boot);
  if (el.gate) {
    var gp = el.gate.querySelector('p');
    if (gp) gp.textContent = 'Loading the learned rule and a bank of real test digits\u2026';
    el.load.hidden = true;
  }
  boot();
}());
