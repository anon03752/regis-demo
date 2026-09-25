/* The in-browser NCA.
 *
 * One ONNX forward pass per animation frame: the graph wraps a single call of
 * the trained rule, `state [1,C,H,W] -> next_state`, so the rollout is driven
 * from here rather than baked in. Ported from the research viewer; the wound
 * semantics and the step-to-hours clock below mirror how the model was
 * trained, and getting either wrong produces plausible-looking nonsense, so
 * the reasoning is kept in the comments.
 */
(function () {
  'use strict';

  // onnxruntime-web 1.18.0, vendored. The page is meant to work from any
  // static host with nothing fetched off it, so the runtime lives here rather
  // than on a CDN: no third party gets a request, and it cannot break because
  // someone else's URL changed. ort.wasm.min.js is the wasm-only build (140 KB
  // against 520 KB for the full one); we never touch webgl or webgpu.
  var ORT_DIR = 'assets/vendor/onnxruntime/';
  var MODEL = 'assets/models/regis.onnx';
  var MANIFEST = 'assets/models/seeds.json';
  var SEEDS = 'assets/models/seeds.bin';
  var DISPLAY = 'assets/models/display.json';   // the shared display rules

  /* The a-priori grouping used for the in-silico screen in the paper. Group A
   * (regeneration programme) and B (regulatory stroma) are the implicated
   * side of the contrast, C (blood) and D (structural) the unimplicated one.
   * Indices are channels of the visible state. "Others" is ungrouped and not
   * part of the screen. */
  /* The a-priori grouping used for the in-silico screen in the paper. Groups A
   * and B are the implicated side of the contrast, C and D the unimplicated
   * one. The paper's screen covers nineteen cell types; "Others" is not one of
   * them and is left out here too. Indices are channels of the visible state.
   * The two implicated groups are the ones tracked in the abundance plot. */
  var GROUPS = [
    { key: 'A', name: 'Regeneration programme', tag: 'implicated',
      ids: [0, 1, 2, 3, 4], colour: '#e4561e', ymax: 0.20,
      note: 'border-zone cardiomyocytes, in cascade order' },
    { key: 'B', name: 'Regulatory stroma', tag: 'implicated',
      ids: [10, 11, 12, 13], colour: '#430e8d', ymax: 0.60,
      note: 'macrophages, pro-regenerative fibroblasts, epicardium' },
    { key: 'C', name: 'Blood', tag: 'not implicated', ids: [16, 17, 18],
      colour: '#f09498', note: 'erythrocytes' },
    { key: 'D', name: 'Structural', tag: 'not implicated',
      ids: [5, 6, 7, 8, 9, 14, 15], colour: '#78b7d6',
      note: 'remote-zone and other myocardium, smooth muscle, valves' }
  ];
  var TRACKED = ['A', 'B'];              // the two with an abundance line
  // The axes are FIXED, not fitted to whatever the run has reached so far: a
  // running maximum makes the lines slide downwards every time a new peak
  // arrives, so the same shape means different things at different moments.
  // The two ymax values above sit just above the largest peak measured over
  // twelve full cycles across six hearts with the shipped checkpoint (v57
  // seed 0): the cascade states reach 0.047-0.186 mean density per live bin
  // (median 0.084) and the stroma 0.212-0.560 (median 0.338), so nothing
  // clips. That spread is also why each group gets its own axis -- sharing one
  // would flatten the cascade into the baseline. Re-measure if the checkpoint
  // changes, though v59 lands in the same range (0.071-0.187 / 0.245-0.621).
  var CYCLE_STEPS = 120;                 // 7 legs x 15, plus the closing hold

  var FULL_NAME = {
    'BZm (activ.)': 'Border-zone myocardium, activated',
    'BZm (dediff.)': 'Border-zone myocardium, dedifferentiating',
    'BZm (dediff.-prolif.)': 'Border-zone myocardium, dedifferentiating and proliferating',
    'BZm (prolif.)': 'Border-zone myocardium, proliferating',
    'BZm (re-diff.)': 'Border-zone myocardium, re-differentiating',
    RZm1: 'Remote-zone myocardium 1', RZm2: 'Remote-zone myocardium 2',
    'Vm (comp.)': 'Ventricular compact myocardium', Am: 'Atrial myocardium',
    VAm: 'Ventricular and atrial shared myocardium', MC: 'Macrophages',
    'FB (reg.)': 'Fibroblasts, pro-regenerative',
    MFE: 'Mixed immune, fibroblast and endocardial', Epi: 'Epicardium',
    SMC: 'Smooth muscle, bulbus arteriosus', Valves: 'Cardiac valves',
    RBC: 'Erythrocytes, wound', 'RBC (V)': 'Erythrocytes, ventricular',
    'RBC (A)': 'Erythrocytes, atrial', Others: 'Other cell types',
    Damage: 'Damaged tissue'
  };

  // The rule's own alive threshold: below this a bin is outside the tissue.
  var VOID = 0.05;
  var BG = [250, 250, 249];      // the panel
  var SCALE = 8;                 // canvas pixels per bin
  // Display rules, loaded from display.json so this and the header video
  // cannot drift apart. Defaults only matter if that fetch fails.
  var D = {
    bzm_channels: [0, 1, 2, 3, 4], bzm_thresholds: null,
    capacity: 4.0, empty_below: 0.08, alpha_floor: 0.28,
    min_density: 0.5, dot: 0.42
  };

  var S = {
    session: null, hasDt: false,
    manifest: null, seeds: null, seedIdx: 0,
    H: 0, W: 0, C: 0, nv: 0, aliveIdx: 0, dmgIdx: 0,
    current: null, painted: null,
    injured: false, hours: 0, leg: 0, stepInLeg: 0, steps: 0,
    playing: false, stepping: false, painting: false,
    brush: 3, sps: 20, timer: 0, ctx: null, plot: {}, trace: []
  };

  var el = {};
  ['status', 'gate', 'load', 'canvas', 'now', 'sub', 'tps', 'seed',
   'play', 'step', 'injure', 'reset', 'stop', 'legend',
   'plot-a', 'plot-b'].forEach(function (k) {
    el[k] = document.getElementById('d-' + k);
  });
  var widget = document.getElementById('demo-widget');
  if (!widget || !el.load) return;

  function setStatus(t) { if (el.status) el.status.textContent = t; }

  /* ------------------------------------------------------------- loading --- */

  function loadScript(src) {
    // The MNIST widget loads the same runtime; whichever gets there first
    // wins and the other reuses it rather than evaluating a second copy.
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
      setStatus('loading model and hearts…');
      return Promise.all([
        EMBED.json('seedsManifest'),
        EMBED.binary('seeds'),
        EMBED.json('display').catch(function () { return null; })
      ]);
    }).then(function (parts) {
      if (parts[2]) Object.assign(D, parts[2]);
      S.manifest = parts[0];
      var m = S.manifest;
      S.C = m.shape[0]; S.H = m.shape[1]; S.W = m.shape[2];
      S.nv = m.n_visible; S.aliveIdx = m.alive_idx; S.dmgIdx = m.damage_channel_idx;
      S.seeds = new Float32Array(parts[1].buffer, parts[1].byteOffset,
                                 parts[1].byteLength / 4);
      EMBED.release('seeds');   // 5 MB of base64, decoded and done with
      return loadModel();
    }).then(function () {
      buildSeedList();
      buildLegend();
      initCanvas();
      resetHeart();
      widget.classList.add('ready');
      // after the widget is shown: a hidden element has no clientWidth, so
      // sizing the plot before this point gives a zero-width canvas
      initPlots();
      drawTrace();
      wireRunControls();
      setStatus('ready');
    }).catch(function (e) {
      // fetch() is refused for local files, so opening index.html straight
      // from disk fails right here and nowhere else obvious. Say that,
      // rather than leaving the reader with a bare "Failed to fetch".
      var local = location.protocol === 'file:';
      setStatus(local ? 'needs a web server' : 'failed: ' + e.message);
      el.load.disabled = false;
      el.load.hidden = false;
      el.gate.querySelector('p').textContent = local
        ? 'A browser will not read a model out of a file:// page. Serve the '
          + 'folder \u2014 python3 -m http.server \u2014 and reload. The figures above show the same experiments.'
        : 'The model could not be loaded (' + e.message + '). The figures above show the same experiments.';
    });
  }

  function loadModel() {
    setStatus('loading rule…');
    return EMBED.binary('regis').then(function (bytes) {
      EMBED.release('regis');
      return ort.InferenceSession.create(bytes, {
        executionProviders: ['wasm'], graphOptimizationLevel: 'all'
      });
    }).then(function (sess) {
      S.session = sess;
      S.hasDt = sess.inputNames.indexOf('dt') >= 0;
      setStatus('ready');
    });
  }

  /* --------------------------------------------------------------- state --- */

  function seedView(i) {
    var n = S.C * S.H * S.W;
    return S.seeds.subarray(i * n, (i + 1) * n);
  }

  function resetHeart() {
    S.current = Float32Array.from(seedView(S.seedIdx));
    S.painted = new Uint8Array(S.H * S.W);
    S.injured = false; S.hours = 0; S.leg = 0; S.stepInLeg = 0; S.steps = 0;
    S.held = 0; S.stopped = false; S.idle = 0;
    S.trace = [];
    renderClock();
    render();
  }

  /* ------------------------------------------------------------- wounding ---
   * Mirrors the wound the model was trained on. Damage is one of the
   * composition channels, so it tops out at the bin's own capacity rather
   * than at a universal 1.0 -- on this 2x2 sum-pooled grid that is ~4.0, and
   * stamping 1.0 would be a wound four times weaker than anything in
   * training. The alive channel is left alone when the model was trained with
   * a live wound: zeroing it would shut the 3x3 gate and freeze the wound
   * interior forever, because under those training semantics alive is never 0
   * inside the heart. Hidden channels are never touched. */
  function woundBin(px) {
    var HW = S.H * S.W;
    if (S.current[S.aliveIdx * HW + px] < VOID) return false;  // outside tissue
    var cap = 1.0, ch;
    if (S.manifest.conserve_capacity_wound) {
      cap = 0;
      for (ch = 0; ch < S.nv; ch++) cap += S.current[ch * HW + px];
    }
    for (ch = 0; ch < S.nv; ch++) {
      S.current[ch * HW + px] = (ch === S.dmgIdx) ? cap : 0;
    }
    if (!S.manifest.wound_keep_alive) S.current[S.aliveIdx * HW + px] = 0;
    S.painted[px] = 1;
    return true;
  }

  // The first damaged bin is the injury event: the clock starts at 0 hpa.
  // Later paints on an already-injured heart do not restart it, so re-wounding
  // a regenerated heart keeps accumulating time rather than pretending it is a
  // fresh animal.
  function startClockIfNeeded(any) {
    if (any && !S.injured) {
      S.injured = true; S.hours = 0; S.leg = 0; S.stepInLeg = 0; S.held = 0;
      S.stopped = false; S.idle = 0;
      S.trace = [];                 // a new episode starts a new profile
    }
  }

  function paintAt(r, c) {
    var rad = S.brush - 1, r2 = rad * rad, any = false;
    for (var dr = -rad; dr <= rad; dr++) {
      for (var dc = -rad; dc <= rad; dc++) {
        if (dr * dr + dc * dc > r2) continue;
        var rr = r + dr, cc = c + dc;
        if (rr < 0 || cc < 0 || rr >= S.H || cc >= S.W) continue;
        if (woundBin(rr * S.W + cc)) any = true;
      }
    }
    startClockIfNeeded(any);
    renderClock();
    render();
  }

  /* The injury the rule was trained on: a real hand-painted wound footprint
   * from an injured section, flipped and rotated and dropped on this heart's
   * apex. Deliberately not the standardised assay disc, which is a good deal
   * larger -- a footprint runs to a median of 39 bins on this grid where the
   * disc covers nearer 80, and the apex zone it lands inside runs to 170. */
  function shapedWound() {
    var m = S.manifest, sid = m.seed_ids[S.seedIdx];
    var zone = m.zones[sid], shapes = m.wound_shapes;
    if (!zone || !shapes || !shapes.length) return null;
    var HW = S.H * S.W, aOff = S.aliveIdx * HW;
    var inZone = new Uint8Array(HW), allowed = [];
    for (var i = 0; i < zone.length; i++) {
      inZone[zone[i]] = 1;
      if (S.current[aOff + zone[i]] >= VOID) allowed.push(zone[i]);
    }
    if (!allowed.length) return null;

    var best = null, bestScore = -1;
    for (var tries = 0; tries < 40; tries++) {
      var sp = shapes[(Math.random() * shapes.length) | 0];
      var cells = [];
      for (i = 0; i < sp.px.length; i++) {
        cells.push([(sp.px[i] / sp.w) | 0, sp.px[i] % sp.w]);
      }
      // the annotations have no canonical orientation, so flip and rotate
      var k = (Math.random() * 4) | 0, fr = Math.random() < 0.5,
          fc = Math.random() < 0.5, h = sp.h, w = sp.w;
      cells = cells.map(function (rc) {
        var r = rc[0], c = rc[1], t;
        for (var q = 0; q < k; q++) { t = r; r = c; c = h - 1 - t; t = h; h = w; w = t; }
        return [fr ? h - 1 - r : r, fc ? w - 1 - c : c];
      });
      var anchor = allowed[(Math.random() * allowed.length) | 0];
      var r0 = ((anchor / S.W) | 0) - ((h / 2) | 0);
      var c0 = (anchor % S.W) - ((w / 2) | 0);
      r0 = Math.min(Math.max(r0, 0), S.H - h);
      c0 = Math.min(Math.max(c0, 0), S.W - w);
      var hit = [], inside = 0;
      for (i = 0; i < cells.length; i++) {
        var px = (r0 + cells[i][0]) * S.W + (c0 + cells[i][1]);
        if (px < 0 || px >= HW) continue;
        if (S.current[aOff + px] >= VOID) { hit.push(px); if (inZone[px]) inside++; }
      }
      if (!hit.length) continue;
      var score = inside / hit.length;
      if (score > bestScore) { bestScore = score; best = hit; }
      if (score > 0.9) break;
    }
    return best || allowed;
  }

  function amputateApex() {
    var bins = shapedWound();
    if (!bins) { setStatus('no apex annotation for this section'); return; }
    var any = false;
    for (var i = 0; i < bins.length; i++) if (woundBin(bins[i])) any = true;
    startClockIfNeeded(any);
    renderClock();
    render();
    if (!S.playing) togglePlay();
  }

  /* ---------------------------------------------------------------- clock ---
   * Each measured interval is traversed in `steps_per_leg` updates, so one
   * update carries leg_hours[leg] / steps_per_leg hours. Past the last
   * timepoint the clock holds: the loop back to "uninjured" is a training
   * device, not elapsed biology. */
  var HOLD_AFTER = 15;        // steps held at the last timepoint
  var IDLE_RESET = 600;       // uninjured updates before going back to the seed

  function advanceClock() {
    if (!S.injured) return;
    var m = S.manifest, spl = m.steps_per_leg, last = m.leg_hours.length - 1;
    // The trained dynamics are a loop: 28 dpa returns to uninjured. What
    // should happen when the loop closes is the reader's call.
    if (S.leg >= last) {
      S.held = (S.held || 0) + 1;
      if (S.held > HOLD_AFTER) {
        if (el.stop && el.stop.checked && !S.stopped) {
          // Pause with the settled state and its density tracks still on
          // screen, which is the state the paper's readout is taken at.
          S.stopped = true;
          pause();
          setStatus('stopped at 28 dpa + ' + HOLD_AFTER + ' steps');
          return;
        }
        // Otherwise the rule keeps running indefinitely from the state it
        // reached -- NOT a reset to the pristine section, which would throw
        // away everything it built. The clock goes back to uninjured and the
        // density tracks clear, so the next injury profiles a fresh episode.
        // The tracks stay up: they are the record of the episode that just
        // finished, and blanking them the instant the loop closes throws the
        // result away at the moment it is worth reading. The next injury
        // clears them (see startClockIfNeeded).
        S.injured = false; S.stopped = false;
        S.hours = 0; S.leg = 0; S.stepInLeg = 0; S.held = 0;
        setStatus('uninjured — tracks held until the next injury');
        return;
      }
    }
    if (S.leg <= last && m.leg_hours[S.leg] > 0) {
      S.hours += m.leg_hours[S.leg] / spl;
    }
    S.stepInLeg += 1;
    if (S.stepInLeg >= spl && S.leg < last) {
      S.leg += 1;
      S.stepInLeg = 0;
      // Snap to the measured value so float drift cannot accumulate.
      S.hours = m.stage_hours[Math.min(S.leg, m.stage_hours.length - 1)];
    }
  }

  function fmtTime(h) {
    if (h < 24) return (Math.round(h * 10) / 10) + ' hpa';
    var d = h / 24;
    return (d < 10 ? Math.round(d * 10) / 10 : Math.round(d)) + ' dpa';
  }

  function renderClock() {
    var m = S.manifest;
    if (!S.injured) {
      el.now.textContent = 'uninjured';
      el.sub.textContent = 'not injured yet — the clock starts at the first damaged bin';
    } else {
      el.now.textContent = fmtTime(S.hours);
      var atEnd = S.leg >= m.leg_hours.length - 1;
      el.sub.textContent = atEnd
        ? 'held at ' + m.stage_names[m.stage_names.length - 1] + ' — regenerated, '
          + (S.stopped ? 'stopped here' : 'still stepping')
        : 'traversing ' + m.stage_names[S.leg] + ' → ' + m.stage_names[S.leg + 1]
          + '  ·  step ' + S.stepInLeg + '/' + m.steps_per_leg
          + '  ·  ' + S.steps + ' updates';
    }
    var html = '';
    for (var i = 0; i < m.stage_names.length; i++) {
      var reached = S.injured && S.hours >= m.stage_hours[i] - 1e-6;
      var target = S.injured && i === Math.min(S.leg + 1, m.stage_names.length - 1);
      html += '<span class="tp' + (reached ? ' reached' : '')
            + (target ? ' target' : '') + '">' + m.stage_names[i] + '</span>';
    }
    el.tps.innerHTML = html;
  }

  /* ------------------------------------------------------------- stepping --- */

  function currentDt() {
    var d = S.manifest.dt_per_stage;
    if (!d) return null;
    return d[Math.min(S.leg, d.length - 1)];
  }

  function runStep() {
    if (S.stepping || !S.session) return Promise.resolve();
    S.stepping = true;
    var feeds = { state: new ort.Tensor('float32', S.current, [1, S.C, S.H, S.W]) };
    if (S.hasDt) {
      var dt = currentDt();
      if (dt !== null) feeds.dt = new ort.Tensor('float32', new Float32Array([dt]), [1]);
    }
    return S.session.run(feeds).then(function (out) {
      S.current = new Float32Array(out.next_state.data);
      S.steps += 1;
      // Nothing is being measured while the heart sits uninjured, and the rule
      // does drift over thousands of updates, so a long unattended wait goes
      // back to the pristine section instead of accumulating that drift.
      if (!S.injured) {
        S.idle = (S.idle || 0) + 1;
        if (S.idle > IDLE_RESET) {
          resetHeart();
          setStatus('reset after ' + IDLE_RESET + ' updates with no wound');
          return;
        }
      } else {
        S.idle = 0;
      }
      advanceClock();
      recordTrace();
      renderClock();
      render();
      drawTrace();
    }).catch(function (e) {
      setStatus('step failed: ' + e.message);
      pause();
    }).then(function () { S.stepping = false; });
  }

  /* Mean density of one cell type over the living tissue. Per bin rather than
   * summed, so the trace reports composition and not heart size -- the tissue
   * area changes as a wound opens and closes. */
  function channelDensity(ch, live) {
    var HW = S.H * S.W, off = ch * HW, total = 0;
    for (var px = 0; px < HW; px++) {
      var v = S.current[off + px];
      if (v > 0) total += v;
    }
    return total / Math.max(live, 1);
  }

  function liveBins() {
    var HW = S.H * S.W, off = S.aliveIdx * HW, n = 0;
    for (var px = 0; px < HW; px++) if (S.current[off + px] >= VOID) n++;
    return n;
  }

  /* The abundance plots: one line per cell type, in that type's own atlas
   * colour, with a plot per tracked group.
   *
   * A plot per group rather than nine lines on one axis, because the two
   * groups differ in scale by enough that sharing an axis would flatten the
   * cascade into the baseline. Within a group the axis IS shared, so the
   * relative sizes of its members stay readable -- which is the whole point of
   * watching the cascade go through its states in order.
   *
   * One sample per update, redrawn from the record each time: cheap at a few
   * hundred points, and it cannot drift out of step with the simulation the
   * way an incrementally-drawn trace would after a reset. */
  function initPlots() {
    TRACKED.forEach(function (key) {
      var cv = el['plot-' + key.toLowerCase()];
      if (!cv) return;
      cv.width = Math.max(cv.clientWidth, 1) * 2;
      cv.height = 104;
      S.plot[key] = cv.getContext('2d');
    });
  }

  function recordTrace() {
    // Nothing to plot before the clock starts: an uninjured heart is at
    // homeostasis with no time post-injury, so recording it would pile every
    // sample onto t = 0 and smear the left edge.
    if (!S.injured) return;
    var live = liveBins();
    var row = { step: S.trace.length, v: {} };
    GROUPS.forEach(function (g) {
      if (TRACKED.indexOf(g.key) < 0) return;
      g.ids.forEach(function (ch) { row.v[ch] = channelDensity(ch, live); });
    });
    S.trace.push(row);
  }

  function drawTrace() {
    var pal = S.manifest.palette, spl = S.manifest.steps_per_leg;
    GROUPS.forEach(function (grp) {
      if (TRACKED.indexOf(grp.key) < 0) return;
      var g = S.plot[grp.key], cv = el['plot-' + grp.key.toLowerCase()];
      if (!g || !cv) return;
      var W = cv.width, H = cv.height, PAD = 20;
      var x_of = function (step) {
        return PAD + (W - PAD * 2) * Math.min(step / CYCLE_STEPS, 1);
      };
      var y_of = function (v) {
        return (H - 12) - (H - 24) * Math.min(v / grp.ymax, 1);
      };
      g.clearRect(0, 0, W, H);

      // one rule per measured interval, because a leg is a fixed 15 updates
      g.strokeStyle = 'rgba(0,0,0,0.07)';
      g.lineWidth = 2;
      for (var k = 0; k <= CYCLE_STEPS; k += spl) {
        g.beginPath(); g.moveTo(x_of(k), 4); g.lineTo(x_of(k), H - 12); g.stroke();
      }
      g.strokeStyle = 'rgba(0,0,0,0.16)';
      g.beginPath(); g.moveTo(PAD, H - 12); g.lineTo(W - PAD, H - 12); g.stroke();

      grp.ids.forEach(function (ch) {
        var c = pal[ch];
        g.strokeStyle = 'rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')';
        g.lineWidth = 3.5;
        g.lineJoin = 'round';
        g.beginPath();
        var started = false;
        S.trace.forEach(function (r) {
          var px = x_of(r.step), py = y_of(r.v[ch]);
          if (started) g.lineTo(px, py); else { g.moveTo(px, py); started = true; }
        });
        g.stroke();
      });
    });
  }

  /* The step pump. Deliberately a timer rather than requestAnimationFrame:
   * a step costs a few milliseconds, so rAF would cap the simulation at the
   * display refresh rate and stall it entirely in a throttled tab. Each step
   * schedules the next one relative to its own start, so the requested rate is
   * honoured without drifting, and a step slower than the interval simply
   * runs back-to-back instead of queueing up. */
  function pump() {
    if (!S.playing) return;
    var started = performance.now();
    runStep().then(function () {
      if (!S.playing) return;
      var wait = Math.max(0, 1000 / S.sps - (performance.now() - started));
      S.timer = setTimeout(pump, wait);
    });
  }

  /* ------------------------------------------------------------ rendering ---
   * Spatial transcriptomics is a regular array of capture spots, so this draws
   * one disc per bin rather than a square pixel: a square-pixel rendering reads
   * as an image, which this is not.
   *
   * Damage never colours a spot. The wound is physically a region with almost
   * no cells, so it renders as faded spots with its boundary traced, rather
   * than as a category of its own -- the same rule the paper's heart panels
   * use. Negative channel values are clipped away rather than flagged: the rule
   * does drive channels slightly below zero, and colouring that would put a
   * loud artefact on top of the data. */
  function initCanvas() {
    el.canvas.width = S.W * SCALE;
    el.canvas.height = S.H * SCALE;
    S.ctx = el.canvas.getContext('2d');
  }

  function render() {
    var HW = S.H * S.W, aOff = S.aliveIdx * HW, pal = S.manifest.palette;
    var ctx = S.ctx, r = D.dot * SCALE, LEVELS = 12;
    var thr = D.bzm_thresholds, bzm = D.bzm_channels;
    ctx.fillStyle = 'rgb(' + BG.join(',') + ')';
    ctx.fillRect(0, 0, el.canvas.width, el.canvas.height);

    // Batch the discs by (type, quantised opacity) so a frame costs a few dozen
    // fills rather than 2304 of them.
    var buckets = new Map();
    for (var px = 0; px < HW; px++) {
      if (S.current[aOff + px] < VOID) continue;
      var best = -1, bestVal = -Infinity, mass = 0, ch, v;
      for (ch = 0; ch < S.nv; ch++) {
        if (ch === S.dmgIdx) continue;     // damage is dropped entirely
        v = S.current[ch * HW + px];
        if (v < 0) v = 0;                  // clip, never flag
        mass += v;
        if (v > bestVal) { bestVal = v; best = ch; }
      }
      // A wounded bin has lost its cells, so its remaining cell sum falls below
      // the floor and it is not drawn at all. The wound is the ABSENCE of
      // tissue, not a substance -- hence no wound colour and no outline.
      if (best < 0 || mass < D.empty_below) continue;
      var dens = Math.min(1, mass / D.capacity);

      // The cascade states lose a plain argmax almost everywhere they appear,
      // so a state that exceeds its own pooled threshold takes the label.
      // Ranking by value/threshold is scale-free, and each state can only claim
      // its own extreme bins, so the stroma is displaced nowhere else.
      if (thr && dens >= D.min_density) {
        var bestR = 1.0;
        for (var b = 0; b < bzm.length; b++) {
          var t = thr[b];
          if (!(t > 0)) continue;
          var ratio = Math.max(0, S.current[bzm[b] * HW + px]) / t;
          if (ratio > bestR) { bestR = ratio; best = bzm[b]; }
        }
      }

      var a = D.alpha_floor + (1 - D.alpha_floor) * dens;
      var key = best * LEVELS + Math.round(a * (LEVELS - 1));
      var path = buckets.get(key);
      if (path === undefined) { path = new Path2D(); buckets.set(key, path); }
      var cx = (px % S.W + 0.5) * SCALE, cy = ((px / S.W) | 0) * SCALE + SCALE / 2;
      path.moveTo(cx + r, cy);
      path.arc(cx, cy, r, 0, Math.PI * 2);
    }
    buckets.forEach(function (path, key) {
      var c = pal[(key / LEVELS) | 0], a = (key % LEVELS) / (LEVELS - 1);
      ctx.fillStyle = 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + a.toFixed(3) + ')';
      ctx.fill(path);
    });
  }

  /* ---------------------------------------------------------------- chrome --- */

  function shortName(raw) { return raw.replace(/^\d+:/, ''); }

  function buildSeedList() {
    el.seed.innerHTML = S.manifest.seed_ids.map(function (id, i) {
      return '<option value="' + i + '">' + id.replace(/_/g, ' ') + '</option>';
    }).join('');
    el.seed.value = String(S.seedIdx);
  }

  function buildLegend() {
    var names = S.manifest.channels, pal = S.manifest.palette, html = '';
    GROUPS.forEach(function (g) {
      var traced = TRACKED.indexOf(g.key) >= 0;
      html += '<div class="legend-group-label" style="grid-column:1/-1;'
           +  'font-size:10.5px;color:#6b6b70;margin:7px 0 1px">'
           +  (traced ? '<i class="trace-key" style="background:' + g.colour
                        + '"></i>' : '')
           +  '<b style="color:#212228">' + g.name + '</b> &middot; ' + g.tag
           +  '</div>';
      g.ids.forEach(function (i) {
        var nm = shortName(names[i]), c = pal[i];
        html += '<span class="legend-item" title="' + (FULL_NAME[nm] || nm) + '">'
             + '<i style="background:rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')"></i>'
             + '<span class="nm">' + nm + '</span></span>';
      });
    });
    el.legend.innerHTML = html;
  }

  function togglePlay() {
    S.playing = !S.playing;
    el.play.innerHTML = S.playing ? '&#10073;&#10073;&ensp;Pause' : '&#9654;&ensp;Play';
    if (S.playing) pump(); else clearTimeout(S.timer);
  }

  function pause() { if (S.playing) togglePlay(); }

  function canvasToBin(ev) {
    var r = el.canvas.getBoundingClientRect();
    var x = (ev.clientX - r.left) / r.width * S.W;
    var y = (ev.clientY - r.top) / r.height * S.H;
    return [Math.floor(y), Math.floor(x)];
  }

  function wireRunControls() {
    el.play.addEventListener('click', togglePlay);
    el.step.addEventListener('click', function () { runStep(); });
    el.injure.addEventListener('click', amputateApex);
    el.reset.addEventListener('click', resetHeart);

    el.seed.addEventListener('change', function () {
      S.seedIdx = Number(el.seed.value);
      resetHeart();
    });

    // Pointer events cover mouse, touch and pen with one path, and
    // setPointerCapture keeps a drag alive if it leaves the canvas.
    el.canvas.addEventListener('pointerdown', function (ev) {
      ev.preventDefault();
      S.painting = true;
      el.canvas.setPointerCapture(ev.pointerId);
      var b = canvasToBin(ev);
      paintAt(b[0], b[1]);
    });
    el.canvas.addEventListener('pointermove', function (ev) {
      if (!S.painting) return;
      var b = canvasToBin(ev);
      paintAt(b[0], b[1]);
    });
    ['pointerup', 'pointercancel'].forEach(function (t) {
      el.canvas.addEventListener(t, function () { S.painting = false; });
    });

    // The plot's backing store is sized in device pixels, so it has to be
    // rebuilt when the column width changes.
    var resizeTimer;
    window.addEventListener('resize', function () {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(function () { initPlots(); drawTrace(); }, 150);
    });

    // Pause when the widget scrolls out of view: there is no reason to spend
    // the reader's CPU on a simulation they cannot see.
    if ('IntersectionObserver' in window) {
      new IntersectionObserver(function (es) {
        es.forEach(function (e) { if (!e.isIntersecting) pause(); });
      }, { threshold: 0 }).observe(widget);
    }
  }

  // Preloaded rather than gated behind a click. The button stays as a
  // retry path when loading fails; it is just not the only way in.
  el.load.addEventListener('click', boot);
  if (el.gate) {
    var gp = el.gate.querySelector('p');
    if (gp) gp.textContent = 'Loading the learned rule and 16 real uninjured heart sections\u2026';
    el.load.hidden = true;
  }
  boot();
})();
