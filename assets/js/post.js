/* Page furniture: contents sidebar, citation and method previews, figure zoom,
   reduced-motion handling and the read-out dots that ride the figure videos.
   No dependencies at all --
   nothing here is loaded from anywhere but this repository. */
(function () {
  'use strict';

  /* ------------------------------------------------------------- previews ---
     Adjacent citations share a preview, built from the bibliography. The
     preview sits immediately after its markers in the DOM, so Tab reaches
     its source links naturally and then continues through the article.
     Method descriptions share the same positioning and dismissal behavior. */
  function wirePreviews() {
    var active = null, activeLink = null, closeTimer;

    function hide() {
      clearTimeout(closeTimer);
      if (active) {
        active.tip.hidden = true;
        active.links.forEach(function (link) { link.setAttribute('aria-expanded', 'false'); });
      }
      active = activeLink = null;
    }

    function place() {
      if (!active) return;
      var rect = active.group.getBoundingClientRect();
      var vw = document.documentElement.clientWidth, vh = window.innerHeight;
      if (rect.bottom < 0 || rect.top > vh || rect.right < 0 || rect.left > vw) {
        hide();
        return;
      }
      var tip = active.tip, gutter = vw <= 600 ? 12 : 16;
      tip.style.setProperty('--cite-max-height', (vh - 2 * gutter - 8) + 'px');
      var below = vh - rect.bottom - gutter, above = rect.top - gutter;
      var onTop = tip.offsetHeight > below && above > below;
      tip.dataset.side = onTop ? 'above' : 'below';
      tip.style.setProperty('--cite-max-height', Math.max(0, (onTop ? above : below) - 8) + 'px');
      var left = Math.max(gutter, Math.min(rect.right, vw - tip.offsetWidth - gutter));
      tip.style.left = left + 'px';
      tip.style.top = (onTop ? rect.top - tip.offsetHeight : rect.bottom) + 'px';
    }

    function show(preview, link) {
      clearTimeout(closeTimer);
      if (active !== preview) hide();
      active = preview;
      activeLink = link;
      active.tip.hidden = false;
      active.links.forEach(function (a) { a.setAttribute('aria-expanded', 'true'); });
      place();
    }

    function scheduleHide() {
      clearTimeout(closeTimer);
      closeTimer = setTimeout(function () {
        if (active && !active.group.matches(':hover') && !active.tip.matches(':hover') &&
            !active.group.contains(document.activeElement) &&
            !active.tip.contains(document.activeElement)) hide();
      }, 250);
    }

    function reference(link) {
      var ref = document.getElementById(link.getAttribute('href').slice(1));
      return ref && ref.querySelector('.t') ? ref : null;
    }

    function makeEntry(ref) {
      var entry = document.createElement('span');
      entry.className = 'cite-entry';
      var title = ref.querySelector('.t').cloneNode(true);
      title.textContent = title.textContent.replace(/\.$/, '');
      entry.appendChild(title);
      ref.querySelectorAll('.ref-authors, .ref-publication').forEach(function (part) {
        entry.appendChild(part.cloneNode(true));
      });
      var sources = document.createElement('span');
      sources.className = 'cite-tooltip-sources';
      ref.querySelectorAll('a[href]').forEach(function (source) {
        var link = source.cloneNode(true);
        var doi = link.href.match(/^https?:\/\/(?:dx\.)?doi\.org\/(.+)$/);
        if (doi) link.textContent = 'DOI: ' + doi[1];
        else if (!/^arxiv:/i.test(link.textContent)) link.textContent = 'Read paper';
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        link.setAttribute('aria-label', 'Read ' + title.textContent + ' (opens in a new tab)');
        if (sources.childNodes.length) sources.appendChild(document.createTextNode(' · '));
        sources.appendChild(link);
      });
      entry.appendChild(sources);
      return entry;
    }

    document.querySelectorAll('a.cite[href^="#"]').forEach(function (first, index) {
      // Method cards already show the reference; avoid nested previews.
      if (first.closest('.method-tooltip')) return;
      if (first.parentElement.classList.contains('cite-group') || !reference(first)) return;
      var links = [first], next = first.nextSibling;
      while (next) {
        if (next.nodeType === 3 && !next.textContent.trim()) {
          next = next.nextSibling;
          continue;
        }
        if (next.nodeType !== 1 || !next.matches('a.cite[href^="#"]') || !reference(next)) break;
        links.push(next);
        next = next.nextSibling;
      }
      var group = document.createElement('span');
      group.className = 'cite-group';
      first.before(group);
      group.appendChild(document.createTextNode('['));
      links.forEach(function (link, i) {
        // Remove only the whitespace between markers that are being grouped.
        while (i < links.length - 1 && link.nextSibling !== links[i + 1]) {
          link.nextSibling.remove();
        }
        if (i) group.appendChild(document.createTextNode(', '));
        link.textContent = link.textContent.replace(/[\[\]]/g, '');
        link.setAttribute('aria-label', 'Reference ' + link.textContent);
        group.appendChild(link);
      });
      group.appendChild(document.createTextNode(']'));

      var tip = document.createElement('span');
      tip.id = 'cite-tooltip-' + index;
      tip.className = 'cite-tooltip';
      tip.setAttribute('role', 'dialog');
      tip.setAttribute('aria-label', 'References ' + group.textContent);
      tip.hidden = true;
      var card = document.createElement('span');
      card.className = 'cite-tooltip-card';
      links.forEach(function (link) {
        card.appendChild(makeEntry(reference(link)));
        link.setAttribute('aria-controls', tip.id);
        link.setAttribute('aria-haspopup', 'dialog');
        link.setAttribute('aria-expanded', 'false');
        link.setAttribute('aria-describedby', reference(link).id);
      });
      tip.appendChild(card);
      group.after(tip);
      var preview = { group: group, links: links, tip: tip };
      var touchStarted = false, touchWasOpen = false;
      group.addEventListener('pointerenter', function (ev) {
        if (ev.pointerType !== 'touch') show(preview, group.querySelector('a:focus') || links[0]);
      });
      group.addEventListener('pointerleave', scheduleHide);
      group.addEventListener('focusin', function (ev) { show(preview, ev.target); });
      group.addEventListener('focusout', scheduleHide);
      group.addEventListener('pointerdown', function (ev) {
        touchStarted = ev.pointerType === 'touch';
        touchWasOpen = active === preview;
      });
      links.forEach(function (link) {
        link.addEventListener('click', function (ev) {
          // A first tap opens the preview; a second follows the original link.
          if (touchStarted && !touchWasOpen) {
            ev.preventDefault();
            show(preview, link);
          } else hide();
          touchStarted = false;
        });
      });
      tip.addEventListener('pointerenter', function () { clearTimeout(closeTimer); });
      tip.addEventListener('pointerleave', scheduleHide);
      tip.addEventListener('focusin', function () { clearTimeout(closeTimer); });
      tip.addEventListener('focusout', scheduleHide);
    });

    document.querySelectorAll('button.method-info[aria-controls]').forEach(function (button) {
      var tip = document.getElementById(button.getAttribute('aria-controls'));
      if (!tip) return;
      // Keep the fixed-position card beside its button in the DOM so Tab
      // reaches its reference links before continuing to the next method.
      var preview = { group: button, links: [button], tip: tip };
      var openBeforePress = null;
      button.addEventListener('pointerenter', function (ev) {
        if (ev.pointerType !== 'touch') show(preview, button);
      });
      button.addEventListener('pointerleave', scheduleHide);
      button.addEventListener('focus', function () { show(preview, button); });
      button.addEventListener('blur', scheduleHide);
      button.addEventListener('pointerdown', function () { openBeforePress = active === preview; });
      button.addEventListener('keydown', function (ev) {
        if (ev.key === 'Enter' || ev.key === ' ') openBeforePress = active === preview;
      });
      button.addEventListener('click', function () {
        var wasOpen = openBeforePress === null ? active === preview : openBeforePress;
        openBeforePress = null;
        if (wasOpen) hide();
        else show(preview, button);
      });
      tip.addEventListener('pointerenter', function () { clearTimeout(closeTimer); });
      tip.addEventListener('pointerleave', scheduleHide);
      tip.addEventListener('focusin', function () { clearTimeout(closeTimer); });
      tip.addEventListener('focusout', scheduleHide);
      tip.addEventListener('click', function (ev) {
        var link = ev.target.closest('a[href^="#"]');
        if (!link) return;
        var ref = document.getElementById(link.getAttribute('href').slice(1));
        if (ref) {
          ref.tabIndex = -1;
          ref.focus({ preventScroll: true });
        }
        hide();
      });
    });
    document.querySelectorAll('.table-wrap').forEach(function (wrap) {
      wrap.addEventListener('scroll', hide, { passive: true });
    });

    document.addEventListener('keydown', function (ev) {
      if (ev.key === 'Escape' && active) {
        ev.preventDefault();
        if (active.tip.contains(document.activeElement)) activeLink.focus();
        hide();
      }
    });
    document.addEventListener('pointerdown', function (ev) {
      if (active && !active.tip.contains(ev.target) && !active.group.contains(ev.target)) hide();
    });
    addEventListener('scroll', function (ev) {
      if (active && (ev.target === window || !active.tip.contains(ev.target))) place();
    }, { passive: true, capture: true });
    addEventListener('resize', place);
  }

  /* ------------------------------------------------------------- contents ---
     Built from the headings rather than hand-maintained, so it cannot drift
     out of sync with the article. Only shown on very wide viewports (CSS). */
  function buildToc() {
    var toc = document.getElementById('toc');
    if (!toc) return;
    var heads = Array.prototype.slice.call(
      document.querySelectorAll('article h2[id], article h3[id]'));
    if (!heads.length) return;

    var frag = document.createDocumentFragment();
    var links = {};
    heads.forEach(function (h) {
      var a = document.createElement('a');
      a.href = '#' + h.id;
      a.textContent = h.textContent;
      if (h.tagName === 'H3') a.className = 'h3';
      links[h.id] = a;
      frag.appendChild(a);
    });
    toc.appendChild(frag);

    // The list is position: fixed, so at the top of the page it would sit up
    // level with the site header, above the thing it is a contents FOR. Pin
    // its top to the title's until the title has scrolled past, then let it
    // rest at REST. max-height follows, or a long list would run off-screen.
    var REST = 28;
    var h1 = document.querySelector('article h1');

    function place() {
      if (!h1) return;
      var top = Math.max(REST, h1.getBoundingClientRect().top);
      toc.style.top = top + 'px';
      toc.style.maxHeight = (window.innerHeight - top - 16) + 'px';
    }

    // Which section you are IN is a state, not an event: it is the last
    // heading to have passed LINE. This used to be an IntersectionObserver
    // watching a band 5% of the viewport tall, so except in the instant a
    // heading crossed that strip nothing matched and nothing was lit.
    var LINE = 140;
    var cur = null;

    function mark() {
      var id = heads[0].id;          // the lead-in belongs to the first section
      heads.forEach(function (h) {
        if (h.getBoundingClientRect().top <= LINE) id = h.id;
      });
      // The last section can be too short to ever reach LINE; at the bottom
      // of the page it is unambiguously the one being read.
      if (window.innerHeight + window.scrollY >=
          document.documentElement.scrollHeight - 2) {
        id = heads[heads.length - 1].id;
      }
      if (id === cur) return;
      if (cur && links[cur]) links[cur].classList.remove('on');
      cur = id;
      if (links[cur]) links[cur].classList.add('on');
    }

    function tick() { place(); mark(); }
    tick();
    addEventListener('scroll', tick, { passive: true });
    addEventListener('resize', tick);
  }

  /* ------------------------------------------------------------- lightbox ---
     Figures are shown at the column width but rendered at 2x, so full-screen
     is genuinely worth offering for the dense multi-panel ones. */
  function wireLightbox() {
    var box = document.getElementById('lightbox');
    if (!box) return;
    function close() {
      box.classList.remove('on');
      box.setAttribute('aria-hidden', 'true');
      box.innerHTML = '';
      document.body.style.overflow = '';
    }
    document.querySelectorAll('figure a.zoom').forEach(function (a) {
      a.addEventListener('click', function (ev) {
        ev.preventDefault();
        var img = document.createElement('img');
        img.src = a.getAttribute('href');
        img.alt = (a.querySelector('img') || {}).alt || '';
        box.innerHTML = '';
        box.appendChild(img);
        box.classList.add('on');
        box.setAttribute('aria-hidden', 'false');
        document.body.style.overflow = 'hidden';
      });
    });
    box.addEventListener('click', close);
    document.addEventListener('keydown', function (ev) {
      if (ev.key === 'Escape') close();
    });
  }

  /* ------------------------------------------------------- autoplay videos ---
     A looping autoplay video is exactly what prefers-reduced-motion is for.
     The attribute stays in the markup so the common case works without JS;
     here we undo it and hand the reader the controls instead. */
  function respectReducedMotion() {
    var videos = document.querySelectorAll('.hero video, .mnist-comparison video');
    if (!videos.length || !window.matchMedia) return;
    var mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    function apply() {
      if (!mq.matches) return;
      videos.forEach(function (v) {
        v.autoplay = false;
        v.loop = false;
        v.controls = true;
        v.pause();
        v.currentTime = 0;
      });
    }
    apply();
    if (mq.addEventListener) mq.addEventListener('change', apply);
  }

  /* --------------------------------------------------------- scheme tabs ---
     One drawing with three views: the tab sets a class on the figure and the
     stylesheet does the rest, so there is no per-view drawing code here. */
  // Which groups of the drawing belong to each view, keyed on the group ids.
  // Not on class names: the figure's layout is maintained in Illustrator, and
  // exporting from it rewrites every class attribute while leaving the ids
  // alone, so keying on classes silently breaks the tabs on every re-export.
  var SCHEME_VIEWS = {
    generator: ['#panel-gen', '#view-generator-callout',
                '#trajectories-apart', '#generated-distribution-apart'],
    discriminator: ['#panel-disc', '#view-discriminator-bracket',
                    '#trajectories-near', '#generated-distribution-near']
  };
  var SCHEME_ALL = ['#panel-gen', '#panel-disc',
                    '#view-generator-callout', '#view-discriminator-bracket',
                    '#trajectories-apart', '#trajectories-near',
                    '#generated-distribution-apart',
                    '#generated-distribution-near'];

  function wireSchemeTabs() {
    var fig = document.getElementById('scheme');
    if (!fig) return;
    var tabs = [].slice.call(fig.querySelectorAll('.scheme-tabs button'));
    var notes = [].slice.call(fig.querySelectorAll('.scheme-note'));

    function set(sels, v) {
      sels.forEach(function (sel) {
        [].forEach.call(fig.querySelectorAll(sel), function (el) {
          el.style.opacity = v;
        });
      });
    }

    function show(view) {
      set(SCHEME_ALL, '0');
      set(SCHEME_VIEWS[view] || [], '1');
      tabs.forEach(function (b) {
        b.setAttribute('aria-selected', String(b.dataset.view === view));
      });
      notes.forEach(function (n) { n.hidden = n.dataset.view !== view; });
    }

    tabs.forEach(function (b) {
      b.addEventListener('click', function () { show(b.dataset.view); });
      b.addEventListener('keydown', function (ev) {
        var d = ev.key === 'ArrowRight' ? 1 : ev.key === 'ArrowLeft' ? -1 : 0;
        if (!d) return;
        ev.preventDefault();
        var i = (tabs.indexOf(b) + d + tabs.length) % tabs.length;
        tabs[i].focus();
        show(tabs[i].dataset.view);
      });
    });
    show('generator');
  }

  /* ---------------------------------------------------- tracked spot ---
     The dot that marks where on a figure's curve the simulation in its
     panel currently is. Used by the Ising figure (log L against log t)
     and the heart figure (the regeneration loop).

     Driven by requestVideoFrameCallback, which fires once per PRESENTED
     video frame and reports that frame's mediaTime. The obvious
     alternatives are both wrong here. A CSS animation with a matching
     duration drifts away from a looping video within a minute. And
     requestAnimationFrame gets throttled while video playback carries on --
     measured in this app's own browser pane, where the dot froze at frame 0
     while the video ran to completion -- so the dot would annotate the wrong
     t without anything looking broken.

     The track is one (x, y) per video frame in the SVG's viewBox units, so
     placing it is a division by the viewBox size: no assumption about the
     rendered width. */
  /* Reveal a set of SVG traces in step with a video.

     The plots are drawn complete and then hidden by their own stroke length;
     advancing stroke-dashoffset uncovers exactly as much as the rollout has
     reached. Same presented-frame clock as the dots, so the curve and the
     panel above it can never disagree. */
  function wireTraceReveal(videoSel, traceSel) {
    var video = document.querySelector(videoSel);
    var traces = Array.prototype.slice.call(document.querySelectorAll(traceSel));
    if (!video || !traces.length) return;
    // The clip ends with a hold on its last frame. The curves have to finish
    // when the ROLLOUT does, not when the clip does, or they carry on being
    // drawn while the panels above them are frozen.
    var host = traces[0].ownerSVGElement;
    var runF = parseFloat(host && host.dataset.runFrames) || 0;
    var fps = parseFloat(host && host.dataset.fps) || 30;
    // runF FRAMES span runF - 1 intervals, and the trace's last point sits at
    // the last of them. Dividing by runF leaves the curves 99.2% drawn at the
    // frame the hearts freeze on, finishing during the hold; runF - 1 lands
    // the last point exactly on the last rollout frame.
    var span = runF > 1 ? (runF - 1) / fps : 0;
    var lens = traces.map(function (t) { return t.getTotalLength(); });
    traces.forEach(function (t, i) {
      t.style.strokeDasharray = lens[i];
      t.style.strokeDashoffset = lens[i];
    });
    function show(mediaTime) {
      var d = span || video.duration;
      var p = Math.min(Math.max((mediaTime || 0) / d, 0), 1);
      traces.forEach(function (t, i) {
        t.style.strokeDashoffset = lens[i] * (1 - p);
      });
    }
    if (video.requestVideoFrameCallback) {
      var f = function (now, meta) {
        show(meta.mediaTime);
        video.requestVideoFrameCallback(f);
      };
      video.requestVideoFrameCallback(f);
    }
    video.addEventListener('timeupdate', function () { show(video.currentTime); });
    show(video.currentTime);
  }

  // Every track is inlined into the page by tools/inline-tracks.py. It used
  // to be fetch()ed, which is the one thing on the page that needed a web
  // server: browsers refuse fetch() for local files, so opening the page from
  // disk silently left every dot at its CSS parking spot to the left of its
  // figure while everything else still worked.
  var TRACKS = (function () {
    var el = document.getElementById('tracks');
    if (!el) return {};
    try { return JSON.parse(el.textContent); } catch (e) { return {}; }
  })();

  function wireTrackedSpot(videoSel, spotId, trackName, key) {
    var spot = document.getElementById(spotId);
    var video = document.querySelector(videoSel);
    var tr = TRACKS[trackName];
    if (!spot || !video || !tr || !tr[key || 'pts']) return;
    if (window.matchMedia
        && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    (function () {
        var pts = tr[key || 'pts'], n = pts.length;
        // Interpolate between the two bracketing track points instead of
        // snapping to the nearest one. Snapping made the first second visibly
        // jerky: the early lattice is HELD for several frames, so callbacks
        // arrive unevenly there, and a dot that only moves when the frame
        // index changes inherits that unevenness. The curve is smooth in
        // mediaTime, so read it at mediaTime.
        function put(mediaTime) {
          var f = (mediaTime || 0) * tr.fps;
          f = ((f % n) + n) % n;
          // Clamp rather than wrap at the end: interpolating from the last
          // point back to the first would slide the dot across the whole axis
          // at the loop seam. The video restarting SHOULD jump it back.
          var i = Math.min(Math.floor(f), n - 1), j = Math.min(i + 1, n - 1);
          var u = f - i;
          var x = pts[i][0] + (pts[j][0] - pts[i][0]) * u;
          var y = pts[i][1] + (pts[j][1] - pts[i][1]) * u;
          spot.style.left = (100 * x / tr.w) + '%';
          spot.style.top = (100 * y / tr.h) + '%';
        }
        if (video.requestVideoFrameCallback) {
          var onFrame = function (now, meta) {
            put(meta.mediaTime);
            video.requestVideoFrameCallback(onFrame);
          };
          video.requestVideoFrameCallback(onFrame);
        } else {
          var tick = function () {
            put(video.currentTime);
            requestAnimationFrame(tick);
          };
          requestAnimationFrame(tick);
        }
        // Covers the fallback path when rAF is throttled, and a seek while
        // paused in either path.
        video.addEventListener('timeupdate', function () { put(video.currentTime); });
        video.addEventListener('seeked', function () { put(video.currentTime); });
        put(video.currentTime);
        // Autoplay can be refused (low-power mode, a per-site setting), and a
        // paused clip leaves the dot parked on frame 0 -- which on the heart
        // and perturbation loops is the uninjured node at the far left, so it
        // reads as a broken dot rather than a stopped film. Ask once.
        if (video.paused && video.autoplay) {
          var pr = video.play();
          if (pr && pr.catch) pr.catch(function () {});
        }
      })();
  }
  respectReducedMotion();
  wireTrackedSpot('.cycle-video', 'cycle-spot', 'cycle');
  wireTrackedSpot('.ising-video', 'ising-spot', 'ising');
  wireTrackedSpot('.heart-video', 'heart-spot', 'heart');
  wireTrackedSpot('.perturb-video', 'perturb-gold', 'perturb', 'gold');
  wireTrackedSpot('.perturb-video', 'perturb-blue', 'perturb', 'blue');
  wireTrackedSpot('.perturb-video', 'perturb-red', 'perturb', 'red');
  wireTraceReveal('.perturb-video', '.perturb-plots .pp-trace');
  wireSchemeTabs();
  buildToc();
  wirePreviews();
  wireLightbox();
})();
