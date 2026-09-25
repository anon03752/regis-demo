/* Loading the widgets' binaries without fetch().

   Browsers refuse fetch() for local files, so a page opened from disk can
   show every figure (src/href are exempt) but cannot read a model. Script
   tags are exempt too, so each binary ships as base64 in its own .js under
   assets/embed/ and is pulled in on demand from here.

   Nothing is loaded until a widget asks for it: the runtime alone is 13 MB
   of base64, and a reader who never reaches a widget should not pay for it.
*/
window.EMBED = (function () {
  'use strict';
  var DIR = 'assets/embed/';
  var pending = {};
  var wasmUrlPromise = null;

  function script(key) {
    if (pending[key]) return pending[key];
    pending[key] = new Promise(function (res, rej) {
      if (window.__EMBED && window.__EMBED[key]) return res();
      var s = document.createElement('script');
      s.src = DIR + key + '.js';
      s.onload = function () {
        (window.__EMBED && window.__EMBED[key]) ? res()
          : rej(new Error(key + '.js loaded but set nothing'));
      };
      s.onerror = function () { rej(new Error('could not load ' + s.src)); };
      document.head.appendChild(s);
    });
    return pending[key];
  }

  // atob gives a binary string; walk it out into real bytes. Chunked so a
  // multi-megabyte payload does not build one enormous intermediate array.
  function bytes(b64) {
    var bin = atob(b64), n = bin.length, out = new Uint8Array(n);
    for (var i = 0; i < n; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  return {
    /* -> Promise<Uint8Array> */
    binary: function (key) {
      return script(key).then(function () { return bytes(window.__EMBED[key]); });
    },
    /* -> Promise<object> */
    json: function (key) {
      return script(key).then(function () {
        return JSON.parse(new TextDecoder().decode(bytes(window.__EMBED[key])));
      });
    },
    /* -> Promise<string>, a data: URL onnxruntime can load the wasm from.
       fetch() accepts data: URLs even on a file:// page, which is the whole
       reason this works where fetching the .wasm beside it does not. */
    wasmUrl: function () {
      // Memoised on the PROMISE, not on the result. Both widgets ask for
      // this at once; caching the value instead let the first one build the
      // URL and drop the base64, and the second then built
      // "...;base64,undefined" and clobbered the good one with it.
      if (!wasmUrlPromise) {
        wasmUrlPromise = script('wasm').then(function () {
          var url = 'data:application/wasm;base64,' + window.__EMBED.wasm;
          // The URL holds the only copy that matters now, and this is 13 MB
          // of string -- keeping both doubles the runtime's footprint for
          // the life of the page.
          delete window.__EMBED.wasm;
          return url;
        });
      }
      return wasmUrlPromise;
    },
    /* Frees the base64 once it has been decoded; these are the largest
       strings on the page and nothing needs them twice. */
    release: function (key) {
      if (window.__EMBED) delete window.__EMBED[key];
    }
  };
}());
