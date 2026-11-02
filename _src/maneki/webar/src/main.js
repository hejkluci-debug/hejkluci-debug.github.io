import { MindARThree } from 'mind-ar/dist/mindar-image-three.prod.js';
import { Clock } from 'three';
import { createCatRig, loadTextures } from './cat-rig.js';
import { createVideoElement, createVideoOverlay } from './video-overlay.js';

const BASE = './assets';
const $ = (id) => document.getElementById(id);
const t0 = performance.now();
const marks = {};
const mark = (name) => {
  marks[name] = Math.round(performance.now() - t0);
  console.log(`[maneki] ${name} ${marks[name]} ms`);
  if (debug) $('debug').textContent = JSON.stringify(marks, null, 1);
};
const params = new URLSearchParams(location.search);
const debug = params.has('debug');
const num = (k, d) => (params.has(k) ? Number(params.get(k)) : d);
if (debug) $('debug').hidden = false;
window.__maneki = { marks };
mark('script');

// Prefetch everything that does not need the camera while the user reads the intro.
const scenePromise = fetch(`${BASE}/scene.json`, { cache: 'no-cache' }).then((r) => r.json());
let video = null;
const texPromise = scenePromise.then((sc) => {
  if (sc.mode === 'video') {
    video = createVideoElement(BASE, sc.video.src.map((f) => (sc.ver ? `${f}?v=${sc.ver}` : f)));
    window.__maneki.overlayVideo = video;
    return new Promise((res) => {
      if (video.readyState >= 2) return res(null);
      video.addEventListener('loadeddata', () => res(null), { once: true });
      video.addEventListener('error', () => res(null), { once: true });
      setTimeout(() => res(null), 8000);
    });
  }
  return loadTextures(BASE, ['arm', 'underlay']);
});
const targetPromise = scenePromise.then((sc) => fetch(`${BASE}/target.mind${sc.ver ? `?v=${sc.ver}` : ''}`)).then((r) => r.blob()).then((b) => URL.createObjectURL(b));
Promise.all([scenePromise, texPromise, targetPromise]).then(() => {
  mark('assets');
  $('start').disabled = false;
  $('start').textContent = 'Spustit kameru';
});

$('start').addEventListener('click', async () => {
  $('intro').hidden = true;
  $('hint').hidden = false;
  $('hint').textContent = 'Spouštím kameru…';
  mark('tap');
  // iOS: video playback must start inside the user gesture
  if (video) video.play().catch((e) => console.warn('video play', e));
  try {
    const [scene, textures, targetUrl] = await Promise.all([scenePromise, texPromise, targetPromise]);
    const mindar = new MindARThree({
      container: $('ar'), imageTargetSrc: targetUrl, maxTrack: 1,
      uiLoading: 'no', uiScanning: 'no', uiError: 'no',
      filterMinCF: num('cf', 0.0005), filterBeta: num('beta', 50),
    });
    const { renderer, scene: s3, camera } = mindar;
    const anchor = mindar.addAnchor(0);
    const cat = scene.mode === 'video' ? createVideoOverlay(scene, video) : createCatRig(scene, textures);
    anchor.group.add(cat.group);
    let found = false;
    anchor.onTargetFound = () => {
      found = true; cat.reset();
      if (!marks.found) mark('found');
      $('hint').hidden = true;
    };
    anchor.onTargetLost = () => {
      found = false;
      $('hint').hidden = false;
      $('hint').textContent = 'Namiř telefon na kočku na tričku';
    };
    await mindar.start();
    mark('camera');
    $('hint').textContent = 'Namiř telefon na kočku na tričku';
    // White balance: match the virtual layers to the camera's view of the shirt.
    // Every 400 ms take the bright end of the frame centre as the "white" of the
    // fabric and tint the overlay with it.
    const probe = document.createElement('canvas'); probe.width = probe.height = 48;
    const pctx = probe.getContext('2d', { willReadFrequently: true });
    const camVideo = mindar.video;
    let lastProbe = 0;
    const sampleWhite = () => {
      if (!camVideo || camVideo.readyState < 2) return;
      const vw = camVideo.videoWidth, vh = camVideo.videoHeight;
      pctx.drawImage(camVideo, vw * 0.2, vh * 0.2, vw * 0.6, vh * 0.6, 0, 0, 48, 48);
      const d = pctx.getImageData(0, 0, 48, 48).data;
      const lum = [];
      for (let i = 0; i < d.length; i += 4) lum.push([0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2], i]);
      lum.sort((a, b) => b[0] - a[0]);
      let r = 0, g = 0, b = 0; const n = Math.max(1, Math.floor(lum.length * 0.1));
      for (let k = 0; k < n; k++) { const i = lum[k][1]; r += d[i]; g += d[i + 1]; b += d[i + 2]; }
      cat.setTint(r / n / 255, g / n / 255, b / n / 255);
    };
    const clock = new Clock();
    renderer.setAnimationLoop(() => {
      const dt = clock.getDelta();
      if (found) cat.update(clock.elapsedTime, dt);
      if (clock.elapsedTime - lastProbe > 0.4) { lastProbe = clock.elapsedTime; sampleWhite(); }
      renderer.render(s3, camera);
    });
  } catch (e) {
    console.error(e);
    $('hint').hidden = false;
    $('hint').textContent = 'Kameru se nepodařilo spustit. Povol přístup ke kameře a obnov stránku.';
    mark('error');
  }
});
