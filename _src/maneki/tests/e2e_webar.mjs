// End-to-end test of the WebAR page in headless Chromium with a fake camera.
// Measures: assets ready, camera ready, target found; takes screenshots.
import { createRequire } from 'node:module';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
const { chromium } = createRequire(import.meta.url)('playwright');
const root = path.resolve('../webar/public');
const types = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.webp': 'image/webp', '.png': 'image/png', '.mind': 'application/octet-stream' };
let bytes = 0;
const server = http.createServer((req, res) => {
  let p = path.join(root, decodeURIComponent(req.url.split('?')[0]));
  if (p.endsWith('/')) p += 'index.html';
  if (!fs.existsSync(p)) { res.writeHead(404); return res.end(); }
  bytes += fs.statSync(p).size;
  res.writeHead(200, { 'content-type': types[path.extname(p)] || 'application/octet-stream' });
  fs.createReadStream(p).pipe(res);
}).listen(8766);
const browser = await chromium.launch({ args: [
  '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
  `--use-file-for-fake-video-capture=${path.resolve('fake_camera.y4m')}`,
  '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const ctx = await browser.newContext({ viewport: { width: 390, height: 780 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true, permissions: ['camera'] });
const page = await ctx.newPage();
const logs = [];
page.on('console', (m) => { if (m.text().startsWith('[maneki]') || m.type() === 'error') logs.push(m.text()); });
await page.goto('http://localhost:8766/?debug=1');
await page.waitForSelector('#start:not([disabled])', { timeout: 30000 });
await page.screenshot({ path: 'shot_0_intro.png' });
await page.tap('#start');
let found = false;
try { await page.waitForFunction(() => window.__maneki.marks.found, null, { timeout: 60000 }); found = true; } catch {}
const marks = await page.evaluate(() => window.__maneki.marks);
if (found) {
  await page.waitForTimeout(500); await page.screenshot({ path: 'shot_1_found.png' });
  await page.waitForTimeout(800); await page.screenshot({ path: 'shot_2_anim.png' });
} else {
  await page.screenshot({ path: 'shot_fail.png' });
}
const videoState = await page.evaluate(() => { const v = window.__maneki.overlayVideo; return v ? {src: v.currentSrc.split('/').pop(), t: v.currentTime, paused: v.paused, rs: v.readyState, err: v.error && v.error.code} : null; });
const fps = await page.evaluate(() => new Promise((res) => { let n = 0; const t = performance.now(); const f = () => { n++; if (performance.now() - t < 2000) requestAnimationFrame(f); else res(n / 2); }; requestAnimationFrame(f); }));
console.log(JSON.stringify({ found, marks, bytesServed: bytes, rafFps: fps, videoState, logs: logs.slice(0, 20) }, null, 1));
await browser.close(); server.close();
