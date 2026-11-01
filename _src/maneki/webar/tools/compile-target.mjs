// Compile public/assets/target.png -> public/assets/target.mind with MindAR's own
// browser compiler, run headless in Chromium (no native node deps needed).
import { createRequire } from 'node:module';
const { chromium } = createRequire(import.meta.url)('playwright');
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
const root = path.resolve('.');
const types = { '.html': 'text/html', '.js': 'text/javascript', '.png': 'image/png' };
const server = http.createServer((req, res) => {
  const p = path.join(root, decodeURIComponent(req.url.split('?')[0]));
  if (!fs.existsSync(p)) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { 'content-type': types[path.extname(p)] || 'application/octet-stream' });
  fs.createReadStream(p).pipe(res);
}).listen(8765);
const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage();
page.on('console', (m) => console.log('[page]', m.text()));
await page.goto('http://localhost:8765/tools/compile/index.html');
await page.waitForFunction(() => window.compileTarget);
const r = await page.evaluate(() => window.compileTarget('/tools/target.png'));
fs.writeFileSync('public/assets/target.mind', Buffer.from(r.b64, 'base64'));
console.log(`target.mind ${r.size} bytes, compiled in ${Math.round(r.ms)} ms`);
await browser.close(); server.close();
