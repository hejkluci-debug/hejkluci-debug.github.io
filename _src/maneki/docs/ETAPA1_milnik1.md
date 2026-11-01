# MANEKI-NEKO – Etapa 1 a první milník (1. 11. 2026)

## Rozhodnutí
| Oblast | Volba | Proč |
| --- | --- | --- |
| QR | standardní QR Model 2, EC H, krátká URL VELKÝMI písmeny (Alphanumeric mode) | čte ho fotoaparát každého telefonu; `HTTPS://MNK.CZ/1` = verze 2 (25×25) i při EC H |
| Image tracking | MindAR 1.2.5 (MIT) + three.js 0.160 | běží v prohlížeči přes getUserMedia; WebXR image tracking je na Androidu jen za příznakem Chrome a na iOS jen přes App Clip |
| Animace | vlastní rig v three.js: mřížka 18×32 s 1kostrovým skinningem (rotace kolem ramene) | žádný další runtime (Rive/Lottie/Spine), textury 61 kB, přesně sedí na tisk |
| Build | esbuild, 1 soubor app.js | 456 kB gzip místo cca 700 kB nerozbalených modulů |
| Hosting | statický (GitHub Pages / Cloudflare Pages), zatím NEnasazeno | čeká na tvůj souhlas |

## Soubory
- `animation/build_rig.py` – rozřeže kočku na tělo (doplněný trup za tlapkou) a tlapku; `rig.json` (rameno, osa, rozsah −4° až +14°, perioda 1,6 s)
- `animation/preview.py` – náhledové video rigu (`animation/out/preview.mp4`)
- `qr-engine/qr_engine.py` – generátor QR (SVG + PNG, styly square/rounded/dots) + validace 3 dekodéry a 7 zkreslení
- `qr-design/place_qr.py` – automatické umístění QR mimo chráněnou kočku (4 varianty)
- `qr-design/place_qr_behind.py` – QR za kočkou jako na referenci, kočka smí zakrýt max. 12 % symbolu, ne rohové vzory
- `webar/src/main.js`, `webar/src/cat-rig.js`, `webar/public/index.html` – WebAR stránka
- `webar/tools/build_web_assets.py`, `webar/tools/compile-target.mjs` – příprava textur a `target.mind`
- `tests/make_fake_camera.py`, `tests/e2e_webar.mjs` – automatický test s falešnou kamerou

## Výsledky testů (sandbox, ne telefon)
- QR na referenčním PDF: kočka zakrývá 61 % plochy QR, žádný dekodér ho nepřečte.
- QR `HTTPS://MNK.CZ/1`, EC H, verze 2: zxing-cpp přečte ve všech 7 testech; ZBar a WeChat neprojdou 30% náklonem a ZBar šumem.
- Rozvržení „QR za kočkou“ (kočka zakrývá 10,7 % symbolu): čtou všechny 3 dekodéry v měřítku fotky z telefonu, s 15% náklonem i rozmazáním. Ze všech zkoušených poloh prošla jen 1, rezerva je tedy malá.
- Rozvržení s QR vedle kočky: všechny 4 varianty čtou všechny 3 dekodéry.
- E2E (headless Chromium, softwarové WebGL, falešná kamera s tričkem): kočka rozpoznána; kamera připravena 2,9 s po klepnutí, kočka nalezena 1,3 s poté. Animace sedí na tisku, vyvážení bílé podle kamery funguje. FPS 8 je daný softwarovým vykreslováním, na telefonu neměřeno.

## Co nefunguje / není ověřeno
- Nic není ověřené na skutečném iPhonu ani Androidu.
- `MNK.CZ` je zástupná doména, QR se před tiskem musí vygenerovat znovu na skutečnou adresu.
- Video `maneki_neko_alpha.webm` z Disku se nepodařilo stáhnout (8 MB přes konektor). WebM s průhledností navíc Safari na iPhonu pravděpodobně nezobrazí průhledně (k ověření).
- Barva tisku na látce, praní a prohnutí trička nejsou otestované.
