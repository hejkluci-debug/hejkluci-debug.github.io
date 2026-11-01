import sys, json
import qart_layout as L
URL = "HTTPS://HEJKLUCI-DEBUG.GITHUB.IO/M/"
cands = [(11, 'M', 200, 180, 1900), (11, 'M', 140, 120, 2000), (11, 'M', 260, 60, 1900), (11, 'Q', 20, 180, 2100)]
for v, e, x0, y0, sym in cands:
    for mk in (0, 3, 6):
        for dots in (False, True):
            r = L.run(URL, v, e, mk, x0, y0, sym, dots=dots, rot=1, save=True)
            print(v, e, mk, x0, y0, sym, 'dots' if dots else 'clean', 'cover', r['cover'], 'score', r['score'], 'quick', r['quick_ok'], 'UEC %.2f' % r['zxing_unused_ec'], 'marks', r['marks'],
                  {k: ''.join('1' if x else '0' for x in t.values()) for k, t in r['tests'].items()}, flush=True)
