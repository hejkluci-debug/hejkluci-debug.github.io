import sys, numpy as np
sys.path.insert(0,'/home/claude/maneki/qr-engine'); sys.path.insert(0,'.')
import qart_search as S
from qart import QArtEncoder, codeword_map
geoms=[(340,520,1540),(300,560,1600),(380,480,1500)]
for url in ["HTTPS://HEJKLUCI-DEBUG.GITHUB.IO/M/","HTTPS://MNKO.GITHUB.IO","HTTPS://MNK.CZ"]:
  for v,e in ((9,'M'),(10,'M'),(11,'M'),(12,'M'),(10,'L'),(12,'L')):
    enc=QArtEncoder(url,v,e); cm=codeword_map(enc)
    fixedcw=len([c for c in range(cm[0].max()+1) if cm[2][c] and cm[3][c] not in set(np.array(enc.free)//8)])
    best=None
    for g in geoms:
      for mk in range(8):
        r=S.stage1(enc,cm,mk,*g)
        if r and (best is None or r['use']<best['use']): best={**r,'mask':mk,'g':g}
    print(f"{url[8:]:28s} v{v}{e} fixedCW {fixedcw:2d} blocks {len(cm[4])} t {cm[5]//2} -> best fixed_bad {best['fixed_bad']} use {best['use']:.2f} cover {best['cover']:.2f}", flush=True)
