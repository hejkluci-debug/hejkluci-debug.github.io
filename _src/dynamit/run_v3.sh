#!/bin/bash
# usage: run_v2.sh <version> <side> <x0> <y0> <rot> <masks...>
cd /home/claude/dynamit2
v=$1; side=$2; x0=$3; y0=$4; rot=$5; shift 5
for mk in "$@"; do
ART=${ARTF:-in/obrazek.png} ATOPONE=1 BGC=210,45,143 STYLE=lightdots QZ=0 FTOP=1 ATOP=1 FSOLID=1 FINDERC=32,36,46 DOTSIZE=0.76 DOTDARK=32,36,46 \
PALETTE="254,183,153;254,162,122;199,177,243;227,232,236;2,163,254" PALW="40,25,20,10,5" python3 -c "
import dyn_layout as L, cv2, json
from dyn_search import fitness
v=$v; n=4*v+17
r=L.run('https://HEJKLUCI-DEBUG.GITHUB.IO/D/',v,'Q',$mk,$x0,$y0,$side,dots=True,rot=$rot)
f,res=fitness(r['_img'],r['text'],$x0,$y0,$side,n)
print('v',v,'side $side rot $rot mask $mk cover',r['cover'],'dots',r['marks'],'fit',f, flush=True)
cv2.imwrite(f'builds/v3_v{v}_s${side}_r${rot}_m$mk.png', r['_img']); json.dump({'text':r['text'],'marks':r['marks'],'cover':r['cover'],'fit':f,'res':res},open(f'builds/v3_v{v}_s${side}_r${rot}_m$mk.json','w'))
" 2>&1 | grep -v "WARN\|i="
done
