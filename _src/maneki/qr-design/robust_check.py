import json, numpy as np, cv2, sys, os
SYM=float(os.environ.get("SYM",1600)); X0=float(os.environ.get("X0",440)); Y0=float(os.environ.get("Y0",395)); ROT=int(os.environ.get("ROT",3))
sys.path.insert(0,'/home/claude/maneki/qr-engine'); sys.path.insert(0,'/home/claude/maneki/qr-design')
from qr_engine import decoders
def robust(img, text, crop_box=None):
    crop_box = crop_box or (int(Y0-200), int(Y0+SYM+250), int(X0-200), int(X0+SYM+200))
    y0,y1,x0,x1=crop_box; crop=img[y0:y1,x0:x1]; tot=np.zeros(3,int); n=0
    for ppm in (8,11):
        s=ppm/(SYM/57); im=cv2.resize(crop,None,fx=s,fy=s,interpolation=cv2.INTER_AREA); h,w=im.shape[:2]
        for t in (0.06,0.1,0.15,0.2):
            for side in range(4):
                src=np.float32([[0,0],[w,0],[w,h],[0,h]]); dst=src.copy(); k=t*w/2
                if side==0: dst[0,0]+=k; dst[1,0]-=k
                elif side==1: dst[3,0]+=k; dst[2,0]-=k
                elif side==2: dst[0,1]+=k; dst[3,1]-=k
                else: dst[1,1]+=k; dst[2,1]-=k
                v=cv2.warpPerspective(im,cv2.getPerspectiveTransform(src,dst),(w,h),borderValue=(255,255,255))
                r=decoders(v,text); tot+=np.array([int(x) for x in r.values()]); n+=1
    tt=np.zeros(3,int); tn=0
    for seed in (1,2,3,12,21):
        rng=np.random.default_rng(seed)
        fab=np.clip(232+rng.normal(0,6,crop.shape[:2]),0,255).astype(np.float32)[...,None]; fb=fab*(0.12+0.88*crop.astype(np.float32)/255)
        for ppm in (6,9,12):
            s=ppm/(SYM/57); im=cv2.resize(fb,None,fx=s,fy=s,interpolation=cv2.INTER_AREA)
            for light in (0.8,0.55):
                v=cv2.GaussianBlur(np.clip(im*light+rng.normal(0,4,im.shape),0,255).astype(np.uint8),(3,3),0.8)
                r=decoders(v,text); tt+=np.array([int(x) for x in r.values()]); tn+=1
    return tot, n, tt, tn
if __name__=="__main__":
    import qart_layout as L
    L.DOT_R=0.38
    for mk in map(int, sys.argv[1:]):
        r=L.run("https://HEJKLUCI-DEBUG.GITHUB.IO/M/",10,"Q",mk,X0,Y0,int(SYM),dots=True,rot=ROT)
        cv2.imwrite(f'/home/claude/maneki/builds/qart/OsaB_m{mk}_full.png', r["_img"])
        json.dump({"text":r["text"],"marks":r["marks"],"cover":r["cover"]},open(f'/home/claude/maneki/builds/qart/OsaB_m{mk}_full.json','w'))
        print(mk, 'teček', r["marks"], robust(r["_img"], r["text"]), flush=True)
