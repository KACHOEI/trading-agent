import ccxt, sys
from datetime import datetime, timezone
import pandas as pd, numpy as np
from numpy.lib.stride_tricks import sliding_window_view

SYMS = ["XLM/USDT","BTC/USDT","ETH/USDT","SOL/USDT","BNB/USDT"]
TF = "1h"; DAYS = 30
SW_LEFT=5; SW_RIGHT=5; ATR_LEN=14; MAX_ZONES=5
MIN_FVG_MULT=0.5; MAX_OB_DIST=15
ENTRY_ATR=1.2; SL_BUF_ATR=0.15; TP_ATR_FALLBACK=2.5

ex = ccxt.mexc({"enableRateLimit": True, "timeout": 30000})
ex.load_markets()

def fetch(sym):
    since = ex.milliseconds() - DAYS*86400000
    out=[]; b=0
    while b<10:
        r = ex.fetch_ohlcv(sym, timeframe=TF, since=since, limit=500)
        if not r: break
        out+=r; b+=1; since=r[-1][0]+1
        if len(r)<500: break
    df = pd.DataFrame([x[:6] for x in out], columns=["t","o","h","l","c","v"]).astype(float)
    df = df.drop_duplicates("t").sort_values("t").reset_index(drop=True)
    cutoff = ex.milliseconds() - 3600000
    return df[df["t"]<=cutoff].reset_index(drop=True)

def atr_series(h,l,c,p=14):
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.rolling(p).mean()

def pivots(arr,left,right):
    a=np.array(arr,dtype=float); n=len(a); W=left+right+1
    ph=np.zeros(n,bool); pl=np.zeros(n,bool)
    if n<W: return ph,pl
    win=sliding_window_view(a,W); ctr=left; vals=win[:,ctr]
    mx=win.max(axis=1); mn=win.min(axis=1)
    umax=(win==mx[:,None]).sum(axis=1)==1; umin=(win==mn[:,None]).sum(axis=1)==1
    ph[left:n-right]=(vals==mx)&umax; pl[left:n-right]=(vals==mn)&umin
    return ph,pl

def analyze(df):
    n=len(df)
    if n<80: return None,{"reason":"too few bars","n":n}
    h=df["h"].values; l=df["l"].values; c=df["c"].values; o=df["o"].values
    atr=atr_series(df["h"],df["l"],df["c"],ATR_LEN).values
    is_bull=c>o; is_bear=c<o
    ph,pl=pivots(h,SW_LEFT,SW_RIGHT)
    swH=np.nan; swHD=True; swL=np.nan; swLD=True; trend=0
    lbb=None;lbh=np.nan;lbl=np.nan
    lub=None;luh=np.nan;lul=np.nan
    uB=None;uU=None
    bOB=[];rOB=[];bFV=[];rFV=[]
    entries=[]; diag={}
    for t in range(n):
        if np.isnan(atr[t]): continue
        pb=t-SW_RIGHT
        if pb>=0:
            if ph[pb]: swH=h[pb]; swHD=False
            if pl[pb]: swL=l[pb]; swLD=False
        bosUp=(not swHD) and (not np.isnan(swH)) and c[t]>swH
        bosDn=(not swLD) and (not np.isnan(swL)) and c[t]<swL
        chochUp=bosUp and trend==-1; chochDn=bosDn and trend==1
        if bosUp: trend=1; swHD=True
        if bosDn: trend=-1; swLD=True
        if bosUp and lbb is not None and lbb!=uB and (t-lbb)<=MAX_OB_DIST:
            bOB.append((lbh,lbl,lbb)); uB=lbb
            if len(bOB)>MAX_ZONES: bOB.pop(0)
        if bosDn and lub is not None and lub!=uU and (t-lub)<=MAX_OB_DIST:
            rOB.append((luh,lul,lub)); uU=lub
            if len(rOB)>MAX_ZONES: rOB.pop(0)
        ms=atr[t]*MIN_FVG_MULT
        if t>=2:
            if l[t]>h[t-2] and (l[t]-h[t-2])>ms: bFV.append((l[t],h[t-2],t-2)); bFV=bFV[-MAX_ZONES:]
            if h[t]<l[t-2] and (l[t-2]-h[t])>ms: rFV.append((l[t-2],h[t],t-2)); rFV=rFV[-MAX_ZONES:]
        if is_bear[t]: lbb=t;lbh=h[t];lbl=l[t]
        if is_bull[t]: lub=t;luh=h[t];lul=l[t]
        bOB=[z for z in bOB if not c[t]<z[1]]; rOB=[z for z in rOB if not c[t]>z[0]]
        bFV=[z for z in bFV if not c[t]<z[1]]; rFV=[z for z in rFV if not c[t]>z[0]]
        bull=bOB+bFV; bear=rOB+rFV
        if t==n-1:
            diag={"trend":trend,"chochUp":chochUp,"chochDn":chochDn,
                  "bull":len(bull),"bear":len(bear),"price":float(c[t]),"atr":float(atr[t])}
        if chochUp and bull:
            near=[z for z in bull if z[1]<=c[t]<=z[0]] or [z for z in bull if (c[t]-z[0])<=ENTRY_ATR*atr[t] and z[0]<=c[t]]
            if near:
                sl=min(z[1] for z in near)-SL_BUF_ATR*atr[t]
                above=[z[0] for z in bear if z[0]>c[t]]
                tp=min(above) if above else c[t]+TP_ATR_FALLBACK*atr[t]
                if tp>c[t] and sl<c[t]:
                    entries.append({"bar":t,"dir":1,"entry":float(c[t]),"sl":float(sl),"tp":float(tp),
                                    "nOB":len(bOB),"nFV":len(bFV)})
        elif chochDn and bear:
            near=[z for z in bear if z[1]<=c[t]<=z[0]] or [z for z in bear if (z[1]-c[t])<=ENTRY_ATR*atr[t] and z[1]>=c[t]]
            if near:
                sl=max(z[0] for z in near)+SL_BUF_ATR*atr[t]
                below=[z[1] for z in bull if z[1]<c[t]]
                tp=max(below) if below else c[t]-TP_ATR_FALLBACK*atr[t]
                if tp<c[t] and sl>c[t]:
                    entries.append({"bar":t,"dir":-1,"entry":float(c[t]),"sl":float(sl),"tp":float(tp),
                                    "nOB":len(rOB),"nFV":len(rFV)})
    if not diag:
        diag={"trend":0,"chochUp":False,"chochDn":False,"bull":0,"bear":0,
              "price":float(c[-1]) if n else 0.0,"atr":0.0}
    last=[e for e in entries if e["bar"]==n-1]
    return (last[0] if last else None), diag

print("STRATEGY PROBE |", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
any_sig=False
for s in SYMS:
    try:
        df=fetch(s); sig,diag=analyze(df)
    except Exception as e:
        print(f"{s}: ERROR {type(e).__name__} {str(e)[:120]}"); continue
    if "reason" in diag:
        print(f"{s}: SKIP {diag['reason']} (n={diag.get('n')})"); continue
    tr={1:"UP",-1:"DN",0:"FLAT"}[diag["trend"]]
    if sig:
        any_sig=True
        side="LONG" if sig["dir"]==1 else "SHORT"
        print(f"{s}: SIGNAL {side} @ {sig['entry']:.6f} SL {sig['sl']:.6f} TP {sig['tp']:.6f} | trend={tr} zones={sig['nOB']}OB+{sig['nFV']}FVG")
    else:
        co="chochUp" if diag["chochUp"] else ("chochDn" if diag["chochDn"] else "no-ChoCH")
        print(f"{s}: no-signal | trend={tr} {co} bullZ={diag['bull']} bearZ={diag['bear']} price={diag['price']:.6f}")
print("VERDICT:", "OK - decision engine alive" if any_sig else "OK-but-quiet - engine works, no setup on last closed bar (normal)")
