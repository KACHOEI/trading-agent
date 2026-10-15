import os, ccxt, requests, heapq, json, time
import pandas as pd, numpy as np
from datetime import datetime, timezone
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view

TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
TG_CHAT_ID   = os.getenv("TG_CHAT_ID")
STATE_FILE = Path("state/ipae_xrp.json")

SYM = "XRP/USDT"
ANCHOR = "BTC/USDT"
EXEC_TF = "15m"; CTX_TF = "1h"; BIAS_TF = "4h"
LOOKBACK_DAYS = 30
FEE = 0.0005; SLIP = 0.0005

SW_LEFT=5; SW_RIGHT=5; ATR_LEN=14; MAX_ZONES=5
MIN_FVG_MULT=0.5; MAX_OB_DIST=15
CTX_LOOKBACK=40
DQ_MIN=0.70; DQ_STRONG=1.00
PD_DISC=0.45; PD_PREM=0.55
RR_HOMU=1.8; RR_COUNTER=2.5
SCORE_MIN=55; SCORE_COUNTER=65
SWEEP_MIN_ATR=0.10
MAX_HOLD=96
RISK_PCT=0.0075
COOLDOWN_HOURS=24

ex = ccxt.mexc({"enableRateLimit": True, "timeout": 30000})
try: ex.load_markets()
except Exception as e: print("load_markets warn:", type(e).__name__)

def ts_str(ms): return datetime.fromtimestamp(ms/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def send_tg(text):
    if not TG_BOT_TOKEN or not TG_CHAT_ID: print("(no tg)"); return
    try:
        r=requests.post(f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage",
                        json={"chat_id":TG_CHAT_ID,"text":str(text)[:3900]},timeout=15)
        print("tg:",r.status_code)
    except Exception as e: print("tg err",type(e).__name__)
    time.sleep(1.0)

def load_state():
    try:
        s=json.loads(STATE_FILE.read_text(encoding="utf-8")) if STATE_FILE.exists() else {}
    except Exception: s={}
    s.setdefault("open",{}); s.setdefault("history",[]); s.setdefault("last_exit",{}); s.setdefault("last_heartbeat","")
    return s

def save_state(s):
    s["history"]=s.get("history",[])[-500:]
    STATE_FILE.parent.mkdir(parents=True,exist_ok=True)
    STATE_FILE.write_text(json.dumps(s,indent=2,default=str),encoding="utf-8")

def fetch(sym, tf, days):
    since=ex.milliseconds()-int(days*86400000); out=[]; b=0
    while b<40:
        try: r=ex.fetch_ohlcv(sym,timeframe=tf,since=since,limit=500)
        except Exception as e: print("fetch err",sym,tf,type(e).__name__); break
        if not r: break
        out+=r; b+=1; since=r[-1][0]+1
        if len(r)<500: break
    df=pd.DataFrame([x[:6] for x in out],columns=["t","o","h","l","c","v"]).astype(float)
    df=df.drop_duplicates("t").sort_values("t").reset_index(drop=True)
    tfms={"15m":900000,"1h":3600000,"4h":14400000}
    cutoff=ex.milliseconds()-tfms.get(tf,3600000)
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

def compute_context(df):
    n=len(df); h=df["h"].values; l=df["l"].values; c=df["c"].values
    trend=np.zeros(n,int)
    if n<20: return trend, np.full(n,0.5)
    ph,pl=pivots(h,SW_LEFT,SW_RIGHT)
    swH=np.nan; swHD=True; swL=np.nan; swLD=True; tr=0
    for t in range(n):
        pb=t-SW_RIGHT
        if pb>=0:
            if ph[pb]: swH=h[pb]; swHD=False
            if pl[pb]: swL=l[pb]; swLD=False
        bu=(not swHD) and (not np.isnan(swH)) and c[t]>swH
        bd=(not swLD) and (not np.isnan(swL)) and c[t]<swL
        if bu: tr=1; swHD=True
        if bd: tr=-1; swLD=True
        trend[t]=tr
    rh=pd.Series(h).rolling(CTX_LOOKBACK).max().shift(1).values
    rl=pd.Series(l).rolling(CTX_LOOKBACK).min().shift(1).values
    denom=np.where((rh-rl)>1e-12, rh-rl, np.nan)
    with np.errstate(invalid="ignore",divide="ignore"):
        pdv=np.where(np.isnan(denom),0.5,(c-rl)/denom)
    return trend, pdv

def compute_micro(df):
    n=len(df)
    h=df["h"].values; l=df["l"].values; c=df["c"].values; o=df["o"].values
    atr=atr_series(df["h"],df["l"],df["c"],ATR_LEN).values
    is_bull=c>o; is_bear=c<o
    ph,pl=pivots(h,SW_LEFT,SW_RIGHT)
    trend=np.zeros(n,int); bullN=np.zeros(n,int); bearN=np.zeros(n,int)
    swLow=np.full(n,np.nan); swHigh=np.full(n,np.nan)
    sweepUp=np.zeros(n,bool); sweepDn=np.zeros(n,bool); dq=np.zeros(n)
    swH=np.nan; swHD=True; swL=np.nan; swLD=True; tr=0
    lbb=None; lbh=np.nan; lbl=np.nan; lub=None; luh=np.nan; lul=np.nan
    uB=None; uU=None; bOB=[]; rOB=[]; bFV=[]; rFV=[]
    lastSwL=np.nan; lastSwH=np.nan
    for t in range(n):
        if np.isnan(atr[t]) or atr[t]<=0:
            trend[t]=tr; bullN[t]=len(bOB)+len(bFV); bearN[t]=len(rOB)+len(rFV)
            swLow[t]=lastSwL; swHigh[t]=lastSwH; continue
        pb=t-SW_RIGHT
        if pb>=0:
            if ph[pb]: swH=h[pb]; swHD=False; lastSwH=h[pb]
            if pl[pb]: swL=l[pb]; swLD=False; lastSwL=l[pb]
        bu=(not swHD) and (not np.isnan(swH)) and c[t]>swH
        bd=(not swLD) and (not np.isnan(swL)) and c[t]<swL
        if bu: tr=1; swHD=True
        if bd: tr=-1; swLD=True
        if not np.isnan(lastSwL) and l[t]<lastSwL and c[t]>lastSwL and (lastSwL-l[t])>=SWEEP_MIN_ATR*atr[t]: sweepUp[t]=True
        if not np.isnan(lastSwH) and h[t]>lastSwH and c[t]<lastSwH and (h[t]-lastSwH)>=SWEEP_MIN_ATR*atr[t]: sweepDn[t]=True
        if bu and lbb is not None and lbb!=uB and (t-lbb)<=MAX_OB_DIST:
            bOB.append((lbh,lbl,lbb)); uB=lbb
            if len(bOB)>MAX_ZONES: bOB.pop(0)
        if bd and lub is not None and lub!=uU and (t-lub)<=MAX_OB_DIST:
            rOB.append((luh,lul,lub)); uU=lub
            if len(rOB)>MAX_ZONES: rOB.pop(0)
        ms=atr[t]*MIN_FVG_MULT
        if t>=2:
            if l[t]>h[t-2] and (l[t]-h[t-2])>ms: bFV.append((l[t],h[t-2])); bFV=bFV[-MAX_ZONES:]
            if h[t]<l[t-2] and (l[t-2]-h[t])>ms: rFV.append((l[t-2],h[t])); rFV=rFV[-MAX_ZONES:]
        dq[t]=abs(c[t]-o[t])/atr[t]
        if is_bear[t]: lbb=t; lbh=h[t]; lbl=l[t]
        if is_bull[t]: lub=t; luh=h[t]; lul=l[t]
        bOB=[z for z in bOB if not c[t]<z[1]]; rOB=[z for z in rOB if not c[t]>z[0]]
        bFV=[z for z in bFV if not c[t]<z[1]]; rFV=[z for z in rFV if not c[t]>z[0]]
        trend[t]=tr; bullN[t]=len(bOB)+len(bFV); bearN[t]=len(rOB)+len(rFV)
        swLow[t]=lastSwL; swHigh[t]=lastSwH
    return dict(trend=trend,bullN=bullN,bearN=bearN,swLow=swLow,swHigh=swHigh,
                sweepUp=sweepUp,sweepDn=sweepDn,dq=dq,atr=atr)

def score_and_gate(sig_dir, t, m, ctx):
    sc=0; reasons=[]
    bias=ctx["trend4h"]; loc=ctx["pd1h"]; t1h=ctx["trend1h"]
    btc1=ctx["btc1h_t"]; btc1dq=ctx["btc1h_dq"]
    if sig_dir==1 and btc1==-1 and btc1dq>=1.0: return 0,True,"BTC strong down blocks long"
    if sig_dir==-1 and btc1==1 and btc1dq>=1.0: return 0,True,"BTC strong up blocks short"
    if bias==sig_dir: sc+=15; reasons.append("4H+")
    elif bias==-sig_dir: reasons.append("counter")
    if sig_dir==1 and loc<=PD_DISC: sc+=10; reasons.append("disc+")
    elif sig_dir==-1 and loc>=PD_PREM: sc+=10; reasons.append("prem+")
    elif PD_DISC<loc<PD_PREM: sc-=5
    if t1h==sig_dir: sc+=8; reasons.append("1H+")
    bz=m["bullN"][t] if sig_dir==1 else m["bearN"][t]
    if bz>=2: sc+=10; reasons.append(f"z{bz}")
    elif bz==1: sc+=5
    if (sig_dir==1 and m["sweepUp"][t]) or (sig_dir==-1 and m["sweepDn"][t]): sc+=12; reasons.append("sweep+")
    d=m["dq"][t]
    if d>=DQ_STRONG: sc+=10; reasons.append("DQ+")
    elif d>=DQ_MIN: sc+=5
    else: return 0,True,"weak disp"
    thr=SCORE_COUNTER if bias==-sig_dir else SCORE_MIN
    if sc<thr: return sc,True,f"score {sc}<{thr}"
    return sc,False,";".join(reasons)

def map_ctx(htf_t, htf_time, n, tv):
    out=np.zeros(n,int); cur=0; j=0
    for i in range(n):
        while j<len(htf_time) and htf_time[j]<=tv[i]: cur=htf_t[j]; j+=1
        out[i]=cur
    return out

def map_ctx_f(htf_v, htf_time, n, tv):
    out=np.full(n,0.5); cur=0.5; j=0
    for i in range(n):
        while j<len(htf_time) and htf_time[j]<=tv[i]: cur=htf_v[j]; j+=1
        out[i]=cur
    return out

def evaluate_last_bar(df15, m, c1h_t, c1h_pd, c4h_t, cb1h_t, cb1h_dq):
    n=len(df15); t=n-1
    c=df15["c"].values; atr=m["atr"]
    if np.isnan(atr[t]) or atr[t]<=0: return None
    d=m["trend"][t]; prev_d=m["trend"][t-1]
    chochUp=(prev_d==-1 and d==1); chochDn=(prev_d==1 and d==-1)
    cand=[]
    if chochUp: cand.append(("C",1))
    if chochDn: cand.append(("C",-1))
    if m["sweepUp"][t]: cand.append(("B",1))
    if m["sweepDn"][t]: cand.append(("B",-1))
    if c1h_t[t]==1 and d>=0 and m["dq"][t]>=DQ_MIN and c1h_pd[t]<=PD_DISC+0.1: cand.append(("A",1))
    if c1h_t[t]==-1 and d<=0 and m["dq"][t]>=DQ_MIN and c1h_pd[t]>=PD_PREM-0.1: cand.append(("A",-1))
    if not cand: return None
    best=None
    for setup,dirn in cand:
        ctx=dict(trend1h=int(c1h_t[t]),pd1h=float(c1h_pd[t]),trend4h=int(c4h_t[t]),
                 btc1h_t=int(cb1h_t[t]),btc1h_dq=float(cb1h_dq[t]))
        sc,blocked,why=score_and_gate(dirn,t,m,ctx)
        if blocked: continue
        if dirn==1:
            sl_ref=m["swLow"][t]
            sl=(sl_ref-0.15*atr[t]) if not np.isnan(sl_ref) else c[t]-1.2*atr[t]
            tp_ref=m["swHigh"][t]
            tp=(tp_ref+0.1*atr[t]) if (not np.isnan(tp_ref) and tp_ref>c[t]+1.0*atr[t]) else c[t]+2.5*atr[t]
        else:
            sl_ref=m["swHigh"][t]
            sl=(sl_ref+0.15*atr[t]) if not np.isnan(sl_ref) else c[t]+1.2*atr[t]
            tp_ref=m["swLow"][t]
            tp=(tp_ref-0.1*atr[t]) if (not np.isnan(tp_ref) and tp_ref<c[t]-1.0*atr[t]) else c[t]-2.5*atr[t]
        risk=abs(c[t]-sl); rew=abs(tp-c[t])
        if risk<=0: continue
        rr=rew/risk; bias=c4h_t[t]
        need_rr=RR_COUNTER if bias==-dirn else RR_HOMU
        if rr<need_rr: continue
        if best is None or sc>best[0]: best=(sc,dirn,sl,tp,setup,rr)
    if best is None: return None
    sc,dirn,sl,tp,setup,rr=best
    return dict(dir=dirn,entry=float(c[t]),sl=float(sl),tp=float(tp),
                entry_time=float(df15["t"].iloc[t]),setup=setup,score=int(sc),rr=float(rr),
                nZ=int(m["bullN"][t] if dirn==1 else m["bearN"][t]))

def manage_open(df15, state):
    pos=state["open"].get(SYM)
    if not pos: return None
    h=df15["h"].values; l=df15["l"].values; c=df15["c"].values; tv=df15["t"].values
    et=pos["entry_time"]
    idxs=np.where(tv>et)[0]
    for j in idxs:
        exit_px=None; reason=None
        if pos["dir"]==1:
            if l[j]<=pos["sl"]: exit_px=pos["sl"]; reason="SL"
            elif h[j]>=pos["tp"]: exit_px=pos["tp"]; reason="TP"
            elif tv[j]>=et+MAX_HOLD*900000: exit_px=c[j]; reason="TIME"
        else:
            if h[j]>=pos["sl"]: exit_px=pos["sl"]; reason="SL"
            elif l[j]<=pos["tp"]: exit_px=pos["tp"]; reason="TP"
            elif tv[j]>=et+MAX_HOLD*900000: exit_px=c[j]; reason="TIME"
        if exit_px is not None:
            entry=pos["entry"]; sl=pos["sl"]; d=pos["dir"]
            risk=abs(entry-sl)/entry if entry else 0.0
            if d==1: ret=(exit_px*(1-FEE-SLIP)-entry*(1+FEE+SLIP))/entry
            else: ret=(entry*(1+FEE+SLIP)-exit_px*(1-FEE-SLIP))/entry
            R=ret/risk if risk>0 else 0.0
            tr=dict(sym=SYM,dir=d,entry=entry,sl=sl,tp=pos["tp"],exit=float(exit_px),
                    entry_time=et,exit_time=float(tv[j]),reason=reason,R=float(R),ret=float(ret))
            state["history"].append(tr); state["open"].pop(SYM,None)
            state.setdefault("last_exit",{})[SYM]=float(tv[j])
            side="LONG" if d==1 else "SHORT"
            col="✅" if reason=="TP" else ("🛑" if reason=="SL" else "⏱")
            dur=(tr["exit_time"]-tr["entry_time"])/3600000.0
            return (f"{col} EXIT {SYM} {side}\nreason {reason}\n"
                    f"entry {entry:.5f} -> exit {exit_px:.5f}\n"
                    f"result {R:+.2f}R | {ret*100:+.2f}%\n"
                    f"dur {dur:.1f}h | {ts_str(tr['exit_time'])}")
    return None

def entry_message(e):
    side="LONG" if e["dir"]==1 else "SHORT"; col="🟢" if e["dir"]==1 else "🔴"
    return (f"{col} {SYM} | {side} | 15m IPAE-lite\n"
            f"setup {e['setup']} | score {e['score']} | zones {e['nZ']}\n"
            f"ENTRY MARKET {e['entry']:.5f}\nSL {e['sl']:.5f}\nTP {e['tp']:.5f}\n"
            f"R:R 1:{e['rr']:.2f}\nrisk {RISK_PCT*100:.2f}% | PAPER (no real money)\n"
            f"{ts_str(e['entry_time'])}")

def heartbeat(state, new, exits):
    now=ex.milliseconds(); today=datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if state.get("last_heartbeat","")==today: return
    recent=[x for x in state["history"] if float(x.get("exit_time",0))>=now-86400000]
    sumR=sum(float(x.get("R",0)) for x in recent)
    msg=(f"HEARTBEAT {SYM} IPAE-lite\n"
         f"scan 15m | new {new} | exits {exits}\n"
         f"open {len(state['open'])} | closed 24h {len(recent)} sum {sumR:+.2f}R\n"
         f"risk {RISK_PCT*100:.2f}% | {ts_str(now)}\n"
         f"status ALIVE | silence = no setup, not failure")
    send_tg(msg); state["last_heartbeat"]=today

def mode_live():
    state=load_state(); new=0; exits=0
    try:
        df15=fetch(SYM,EXEC_TF,LOOKBACK_DAYS)
        df1h=fetch(SYM,CTX_TF,LOOKBACK_DAYS)
        df4h=fetch(SYM,BIAS_TF,LOOKBACK_DAYS)
        b15=fetch(ANCHOR,CTX_TF,LOOKBACK_DAYS)
    except Exception as e:
        print("fetch err",type(e).__name__,str(e)[:120]); return
    if len(df15)<200 or len(df1h)<60 or len(df4h)<30 or len(b15)<60:
        print("data small",len(df15),len(df1h),len(df4h),len(b15)); return
    t1h,pd1h=compute_context(df1h); t4h,_=compute_context(df4h)
    bt1h,_=compute_context(b15)
    batr=atr_series(b15["h"],b15["l"],b15["c"],ATR_LEN).values
    bdq=np.abs(b15["c"].values-b15["o"].values)/np.where(batr>0,batr,1)
    m=compute_micro(df15); n=len(df15); tv=df15["t"].values
    c1h_t=map_ctx(t1h,df1h["t"].values,n,tv)
    c1h_pd=map_ctx_f(pd1h,df1h["t"].values,n,tv)
    c4h_t=map_ctx(t4h,df4h["t"].values,n,tv)
    cb1h_t=map_ctx(bt1h,b15["t"].values,n,tv)
    cb1h_dq=map_ctx_f(bdq,b15["t"].values,n,tv)
    if SYM in state["open"]:
        msg=manage_open(df15,state)
        if msg: exits+=1; send_tg(msg)
    if SYM not in state["open"]:
        le=state.get("last_exit",{}).get(SYM,0); now=ex.milliseconds()
        if now-le >= COOLDOWN_HOURS*3600000:
            sig=evaluate_last_bar(df15,m,c1h_t,c1h_pd,c4h_t,cb1h_t,cb1h_dq)
            if sig:
                state["open"][SYM]=dict(dir=sig["dir"],entry=sig["entry"],sl=sig["sl"],
                                        tp=sig["tp"],entry_time=sig["entry_time"])
                new+=1; send_tg(entry_message(sig))
    heartbeat(state,new,exits)
    save_state(state)
    print(f"LIVE done new={new} exits={exits} open={len(state['open'])}")

if __name__=="__main__":
    print("IPAE AGENT start",datetime.now(timezone.utc).isoformat())
    mode_live(); print("DONE")
