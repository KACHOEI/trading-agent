import os, json, time, requests, ccxt
import pandas as pd, numpy as np
from datetime import datetime, timezone
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view

TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
TG_CHAT_ID   = os.getenv("TG_CHAT_ID")
STATE_FILE = Path("state/portfolio.json")

SYMS = ["XLM/USDT","BTC/USDT","XRP/USDT","TRX/USDT","ADA/USDT"]
TF="1h"; DAYS=30
SW_LEFT=5; SW_RIGHT=5; ATR_LEN=14; MAX_ZONES=5
MIN_FVG_MULT=0.5; MAX_OB_DIST=15
ENTRY_ATR=1.2; SL_BUF_ATR=0.15; TP_ATR_FALLBACK=2.5
MAX_HOLD=96; FEE=0.0005; RISK_PCT=0.01

ex = ccxt.mexc({"enableRateLimit": True, "timeout": 30000})
ex.load_markets()

def utc_now_ms(): return ex.milliseconds()
def ts_str(ms): return datetime.fromtimestamp(ms/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def send_tg(text):
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        print("(tg skipped: creds)"); return
    try:
        r = requests.post(f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage",
                          json={"chat_id":TG_CHAT_ID,"text":str(text)[:3900],"parse_mode":"HTML"}, timeout=15)
        print("tg:", r.status_code)
    except Exception as e:
        print("tg err:", type(e).__name__)
    time.sleep(1.2)

def load_state():
    try:
        if STATE_FILE.exists():
            s=json.loads(STATE_FILE.read_text(encoding="utf-8"))
        else: s={}
    except Exception: s={}
    s.setdefault("open",{}); s.setdefault("history",[]); s.setdefault("last_heartbeat","")
    return s

def save_state(s):
    s["history"]=s.get("history",[])[-500:]
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(s, indent=2, default=str), encoding="utf-8")

def fetch(sym):
    since=utc_now_ms()-DAYS*86400000; out=[]; b=0
    while b<10:
        r=ex.fetch_ohlcv(sym, timeframe=TF, since=since, limit=500)
        if not r: break
        out+=r; b+=1; since=r[-1][0]+1
        if len(r)<500: break
    df=pd.DataFrame([x[:6] for x in out], columns=["t","o","h","l","c","v"]).astype(float)
    df=df.drop_duplicates("t").sort_values("t").reset_index(drop=True)
    cutoff=utc_now_ms()-3600000
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
    lbb=None;lbh=np.nan;lbl=np.nan; lub=None;luh=np.nan;lul=np.nan
    uB=None;uU=None; bOB=[];rOB=[];bFV=[];rFV=[]; sig=None; diag={}
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
                        sig={"dir":1,"entry":float(c[t]),"sl":float(sl),"tp":float(tp),
                             "entry_time":float(df["t"].iloc[t]),"nOB":len(bOB),"nFV":len(bFV),
                             "body":float(max(0.0,(c[t]-o[t])/atr[t]))}
            elif chochDn and bear:
                near=[z for z in bear if z[1]<=c[t]<=z[0]] or [z for z in bear if (z[1]-c[t])<=ENTRY_ATR*atr[t] and z[1]>=c[t]]
                if near:
                    sl=max(z[0] for z in near)+SL_BUF_ATR*atr[t]
                    below=[z[1] for z in bull if z[1]<c[t]]
                    tp=max(below) if below else c[t]-TP_ATR_FALLBACK*atr[t]
                    if tp<c[t] and sl>c[t]:
                        sig={"dir":-1,"entry":float(c[t]),"sl":float(sl),"tp":float(tp),
                             "entry_time":float(df["t"].iloc[t]),"nOB":len(rOB),"nFV":len(rFV),
                             "body":float(max(0.0,(o[t]-c[t])/atr[t]))}
    if not diag:
        diag={"trend":0,"chochUp":False,"chochDn":False,"bull":0,"bear":0,
              "price":float(c[-1]) if n else 0.0,"atr":0.0}
    return sig, diag

def build_trade(sym,pos,exit_px,reason,exit_time):
    entry=pos["entry"]; sl=pos["sl"]; d=pos["dir"]
    risk=abs(entry-sl)/entry if entry else 0.0
    if d==1: ret=(exit_px*(1-FEE)-entry*(1+FEE))/entry
    else: ret=(entry*(1+FEE)-exit_px*(1-FEE))/entry
    R=ret/risk if risk>0 else 0.0
    return {"symbol":sym,"dir":d,"entry":entry,"sl":sl,"tp":pos["tp"],"exit":float(exit_px),
            "entry_time":pos["entry_time"],"exit_time":float(exit_time),"reason":reason,
            "R":float(R),"ret":float(ret)}

def manage_open(sym, df, state):
    pos=state["open"].get(sym)
    if not pos: return None
    h=df["h"].values; l=df["l"].values; c=df["c"].values; tv=df["t"].values
    et=pos["entry_time"]
    idxs=np.where(tv>et)[0]
    for j in idxs:
        exit_px=None; reason=None
        if pos["dir"]==1:
            if l[j]<=pos["sl"]: exit_px=pos["sl"]; reason="SL"
            elif h[j]>=pos["tp"]: exit_px=pos["tp"]; reason="TP"
            elif tv[j]>=et+MAX_HOLD*3600000: exit_px=c[j]; reason="TIME"
        else:
            if h[j]>=pos["sl"]: exit_px=pos["sl"]; reason="SL"
            elif l[j]<=pos["tp"]: exit_px=pos["tp"]; reason="TP"
            elif tv[j]>=et+MAX_HOLD*3600000: exit_px=c[j]; reason="TIME"
        if exit_px is not None:
            tr=build_trade(sym,pos,exit_px,reason,float(tv[j]))
            state["history"].append(tr); state["open"].pop(sym,None)
            side="LONG" if tr["dir"]==1 else "SHORT"
            col="✅" if reason=="TP" else ("🛑" if reason=="SL" else "⏱")
            dur=(tr["exit_time"]-tr["entry_time"])/3600000.0
            return (f"{col} <b>EXIT {sym} {side}</b>\nدلیل: {reason}\n"
                    f"Entry {tr['entry']:.6f} → Exit {tr['exit']:.6f}\n"
                    f"نتیجه: {tr['R']:+.2f}R | {tr['ret']*100:+.2f}%\n"
                    f"مدت: {dur:.1f}h | {ts_str(tr['exit_time'])}")
    return None

def entry_message(sym,e):
    side="LONG" if e["dir"]==1 else "SHORT"; col="🟢" if e["dir"]==1 else "🔴"
    rr=abs(e["tp"]-e["entry"])/max(abs(e["entry"]-e["sl"]),1e-9)
    sc=40
    if e["nOB"]>0 and e["nFV"]>0: sc+=25
    elif (e["nOB"]+e["nFV"])>0: sc+=12
    tz=e["nOB"]+e["nFV"]; sc+= 15 if tz>=3 else (10 if tz==2 else 5)
    b=e.get("body",0); sc+= 15 if b>=1.0 else (8 if b>=0.6 else (4 if b>=0.35 else 0))
    sc=max(50,min(95,sc))
    return (f"{col} <b>{sym} | {side} | 1h Mod3</b>\n🧮 ساختاری: {sc}%\n"
            f"🎯 ورود MARKET: {e['entry']:.6f}\n🛑 SL: {e['sl']:.6f}\n🎯 TP: {e['tp']:.6f}\n"
            f"📊 R:R = 1:{rr:.2f}\n🧱 زون: {e['nOB']}OB + {e['nFV']}FVG\n"
            f"💪 بدنه: {e['body']:.2f}×ATR\n⏱ {ts_str(e['entry_time'])}\n"
            f"⚠️ ریسک ۱٪ | paper track")

def mode_live():
    state=load_state(); now=utc_now_ms()
    today=datetime.now(timezone.utc).strftime("%Y-%m-%d")
    new=0; exits=0; msgs=[]
    for sym in SYMS:
        try: df=fetch(sym)
        except Exception as e:
            msgs.append(f"{sym}: FETCH ERR {type(e).__name__}"); continue
        if len(df)<80:
            msgs.append(f"{sym}: skip (n={len(df)})"); continue
        if sym in state["open"]:
            m=manage_open(sym,df,state)
            if m: exits+=1; msgs.append(m); send_tg(m)
        if sym not in state["open"]:
            sig,diag=analyze(df)
            if "reason" in diag:
                msgs.append(f"{sym}: {diag['reason']}"); continue
            if sig:
                state["open"][sym]={"dir":sig["dir"],"entry":sig["entry"],"sl":sig["sl"],
                                    "tp":sig["tp"],"entry_time":sig["entry_time"]}
                new+=1; m=entry_message(sym,sig); msgs.append(m); send_tg(m)
            else:
                tr={1:"UP",-1:"DN",0:"FLAT"}[diag["trend"]]
                co="chochUp" if diag["chochUp"] else ("chochDn" if diag["chochDn"] else "no-ChoCH")
                msgs.append(f"{sym}: no-signal trend={tr} {co} bullZ={diag['bull']} bearZ={diag['bear']} px={diag['price']:.6f}")
    if state.get("last_heartbeat","")!=today:
        recent=[x for x in state["history"] if float(x.get("exit_time",0))>=now-86400000]
        sumR=sum(float(x.get("R",0)) for x in recent)
        hb=(f"🫀 <b>گزارش سلامت ۲۴ ساعته</b>\nنماد: {len(SYMS)}\nسیگنال این اجرا: {new}\n"
            f"خروج این اجرا: {exits}\nپوزیشن باز: {len(state['open'])}\n"
            f"بسته ۲۴س: {len(recent)} | مجموع {sumR:+.2f}R\n"
            f"⏱ {ts_str(now)}\nوضعیت: ✅ زنده | سکوت = نبود ستاپ، نه خرابی")
        msgs.append(hb); send_tg(hb); state["last_heartbeat"]=today
    save_state(state)
    print("\n".join(msgs))
    print(f"LIVE done new={new} exits={exits} open={len(state['open'])}")

if __name__=="__main__":
    print("Start", datetime.now(timezone.utc).isoformat())
    mode_live()
