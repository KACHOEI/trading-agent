import os, ccxt, requests, heapq
import pandas as pd, numpy as np
from datetime import datetime, timezone
from numpy.lib.stride_tricks import sliding_window_view

TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
TG_CHAT_ID   = os.getenv("TG_CHAT_ID")

# Single symbol chosen by walk-forward evidence (most stable edge), not sentiment.
SYMS = ["XRP/USDT"]
ANCHOR = "BTC/USDT"
MONTHS = 12          # expanded so a single symbol still yields statistical N
FEE = 0.0005
SLIP = {"BTC/USDT":0.0002,"XRP/USDT":0.0005}
DEFAULT_SLIP = 0.0005

# ---- Mod3 core ----
SW_LEFT=5; SW_RIGHT=5; ATR_LEN=14; MAX_ZONES=5
MIN_FVG_MULT=0.5; MAX_OB_DIST=15

# ---- IPAE-lite thresholds ----
CTX_LOOKBACK=40
DQ_MIN=0.70; DQ_STRONG=1.00
PD_DISC=0.45; PD_PREM=0.55
RR_HOMU=1.8; RR_COUNTER=2.5
SCORE_MIN=55; SCORE_COUNTER=65   # lowered because fewer factors remain
SWEEP_MIN_ATR=0.10
MAX_HOLD=96

# ---- Risk layer ----
# This is the only change vs prior run: lower position size to control DD.
RISK_PCT = 0.0075
MAX_CONCURRENT = 1
DAILY_STOP_R = -1.5
WEEKLY_STOP_R = -3.0
COOLDOWN_HOURS = 24

ex = ccxt.mexc({"enableRateLimit": True, "timeout": 30000})
try:
    ex.load_markets()
except Exception as e:
    print("load_markets warn:", type(e).__name__)

def slip_for(sym):
    return SLIP.get(sym, DEFAULT_SLIP)

def fetch(sym, tf, months):
    since = ex.milliseconds() - int(months*30*24*3600*1000)
    out=[]; b=0
    while b<160:
        try:
            r = ex.fetch_ohlcv(sym, timeframe=tf, since=since, limit=500)
        except Exception as e:
            print("  fetch err", sym, tf, type(e).__name__)
            break
        if not r:
            break
        out += r
        b += 1
        since = r[-1][0] + 1
        if len(r) < 500:
            break
    df = pd.DataFrame([x[:6] for x in out], columns=["t","o","h","l","c","v"]).astype(float)
    df = df.drop_duplicates("t").sort_values("t").reset_index(drop=True)
    cutoff = ex.milliseconds() - 3600000
    return df[df["t"] <= cutoff].reset_index(drop=True)

def atr_series(h,l,c,p=14):
    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.rolling(p).mean()

def pivots(arr,left,right):
    a=np.array(arr,dtype=float); n=len(a); W=left+right+1
    ph=np.zeros(n,bool); pl=np.zeros(n,bool)
    if n<W:
        return ph,pl
    win=sliding_window_view(a,W); ctr=left; vals=win[:,ctr]
    mx=win.max(axis=1); mn=win.min(axis=1)
    umax=(win==mx[:,None]).sum(axis=1)==1
    umin=(win==mn[:,None]).sum(axis=1)==1
    ph[left:n-right]=(vals==mx)&umax
    pl[left:n-right]=(vals==mn)&umin
    return ph,pl

def compute_context(df):
    n=len(df)
    h=df["h"].values; l=df["l"].values; c=df["c"].values
    trend=np.zeros(n,int)
    if n<20:
        return trend, np.full(n,0.5)
    ph,pl=pivots(h,SW_LEFT,SW_RIGHT)
    swH=np.nan; swHD=True; swL=np.nan; swLD=True; tr=0
    for t in range(n):
        pb=t-SW_RIGHT
        if pb>=0:
            if ph[pb]:
                swH=h[pb]; swHD=False
            if pl[pb]:
                swL=l[pb]; swLD=False
        bu=(not swHD) and (not np.isnan(swH)) and c[t]>swH
        bd=(not swLD) and (not np.isnan(swL)) and c[t]<swL
        if bu:
            tr=1; swHD=True
        if bd:
            tr=-1; swLD=True
        trend[t]=tr
    rh=pd.Series(h).rolling(CTX_LOOKBACK).max().shift(1).values
    rl=pd.Series(l).rolling(CTX_LOOKBACK).min().shift(1).values
    denom=np.where((rh-rl)>1e-12, rh-rl, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        pdv=np.where(np.isnan(denom),0.5,(c-rl)/denom)
    return trend, pdv

def compute_micro(df):
    n=len(df)
    h=df["h"].values; l=df["l"].values; c=df["c"].values; o=df["o"].values
    atr=atr_series(df["h"],df["l"],df["c"],ATR_LEN).values
    is_bull=c>o; is_bear=c<o
    ph,pl=pivots(h,SW_LEFT,SW_RIGHT)
    trend=np.zeros(n,int)
    bullN=np.zeros(n,int); bearN=np.zeros(n,int)
    swLow=np.full(n,np.nan); swHigh=np.full(n,np.nan)
    sweepUp=np.zeros(n,bool); sweepDn=np.zeros(n,bool)
    dq=np.zeros(n)
    swH=np.nan; swHD=True; swL=np.nan; swLD=True; tr=0
    lbb=None; lbh=np.nan; lbl=np.nan
    lub=None; luh=np.nan; lul=np.nan
    uB=None; uU=None
    bOB=[]; rOB=[]; bFV=[]; rFV=[]
    lastSwL=np.nan; lastSwH=np.nan

    for t in range(n):
        if np.isnan(atr[t]) or atr[t]<=0:
            trend[t]=tr
            bullN[t]=len(bOB)+len(bFV)
            bearN[t]=len(rOB)+len(rFV)
            swLow[t]=lastSwL; swHigh[t]=lastSwH
            continue
        pb=t-SW_RIGHT
        if pb>=0:
            if ph[pb]:
                swH=h[pb]; swHD=False; lastSwH=h[pb]
            if pl[pb]:
                swL=l[pb]; swLD=False; lastSwL=l[pb]
        bu=(not swHD) and (not np.isnan(swH)) and c[t]>swH
        bd=(not swLD) and (not np.isnan(swL)) and c[t]<swL
        if bu:
            tr=1; swHD=True
        if bd:
            tr=-1; swLD=True
        if not np.isnan(lastSwL) and l[t]<lastSwL and c[t]>lastSwL and (lastSwL-l[t])>=SWEEP_MIN_ATR*atr[t]:
            sweepUp[t]=True
        if not np.isnan(lastSwH) and h[t]>lastSwH and c[t]<lastSwH and (h[t]-lastSwH)>=SWEEP_MIN_ATR*atr[t]:
            sweepDn[t]=True
        if bu and lbb is not None and lbb!=uB and (t-lbb)<=MAX_OB_DIST:
            bOB.append((lbh,lbl,lbb)); uB=lbb
            if len(bOB)>MAX_ZONES: bOB.pop(0)
        if bd and lub is not None and lub!=uU and (t-lub)<=MAX_OB_DIST:
            rOB.append((luh,lul,lub)); uU=lub
            if len(rOB)>MAX_ZONES: rOB.pop(0)
        ms=atr[t]*MIN_FVG_MULT
        if t>=2:
            if l[t]>h[t-2] and (l[t]-h[t-2])>ms:
                bFV.append((l[t],h[t-2])); bFV=bFV[-MAX_ZONES:]
            if h[t]<l[t-2] and (l[t-2]-h[t])>ms:
                rFV.append((l[t-2],h[t])); rFV=rFV[-MAX_ZONES:]
        dq[t]=abs(c[t]-o[t])/atr[t]
        if is_bear[t]:
            lbb=t; lbh=h[t]; lbl=l[t]
        if is_bull[t]:
            lub=t; luh=h[t]; lul=l[t]
        bOB=[z for z in bOB if not c[t]<z[1]]
        rOB=[z for z in rOB if not c[t]>z[0]]
        bFV=[z for z in bFV if not c[t]<z[1]]
        rFV=[z for z in rFV if not c[t]>z[0]]
        trend[t]=tr
        bullN[t]=len(bOB)+len(bFV)
        bearN[t]=len(rOB)+len(rFV)
        swLow[t]=lastSwL; swHigh[t]=lastSwH
    return dict(trend=trend,bullN=bullN,bearN=bearN,swLow=swLow,swHigh=swHigh,
                sweepUp=sweepUp,sweepDn=sweepDn,dq=dq,atr=atr)

def score_and_gate(sig_dir, t, m, ctx):
    sc=0; reasons=[]
    bias=ctx["trend4h"]; loc=ctx["pd1h"]; t1h=ctx["trend1h"]
    btc1=ctx["btc1h_t"]; btc1dq=ctx["btc1h_dq"]
    # BTC regime block (XRP is an alt; BTC is a genuine anchor here)
    if sig_dir==1 and btc1==-1 and btc1dq>=1.0:
        return 0,True,"BTC strong down blocks long"
    if sig_dir==-1 and btc1==1 and btc1dq>=1.0:
        return 0,True,"BTC strong up blocks short"
    if bias==sig_dir:
        sc+=15; reasons.append("4H+")
    elif bias==-sig_dir:
        reasons.append("counter")
    if sig_dir==1 and loc<=PD_DISC:
        sc+=10; reasons.append("disc+")
    elif sig_dir==-1 and loc>=PD_PREM:
        sc+=10; reasons.append("prem+")
    elif PD_DISC<loc<PD_PREM:
        sc-=5
    if t1h==sig_dir:
        sc+=8; reasons.append("1H+")
    bz=m["bullN"][t] if sig_dir==1 else m["bearN"][t]
    if bz>=2:
        sc+=10; reasons.append(f"z{bz}")
    elif bz==1:
        sc+=5
    if (sig_dir==1 and m["sweepUp"][t]) or (sig_dir==-1 and m["sweepDn"][t]):
        sc+=12; reasons.append("sweep+")
    d=m["dq"][t]
    if d>=DQ_STRONG:
        sc+=10; reasons.append("DQ+")
    elif d>=DQ_MIN:
        sc+=5
    else:
        return 0,True,"weak disp"
    thr=SCORE_COUNTER if bias==-sig_dir else SCORE_MIN
    if sc<thr:
        return sc,True,f"score {sc}<{thr}"
    return sc,False,";".join(reasons)

def run_symbol(sym):
    df15=fetch(sym,"15m",MONTHS)
    df1h=fetch(sym,"1h",MONTHS)
    df4h=fetch(sym,"4h",MONTHS)
    if len(df15)<600 or len(df1h)<150 or len(df4h)<60:
        print(f"{sym}: DATA SMALL 15m={len(df15)} 1h={len(df1h)} 4h={len(df4h)}")
        return []
    t1h,pd1h=compute_context(df1h)
    t4h,_=compute_context(df4h)
    b15=fetch(ANCHOR,"1h",MONTHS) if sym!=ANCHOR else df1h
    bt1h,_=compute_context(b15)
    batr=atr_series(b15["h"],b15["l"],b15["c"],ATR_LEN).values
    bdq=np.abs(b15["c"].values-b15["o"].values)/np.where(batr>0,batr,1)
    m=compute_micro(df15)
    n=len(df15)
    c=df15["c"].values; o=df15["o"].values; h=df15["h"].values; l=df15["l"].values; tv=df15["t"].values
    atr=m["atr"]
    def map_ctx(htf_t,htf_time):
        out=np.zeros(n,int); cur=0; j=0
        for i in range(n):
            while j<len(htf_time) and htf_time[j]<=tv[i]:
                cur=htf_t[j]; j+=1
            out[i]=cur
        return out
    def map_ctx_f(htf_v,htf_time):
        out=np.full(n,0.5); cur=0.5; j=0
        for i in range(n):
            while j<len(htf_time) and htf_time[j]<=tv[i]:
                cur=htf_v[j]; j+=1
            out[i]=cur
        return out
    c1h_t=map_ctx(t1h,df1h["t"].values)
    c1h_pd=map_ctx_f(pd1h,df1h["t"].values)
    c4h_t=map_ctx(t4h,df4h["t"].values)
    cb1h_t=map_ctx(bt1h,b15["t"].values)
    cb1h_dq=map_ctx_f(bdq,b15["t"].values)
    if sym==ANCHOR:
        cb1h_t=np.zeros(n,int); cb1h_dq=np.zeros(n)
    trades=[]; allowed=-1
    for t in range(2,n):
        if np.isnan(atr[t]) or atr[t]<=0: continue
        if t<=allowed: continue
        d=m["trend"][t]; prev_d=m["trend"][t-1]
        chochUp=(prev_d==-1 and d==1); chochDn=(prev_d==1 and d==-1)
        cand=[]
        if chochUp: cand.append(("C",1))
        if chochDn: cand.append(("C",-1))
        if m["sweepUp"][t]: cand.append(("B",1))
        if m["sweepDn"][t]: cand.append(("B",-1))
        if c1h_t[t]==1 and d>=0 and m["dq"][t]>=DQ_MIN and c1h_pd[t]<=PD_DISC+0.1: cand.append(("A",1))
        if c1h_t[t]==-1 and d<=0 and m["dq"][t]>=DQ_MIN and c1h_pd[t]>=PD_PREM-0.1: cand.append(("A",-1))
        if not cand: continue
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
            rr=rew/risk
            bias=c4h_t[t]
            need_rr=RR_COUNTER if bias==-dirn else RR_HOMU
            if rr<need_rr: continue
            if best is None or sc>best[0]:
                best=(sc,dirn,sl,tp,setup,rr)
        if best is None: continue
        sc,dirn,sl,tp,setup,rr=best
        exit_px=None; reason=None; j=t
        for j in range(t+1,n):
            if dirn==1:
                if l[j]<=sl: exit_px=sl;reason="SL";break
                if h[j]>=tp: exit_px=tp;reason="TP";break
                if j-t>=MAX_HOLD: exit_px=c[j];reason="TIME";break
            else:
                if h[j]>=sl: exit_px=sl;reason="SL";break
                if l[j]<=tp: exit_px=tp;reason="TP";break
                if j-t>=MAX_HOLD: exit_px=c[j];reason="TIME";break
        if exit_px is None:
            exit_px=c[n-1]; reason="OPEN"; j=n-1
        sp=slip_for(sym)
        if dirn==1: ret=(exit_px*(1-FEE-sp)-c[t]*(1+FEE+sp))/c[t]
        else: ret=(c[t]*(1+FEE+sp)-exit_px*(1-FEE-sp))/c[t]
        R=ret/(risk/c[t]) if risk>0 else 0.0
        trades.append(dict(sym=sym,setup=setup,dir=dirn,R=float(R),ret=float(ret),
                           entry_time=float(tv[t]),exit_time=float(tv[j]),reason=reason))
        allowed=j
    return trades

def stats(trades, risk_pct=None):
    if not trades:
        return dict(N=0, WR=0.0, PF=0.0, avgR=0.0, DD=0.0)
    if risk_pct is None:
        risk_pct = RISK_PCT
    R=np.array([x["R"] for x in trades])
    tot=len(R)
    wr=float((R>0).sum())/tot*100
    gp=float(R[R>0].sum())
    gl=float(abs(R[R<=0].sum()))
    pf=gp/gl if gl>0 else 99.0
    eq=np.cumprod(1+R*risk_pct)
    pk=np.maximum.accumulate(eq)
    dd=float(((pk-eq)/pk).max()*100) if len(eq) else 0.0
    return dict(N=tot, WR=wr, PF=min(pf,99.0), avgR=float(R.mean()), DD=dd)

def time_chunks(trades,n_chunks=3):
    if not trades:
        return [[] for _ in range(n_chunks)]
    t0=min(x["entry_time"] for x in trades)
    t1=max(x["entry_time"] for x in trades)
    if t1<=t0:
        ch=[[] for _ in range(n_chunks)]
        ch[0]=trades[:]
        return ch
    step=(t1-t0)/float(n_chunks)
    ch=[[] for _ in range(n_chunks)]
    for x in trades:
        i=int((x["entry_time"]-t0)//step) if step>0 else 0
        i=max(0,min(n_chunks-1,i))
        ch[i].append(x)
    return ch

def walkforward(trades):
    if not trades:
        return 0,[]
    ch=time_chunks(trades,3)
    pc=0
    det=[]
    for i,c in enumerate(ch):
        s=stats(c)
        ok=s["N"]>=8 and s["PF"]>=1.0
        if ok:
            pc+=1
        det.append(f"C{i+1}:N{s['N']} PF{s['PF']:.2f}")
    return pc,det

def portfolio_stats(
    trades,
    max_concurrent=MAX_CONCURRENT,
    daily_stop_R=DAILY_STOP_R,
    weekly_stop_R=WEEKLY_STOP_R,
    risk_pct=RISK_PCT,
    cooldown_hours=COOLDOWN_HOURS
):
    if not trades:
        return dict(N=0, WR=0.0, PF=0.0, avgR=0.0, DD=0.0), []

    cand=sorted(trades,key=lambda x:x["entry_time"])
    heap=[]
    accepted=[]

    equity=1.0
    peak=1.0
    dd=0.0

    cur_d=None
    cur_w=None
    daily_R=0.0
    weekly_R=0.0
    d_blk=False
    w_blk=False

    last_exit={}
    cd_ms=cooldown_hours*3600000

    def date_of(ms):
        return datetime.fromtimestamp(ms/1000,tz=timezone.utc).strftime("%Y-%m-%d")

    def week_of(ms):
        dt=datetime.fromtimestamp(ms/1000,tz=timezone.utc)
        iso=dt.isocalendar()
        return f"{iso.year}-{iso.week}"

    i=0
    while i<len(cand) or heap:
        next_entry = cand[i]["entry_time"] if i < len(cand) else float("inf")
        next_exit  = heap[0][0] if heap else float("inf")

        if next_exit <= next_entry:
            xt,R,sym=heapq.heappop(heap)
            d=date_of(xt)
            w=week_of(xt)

            if d != cur_d:
                cur_d=d
                daily_R=0.0
                d_blk=False
            if w != cur_w:
                cur_w=w
                weekly_R=0.0
                w_blk=False

            equity *= (1.0 + R * risk_pct)
            if equity > peak:
                peak = equity
            cur_dd = ((peak - equity) / peak * 100.0) if peak > 0 else 0.0
            if cur_dd > dd:
                dd = cur_dd

            daily_R += R
            weekly_R += R
            last_exit[sym] = xt

            if daily_R <= daily_stop_R:
                d_blk = True
            if weekly_R <= weekly_stop_R:
                w_blk = True
        else:
            tr=cand[i]
            i += 1
            et=tr["entry_time"]
            d=date_of(et)
            w=week_of(et)

            if d != cur_d:
                cur_d=d
                daily_R=0.0
                d_blk=False
            if w != cur_w:
                cur_w=w
                weekly_R=0.0
                w_blk=False

            if d_blk or w_blk:
                continue
            if len(heap) >= max_concurrent:
                continue

            le=last_exit.get(tr["sym"], -1e18)
            if et - le < cd_ms:
                continue

            heapq.heappush(heap, (tr["exit_time"], tr["R"], tr["sym"]))
            accepted.append(tr)

    s=stats(accepted, risk_pct=risk_pct)
    s["DD"]=dd
    return s, accepted

def send(text):
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        print("(no tg)")
        return
    try:
        r=requests.post(
            f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage",
            json={"chat_id":TG_CHAT_ID,"text":str(text)[:3900]},
            timeout=15
        )
        print("tg:", r.status_code)
    except Exception as e:
        print("tg err", type(e).__name__)

if __name__ == "__main__":
    print("IPAE-LITE start", datetime.now(timezone.utc).isoformat())
    all_tr=[]
    lines=[f"IPAE-LITE | {MONTHS}mo | {SYMS[0]} | fees+slip incl. | risk={RISK_PCT*100:.2f}%"]

    for sym in SYMS:
        tr=run_symbol(sym)
        s=stats(tr, risk_pct=0.01)
        lines.append(f"{sym} @1%: N={s['N']} WR={s['WR']:.0f} PF={s['PF']:.2f} aR={s['avgR']:+.2f} DD={s['DD']:.1f}")
        all_tr += tr

    all_tr.sort(key=lambda x:x["entry_time"])

    raw=stats(all_tr, risk_pct=0.01)
    port,acc=portfolio_stats(all_tr)
    skipped=len(all_tr)-len(acc)
    pc,det=walkforward(acc)

    lines.append("")
    lines.append(f"RAW @1%: N={raw['N']} WR={raw['WR']:.1f} PF={raw['PF']:.2f} aR={raw['avgR']:+.3f} DD={raw['DD']:.1f}")
    lines.append(
        f"PORTFOLIO max{MAX_CONCURRENT} daily{DAILY_STOP_R} weekly{WEEKLY_STOP_R} cd{COOLDOWN_HOURS}h risk{RISK_PCT*100:.2f}%: "
        f"N={port['N']} WR={port['WR']:.1f} PF={port['PF']:.2f} aR={port['avgR']:+.3f} DD={port['DD']:.1f} WF={pc}/3 [{', '.join(det)}]"
    )
    lines.append(f"SKIPPED: {skipped}")

    gate = (
        port["N"] >= 30 and
        port["PF"] >= 1.20 and
        port["WR"] >= 45 and
        port["avgR"] >= 0.08 and
        port["DD"] <= 12 and
        pc >= 2
    )

    lines.append("")
    if gate:
        lines.append("PASS - XRP IPAE-lite has measurable edge under 0.75% risk. Next: paper agent.")
    else:
        miss=[]
        if port["N"] < 30:
            miss.append(f"N {port['N']}<30")
        if port["PF"] < 1.20:
            miss.append(f"PF {port['PF']:.2f}<1.20")
        if port["WR"] < 45:
            miss.append(f"WR {port['WR']:.1f}<45")
        if port["avgR"] < 0.08:
            miss.append(f"aR {port['avgR']:+.3f}<0.08")
        if port["DD"] > 12:
            miss.append(f"DD {port['DD']:.1f}>12")
        if pc < 2:
            miss.append(f"WF {pc}<2")
        lines.append("FAIL - " + ", ".join(miss) + ". Do NOT go live.")

    out="\n".join(lines)
    print(out)
    send(out)
    print("DONE")
