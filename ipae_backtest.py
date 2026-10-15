import os, ccxt, requests
import pandas as pd, numpy as np
from datetime import datetime, timezone
from numpy.lib.stride_tricks import sliding_window_view

TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
TG_CHAT_ID   = os.getenv("TG_CHAT_ID")

SYMS = ["XLM/USDT","BTC/USDT","XRP/USDT","TRX/USDT","ADA/USDT"]
ANCHOR = "BTC/USDT"
MONTHS = 6
FEE = 0.0005
SLIP = {"BTC/USDT":0.0002}   # بقیه 0.0005
DEFAULT_SLIP = 0.0005

# ---- Mod3 core ----
SW_LEFT=5; SW_RIGHT=5; ATR_LEN=14; MAX_ZONES=5
MIN_FVG_MULT=0.5; MAX_OB_DIST=15
# ---- IPAE thresholds ----
CTX_LOOKBACK=40          # dealing range on 1h
SER_MIN=0.55             # efficiency of trigger candle
DQ_MIN=0.70              # displacement body/ATR
DQ_STRONG=1.00
PD_DISC=0.45; PD_PREM=0.55
RR_HOMU=1.8; RR_COUNTER=2.5
SCORE_MIN=70; SCORE_COUNTER=80
SWEEP_MIN_ATR=0.10
MAX_HOLD=96              # 15m bars (=24h)
KILLZONES=[(7,12),(12,16)]   # London, NY (UTC hour ranges)

ex = ccxt.mexc({"enableRateLimit": True, "timeout": 30000})
ex.load_markets()

def slip_for(sym): return SLIP.get(sym, DEFAULT_SLIP)

def fetch(sym, tf, months):
    since = ex.milliseconds() - int(months*30*24*3600*1000)
    out=[]; b=0
    while b<120:
        try: r = ex.fetch_ohlcv(sym, timeframe=tf, since=since, limit=500)
        except Exception as e:
            print("  fetch err", sym, tf, type(e).__name__); break
        if not r: break
        out += r; b += 1; since = r[-1][0] + 1
        if len(r) < 500: break
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
    if n<W: return ph,pl
    win=sliding_window_view(a,W); ctr=left; vals=win[:,ctr]
    mx=win.max(axis=1); mn=win.min(axis=1)
    umax=(win==mx[:,None]).sum(axis=1)==1; umin=(win==mn[:,None]).sum(axis=1)==1
    ph[left:n-right]=(vals==mx)&umax; pl[left:n-right]=(vals==mn)&umin
    return ph,pl

def compute_context(df):
    """trend (pivot-BOS) + premium/discount over rolling dealing range."""
    n=len(df)
    h=df["h"].values; l=df["l"].values; c=df["c"].values
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
    pdv=np.where(np.isnan(denom),0.5,(c-rl)/denom)
    return trend, pdv

def compute_micro(df):
    """per-bar on 15m: trend, zones, confirmed swings, sweep, displacement, SER, OFP."""
    n=len(df)
    h=df["h"].values; l=df["l"].values; c=df["c"].values; o=df["o"].values; v=df["v"].values
    atr=atr_series(df["h"],df["l"],df["c"],ATR_LEN).values
    is_bull=c>o; is_bear=c<o
    ph,pl=pivots(h,SW_LEFT,SW_RIGHT)
    trend=np.zeros(n,int)
    bullN=np.zeros(n,int); bearN=np.zeros(n,int)
    swLow=np.full(n,np.nan); swHigh=np.full(n,np.nan)
    sweepUp=np.zeros(n,bool); sweepDn=np.zeros(n,bool)
    dq=np.zeros(n); ser=np.zeros(n); ofp=np.zeros(n)
    swH=np.nan; swHD=True; swL=np.nan; swLD=True; tr=0
    lbb=None;lbh=np.nan;lbl=np.nan; lub=None;luh=np.nan;lul=np.nan
    uB=None;uU=None; bOB=[];rOB=[];bFV=[];rFV=[]
    lastSwL=np.nan; lastSwH=np.nan
    for t in range(n):
        if np.isnan(atr[t]) or atr[t]<=0:
            trend[t]=tr; bullN[t]=len(bOB)+len(bFV); bearN[t]=len(rOB)+len(rFV)
            swLow[t]=lastSwL; swHigh[t]=lastSwH
            continue
        pb=t-SW_RIGHT
        if pb>=0:
            if ph[pb]: swH=h[pb]; swHD=False; lastSwH=h[pb]
            if pl[pb]: swL=l[pb]; swLD=False; lastSwL=l[pb]
        bu=(not swHD) and (not np.isnan(swH)) and c[t]>swH
        bd=(not swLD) and (not np.isnan(swL)) and c[t]<swL
        chochUp=bu and tr==-1; chochDn=bd and tr==1
        if bu: tr=1; swHD=True
        if bd: tr=-1; swLD=True
        # sweeps vs confirmed swings
        if not np.isnan(lastSwL) and l[t]<lastSwL and c[t]>lastSwL and (lastSwL-l[t])>=SWEEP_MIN_ATR*atr[t]:
            sweepUp[t]=True
        if not np.isnan(lastSwH) and h[t]>lastSwH and c[t]<lastSwH and (h[t]-lastSwH)>=SWEEP_MIN_ATR*atr[t]:
            sweepDn[t]=True
        # OB
        if bu and lbb is not None and lbb!=uB and (t-lbb)<=MAX_OB_DIST:
            bOB.append((lbh,lbl,lbb)); uB=lbb
            if len(bOB)>MAX_ZONES: bOB.pop(0)
        if bd and lub is not None and lub!=uU and (t-lub)<=MAX_OB_DIST:
            rOB.append((luh,lul,lub)); uU=lub
            if len(rOB)>MAX_ZONES: rOB.pop(0)
        # FVG
        ms=atr[t]*MIN_FVG_MULT
        if t>=2:
            if l[t]>h[t-2] and (l[t]-h[t-2])>ms: bFV.append((l[t],h[t-2])); bFV=bFV[-MAX_ZONES:]
            if h[t]<l[t-2] and (l[t-2]-h[t])>ms: rFV.append((l[t-2],h[t])); rFV=rFV[-MAX_ZONES:]
        # displacement / SER / OFP
        rng=h[t]-l[t]
        ser[t]=abs(c[t]-o[t])/rng if rng>1e-12 else 0.0
        dq[t]=abs(c[t]-o[t])/atr[t]
        ofp[t]=(((c[t]-l[t])-(h[t]-c[t]))/rng*v[t]) if rng>1e-12 else 0.0
        # update opposite candles AFTER OB
        if is_bear[t]: lbb=t;lbh=h[t];lbl=l[t]
        if is_bull[t]: lub=t;luh=h[t];lul=l[t]
        # mitigation
        bOB=[z for z in bOB if not c[t]<z[1]]; rOB=[z for z in rOB if not c[t]>z[0]]
        bFV=[z for z in bFV if not c[t]<z[1]]; rFV=[z for z in rFV if not c[t]>z[0]]
        trend[t]=tr; bullN[t]=len(bOB)+len(bFV); bearN[t]=len(rOB)+len(rFV)
        swLow[t]=lastSwL; swHigh[t]=lastSwH
    return dict(trend=trend,bullN=bullN,bearN=bearN,swLow=swLow,swHigh=swHigh,
                sweepUp=sweepUp,sweepDn=sweepDn,dq=dq,ser=ser,ofp=ofp,atr=atr,
                bOB=bOB,rOB=rOB,bFV=bFV,rFV=rFV)  # last-state zones for SL/TP fallback

def in_killzone(t_ms):
    hr=datetime.fromtimestamp(t_ms/1000,tz=timezone.utc).hour
    return any(a<=hr<b for a,b in KILLZONES)

def atr_pctile(atr_arr):
    a=atr_arr[~np.isnan(atr_arr)]
    if len(a)<20: return 50.0
    cur=a[-1]
    return float((a<cur).mean()*100.0)

def ipae_entries(df15, ctx1h_t, ctx1h_pd, ctx4h_t, btc1h_t, btc1h_dq, btc4h_t):
    n=len(df15)
    if n<120: return []
    m=compute_micro(df15)
    c=df15["c"].values; o=df15["o"].values; h=df15["h"].values; l=df15["l"].values
    tv=df15["t"].values
    atr=m["atr"]
    # align context arrays to 15m via merge_asof on time
    base=pd.DataFrame({"t":tv})
    def asof(arr, name):
        s=pd.DataFrame({"t":ctx_times[name],"v":arr})
        mg=pd.merge_asof(base,s,on="t",direction="backward")
        return mg["v"].fillna(0).values
    # build ctx_times dict from the higher-tf dfs (passed separately) -> handled by caller injecting
    return m, c,o,h,l,tv,atr

def score_and_gate(sig_dir, t, m, ctx, atr_pct):
    """ctx = dict(trend1h, pd1h, trend4h, btc1h_t, btc1h_dq, btc4h_t). returns (score, blocked, reason)."""
    sc=0; reasons=[]
    bias = ctx["trend4h"]
    loc  = ctx["pd1h"]
    t1h  = ctx["trend1h"]
    btc1 = ctx["btc1h_t"]; btc1dq=ctx["btc1h_dq"]; btc4=ctx["btc4h_t"]
    # regime block via BTC
    if sig_dir==1 and btc1==-1 and btc1dq>=1.0:
        return 0,True,"BTC strong down blocks long"
    if sig_dir==-1 and btc1==1 and btc1dq>=1.0:
        return 0,True,"BTC strong up blocks short"
    # bias alignment
    if bias==sig_dir: sc+=15; reasons.append("4H bias+")
    elif bias==-sig_dir:
        sc+=0; reasons.append("counter-4H")
    else: reasons.append("4H flat")
    # location
    if sig_dir==1 and loc<=PD_DISC: sc+=10; reasons.append("discount+")
    elif sig_dir==-1 and loc>=PD_PREM: sc+=10; reasons.append("premium+")
    elif PD_DISC<loc<PD_PREM: sc-=5; reasons.append("neutral-zone-")
    # 1h trend alignment
    if t1h==sig_dir: sc+=8; reasons.append("1H align+")
    # confluence zones
    bz = m["bullN"][t] if sig_dir==1 else m["bearN"][t]
    if bz>=2: sc+=10; reasons.append(f"zones{bz}+")
    elif bz==1: sc+=5
    # sweep
    if (sig_dir==1 and m["sweepUp"][t]) or (sig_dir==-1 and m["sweepDn"][t]):
        sc+=12; reasons.append("sweep+")
    # displacement
    d=m["dq"][t]
    if d>=DQ_STRONG: sc+=10; reasons.append("DQ strong+")
    elif d>=DQ_MIN: sc+=5
    else: return 0,True,"weak displacement"
    # SER
    if m["ser"][t]>=SER_MIN: sc+=7; reasons.append("SER+")
    else: sc-=5
    # OFP alignment
    ofp3 = np.nansum(m["ofp"][max(0,t-2):t+1])
    if (sig_dir==1 and ofp3>0) or (sig_dir==-1 and ofp3<0): sc+=8; reasons.append("OFP+")
    # killzone
    if in_killzone(m["_t"][t]): sc+=7; reasons.append("killzone+")
    # BTC mild counter penalty
    if (sig_dir==1 and btc1==-1) or (sig_dir==-1 and btc1==1): sc-=15; reasons.append("BTC mild-")
    # ATR extremes
    if atr_pct>95: return 0,True,"ATR spike (news/illiquidity)"
    if atr_pct<20: sc-=10; reasons.append("low vol-")
    thr = SCORE_COUNTER if bias==-sig_dir else SCORE_MIN
    if sc<thr: return sc,True,f"score {sc}<{thr}"
    return sc,False,";".join(reasons)

def run_symbol(sym):
    df15=fetch(sym,"15m",MONTHS)
    df1h=fetch(sym,"1h",MONTHS)
    df4h=fetch(sym,"4h",MONTHS)
    if len(df15)<300 or len(df1h)<80 or len(df4h)<40:
        print(f"{sym}: DATA SMALL 15m={len(df15)} 1h={len(df1h)} 4h={len(df4h)}"); return []
    t1h,pd1h=compute_context(df1h)
    t4h,_=compute_context(df4h)
    # anchor
    b15=fetch(ANCHOR,"1h",MONTHS) if sym!=ANCHOR else df1h
    b4h=fetch(ANCHOR,"4h",MONTHS) if sym!=ANCHOR else df4h
    bt1h,_=compute_context(b15); _,_=compute_context(b4h)
    bt4h,_=compute_context(b4h)
    # btc displacement on 1h
    batr=atr_series(b15["h"],b15["l"],b15["c"],ATR_LEN).values
    bdq=np.abs(b15["c"].values-b15["o"].values)/np.where(batr>0,batr,1)
    # micro on 15m
    m=compute_micro(df15)
    n=len(df15)
    c=df15["c"].values;o=df15["o"].values;h=df15["h"].values;l=df15["l"].values;tv=df15["t"].values
    atr=m["atr"]
    m["_t"]=tv
    # map higher-tf context onto 15m by time (backward)
    def map_ctx(htf_t, htf_time):
        out=np.zeros(n,int); cur=0; j=0
        for i in range(n):
            while j<len(htf_time) and htf_time[j]<=tv[i]:
                cur=htf_t[j]; j+=1
            out[i]=cur
        return out
    def map_ctx_f(htf_v, htf_time):
        out=np.full(n,0.5); cur=0.5; j=0
        for i in range(n):
            while j<len(htf_time) and htf_time[j]<=tv[i]:
                cur=htf_v[j]; j+=1
            out[i]=cur
        return out
    c1h_t=map_ctx(t1h, df1h["t"].values)
    c1h_pd=map_ctx_f(pd1h, df1h["t"].values)
    c4h_t=map_ctx(t4h, df4h["t"].values)
    cb1h_t=map_ctx(bt1h, b15["t"].values)
    cb1h_dq=map_ctx_f(bdq, b15["t"].values)
    cb4h_t=map_ctx(bt4h, b4h["t"].values)
    atrpct=atr_pctile(atr)
    # entry loop
    trades=[]; pos=None; allowed=-1
    bull_zones_state=[[] for _ in range(n)]  # we recompute zone edges at entry bar from micro last-state? need per-bar zones
    # For SL/TP we need the zone edges AT bar t; recompute lightweight by storing lists per bar is heavy.
    # Instead: at entry, derive SL/TP from confirmed swing + ATR fallback (structural enough for phase-1).
    for t in range(2,n):
        if np.isnan(atr[t]) or atr[t]<=0: continue
        if t<=allowed: continue
        if pos is not None: continue
        d=m["trend"][t]
        # candidate directions
        cand=[]
        # Setup C: Mod3 ChoCH
        # reconstruct choch flags quickly: compare trend change
        prev_d=m["trend"][t-1]
        chochUp = (prev_d==-1 and d==1)
        chochDn = (prev_d==1 and d==-1)
        # Setup A: continuation pullback -> need price near zone in trend dir; approximate: trend1h==dir and 15m touched zone (bullN>0 for long) and pullback happened (recent lower low for long)
        # Setup B: sweep
        if chochUp: cand.append(("C",1))
        if chochDn: cand.append(("C",-1))
        if m["sweepUp"][t]: cand.append(("B",1))
        if m["sweepDn"][t]: cand.append(("B",-1))
        # continuation: 1h trend dir and 15m currently pulling back into value (pd favorable) and displacement in dir
        if c1h_t[t]==1 and d>=0 and m["dq"][t]>=DQ_MIN and c1h_pd[t]<=PD_DISC+0.1: cand.append(("A",1))
        if c1h_t[t]==-1 and d<=0 and m["dq"][t]>=DQ_MIN and c1h_pd[t]>=PD_PREM-0.1: cand.append(("A",-1))
        if not cand: continue
        # pick best by score
        best=None
        for setup,dirn in cand:
            ctx=dict(trend1h=int(c1h_t[t]),pd1h=float(c1h_pd[t]),trend4h=int(c4h_t[t]),
                     btc1h_t=int(cb1h_t[t]),btc1h_dq=float(cb1h_dq[t]),btc4h_t=int(cb4h_t[t]))
            sc,blocked,why=score_and_gate(dirn,t,m,ctx,atrpct)
            if blocked: continue
            # structural SL/TP
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
            need_rr = RR_COUNTER if bias==-dirn else RR_HOMU
            if rr<need_rr: continue
            if best is None or sc>best[0]:
                best=(sc,dirn,sl,tp,setup,rr)
        if best is None: continue
        sc,dirn,sl,tp,setup,rr=best
        # simulate exit stop-first
        exit_px=None; reason=None; eb=t
        for j in range(t+1,n):
            if dirn==1:
                if l[j]<=sl: exit_px=sl;reason="SL";break
                if h[j]>=tp: exit_px=tp;reason="TP";break
                if j-t>=MAX_HOLD: exit_px=c[j];reason="TIME";break
            else:
                if h[j]>=sl: exit_px=sl;reason="SL";break
                if l[j]<=tp: exit_px=tp;reason="TP";break
                if j-t>=MAX_HOLD: exit_px=c[j];reason="TIME";break
        if exit_px is None: exit_px=c[n-1];reason="OPEN"
        sp=slip_for(sym)
        if dirn==1: ret=(exit_px*(1-FEE-sp)-c[t]*(1+FEE+sp))/c[t]
        else: ret=(c[t]*(1+FEE+sp)-exit_px*(1-FEE-sp))/c[t]
        R=ret/(risk/c[t]) if risk>0 else 0.0
        trades.append(dict(sym=sym,setup=setup,dir=dirn,R=float(R),ret=float(ret),
                           entry_time=float(tv[t]),exit_time=float(tv[min(eb+n-1,n-1)] if reason=="OPEN" else tv[j]),
                           reason=reason))
        allowed=j if reason!="OPEN" else n
    return trades

def stats(trades):
    if not trades: return dict(N=0,WR=0.0,PF=0.0,avgR=0.0,DD=0.0)
    R=np.array([x["R"] for x in trades]); tot=len(R)
    wr=float((R>0).sum())/tot*100
    gp=float(R[R>0].sum()); gl=float(abs(R[R<=0].sum()))
    pf=gp/gl if gl>0 else 99.0
    eq=np.cumprod(1+R*0.01); pk=np.maximum.accumulate(eq)
    dd=float(((pk-eq)/pk).max()*100) if len(eq) else 0.0
    return dict(N=tot,WR=wr,PF=min(pf,99.0),avgR=float(R.mean()),DD=dd)

def walkforward(trades, times_ref):
    if not trades: return 0,[]
    t0=min(x["exit_time"] for x in trades); t1=max(x["exit_time"] for x in trades)
    step=(t1-t0)/3.0 if t1>t0 else 1.0
    ch=[[],[],[]]
    for x in trades:
        i=int((x["exit_time"]-t0)//step) if step>0 else 0
        i=max(0,min(2,i)); ch[i].append(x)
    pc=0; det=[]
    for i,c in enumerate(ch):
        s=stats(c); ok=s["N"]>=10 and s["PF"]>=1.0
        if ok: pc+=1
        det.append(f"C{i+1}:N{s['N']} PF{s['PF']:.2f}")
    return pc,det

def send(text):
    if not TG_BOT_TOKEN or not TG_CHAT_ID: print("(no tg)"); return
    try:
        r=requests.post(f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage",
                        json={"chat_id":TG_CHAT_ID,"text":text[:3900],"parse_mode":"HTML"},timeout=15)
        print("tg:",r.status_code)
    except Exception as e: print("tg err",type(e).__name__)

if __name__=="__main__":
    print("IPAE BACKTEST start", datetime.now(timezone.utc).isoformat())
    all_tr=[]; lines=[f"IPAE BACKTEST | {MONTHS}mo | {len(SYMS)} sym | fees+slip incl."]
    for sym in SYMS:
        tr=run_symbol(sym)
        s=stats(tr)
        lines.append(f"{sym}: N={s['N']} WR={s['WR']:.0f} PF={s['PF']:.2f} aR={s['avgR']:+.2f} DD={s['DD']:.1f}")
        all_tr+=tr
    agg=stats(all_tr)
    pc,det=walkforward(all_tr,None)
    lines.append("")
    lines.append(f"AGGREGATE: N={agg['N']} WR={agg['WR']:.1f} PF={agg['PF']:.2f} aR={agg['avgR']:+.3f} DD={agg['DD']:.1f} WF={pc}/3 [{', '.join(det)}]")
    gate = (agg['N']>=120 and agg['PF']>=1.25 and agg['WR']>=45 and agg['avgR']>=0.10 and agg['DD']<=15 and pc>=2)
    lines.append("")
    if gate:
        lines.append("✅ GATE PASS — IPAE has measurable edge. Next: phase-2 exit + paper agent.")
    else:
        miss=[]
        if agg['N']<120: miss.append(f"N {agg['N']}<120")
        if agg['PF']<1.25: miss.append(f"PF {agg['PF']:.2f}<1.25")
        if agg['WR']<45: miss.append(f"WR {agg['WR']:.0f}<45")
        if agg['avgR']<0.10: miss.append(f"aR {agg['avgR']:+.2f}<0.10")
        if agg['DD']>15: miss.append(f"DD {agg['DD']:.1f}>15")
        if pc<2: miss.append(f"WF {pc}<2")
        lines.append("❌ GATE FAIL — "+", ".join(miss)+". Do NOT go live. Tune or accept lower frequency.")
    out="\n".join(lines)
    print(out); send(out)
    print("DONE")
