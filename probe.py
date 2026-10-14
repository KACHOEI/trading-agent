import ccxt, sys
from datetime import datetime, timezone

try:
    ex = ccxt.mexc({"enableRateLimit": True, "timeout": 30000})
    ex.load_markets()
    sym, tf, days = "XLM/USDT", "1h", 30
    since = ex.milliseconds() - days * 86400000
    out, batches = [], 0
    while batches < 10:
        b = ex.fetch_ohlcv(sym, timeframe=tf, since=since, limit=500)
        if not b:
            break
        out += b
        batches += 1
        since = b[-1][0] + 1
        if len(b) < 500:
            break
    print("PROBE RESULT")
    print("bars =", len(out))
    print("batches =", batches)
    if out:
        f = datetime.fromtimestamp(out[0][0] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        l = datetime.fromtimestamp(out[-1][0] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        print("range =", f, "->", l)
        ok = len(out) >= days * 20
        print("VERDICT:", "OK - pagination works, data source is solid" if ok else "FAIL - too few bars, pagination broken")
        sys.exit(0 if ok else 1)
    else:
        print("VERDICT: FAIL - no data returned")
        sys.exit(1)
except Exception as e:
    print("PROBE RESULT")
    print("VERDICT: FAIL - network/exchange error:", type(e).__name__, str(e)[:200])
    sys.exit(1)
