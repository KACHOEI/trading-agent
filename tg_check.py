import os
import requests

TOKEN = os.getenv("TG_BOT_TOKEN")
CHAT = os.getenv("TG_CHAT_ID")

print("TG CHECK START")
print("token_set:", bool(TOKEN))
print("token_len:", len(TOKEN) if TOKEN else 0)
print("chat_set:", bool(CHAT))
print("chat_value_type:", type(CHAT).__name__)

if not TOKEN or not CHAT:
    print("RESULT: MISSING CREDS")
    raise SystemExit(1)

# 1) validate bot token
try:
    me = requests.get(f"https://api.telegram.org/bot{TOKEN}/getMe", timeout=15)
    print("getMe_status:", me.status_code)
    print("getMe_response:", me.text[:300])
except Exception as e:
    print("getMe_error:", type(e).__name__, str(e)[:200])

# 2) send message
try:
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT,
        "text": "✅ TG CHECK OK from GitHub Actions"
    }
    r = requests.post(url, json=payload, timeout=15)
    print("sendMessage_status:", r.status_code)
    print("sendMessage_response:", r.text[:500])
    if r.status_code == 200:
        print("RESULT: TELEGRAM OK")
    else:
        print("RESULT: TELEGRAM REJECTED")
except Exception as e:
    print("sendMessage_error:", type(e).__name__, str(e)[:300])
    print("RESULT: NETWORK OR REQUEST ERROR")
