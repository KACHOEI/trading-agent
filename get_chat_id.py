import os, requests
TOKEN = os.getenv("TG_BOT_TOKEN")
r = requests.get(f"https://api.telegram.org/bot{TOKEN}/getUpdates", timeout=15)
data = r.json()
print("status:", r.status_code)
if not data.get("ok"):
    print("ERROR:", data); raise SystemExit(1)
ups = data.get("result", [])
print("updates_count:", len(ups))
cid = None
for u in reversed(ups):
    m = u.get("message") or u.get("my_chat_member") or {}
    c = m.get("chat", {})
    if c.get("type") == "private":
        cid = c.get("id"); break
print("PRIVATE_CHAT_ID:", cid if cid else "NOT_FOUND")
if not cid:
    print("ACTION: go to your bot in Telegram, send /start, then re-run this workflow.")
