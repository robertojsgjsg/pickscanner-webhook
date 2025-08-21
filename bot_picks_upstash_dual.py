
import os, math, datetime as dt, json, hashlib, asyncio, re, urllib.parse
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import httpx
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import redis.asyncio as redis

# ---------- Config via env ----------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
VALUE_BUFFER = float(os.getenv("VALUE_BUFFER", "0.01"))
LEAGUES = os.getenv("LEAGUES","soccer_epl,soccer_spain_la_liga,soccer_italy_serie_a,soccer_germany_bundesliga,soccer_france_ligue_one").split(",")
TOP_K = int(os.getenv("TOP_K", "4"))
TOP_K_PER_DAY = int(os.getenv("TOP_K_PER_DAY", str(TOP_K)))
TZ = ZoneInfo(os.getenv("TZ", "Europe/Madrid"))
DAY_OFFSET_DEFAULT = int(os.getenv("DAY_OFFSET_DEFAULT", "1"))  # 0=hoy, 1=mañana

# Webhook/Render
PORT = int(os.getenv("PORT", "10000"))
EXTERNAL_URL = os.getenv("WEBHOOK_URL") or os.getenv("RENDER_EXTERNAL_URL")

# Memory (Upstash)
REDIS_BACKEND = os.getenv("REDIS_BACKEND", "auto").lower()  # auto|resp|rest
REDIS_URL = os.getenv("REDIS_URL", "")  # rediss://...
REDIS_REST_URL = os.getenv("REDIS_REST_URL", "")  # https://...
REDIS_REST_TOKEN = os.getenv("REDIS_REST_TOKEN", "")  # token de REST
MEMORY_TTL_DAYS = int(os.getenv("MEMORY_TTL_DAYS", "14"))
MEM_NAMESPACE = os.getenv("MEMORY_NAMESPACE", "pickmem")

# Networking
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "10"))
HTTP_CONCURRENCY = int(os.getenv("HTTP_CONCURRENCY", "5"))

# ---------- Async HTTP client ----------
http_limits = httpx.Limits(max_connections=HTTP_CONCURRENCY, max_keepalive_connections=HTTP_CONCURRENCY)
http_client = httpx.AsyncClient(limits=http_limits, timeout=HTTP_TIMEOUT)

# ---------- Redis RESP (async) ----------
rdb: Optional[redis.Redis] = None
async def get_redis_resp() -> redis.Redis:
    global rdb
    if rdb is None:
        if not REDIS_URL:
            raise RuntimeError("Falta REDIS_URL (rediss://...)")
        params = dict(
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=8,
            health_check_interval=30,
            retry_on_timeout=True,
        )
        if REDIS_URL.startswith("rediss://"):
            params.update(ssl=True, ssl_cert_reqs=None)
        rdb = redis.from_url(REDIS_URL, **params)
        try:
            ok = await rdb.ping()
            if not ok:
                raise RuntimeError("PING a Redis RESP falló")
        except Exception as e:
            rdb = None
            raise
    return rdb

# ---------- Redis REST (Upstash) ----------
def rest_url(path: str) -> str:
    base = REDIS_REST_URL.rstrip("/")
    if not base:
        raise RuntimeError("Falta REDIS_REST_URL (https://...)")
    return f"{base}/{path.lstrip('/')}"

def rest_headers() -> dict:
    if not REDIS_REST_TOKEN:
        raise RuntimeError("Falta REDIS_REST_TOKEN")
    return {"Authorization": f"Bearer {REDIS_REST_TOKEN}"}

async def rest_exists(key: str) -> bool:
    url = rest_url(f"exists/{urllib.parse.quote(key, safe='')}")
    r = await http_client.get(url, headers=rest_headers())
    r.raise_for_status()
    data = r.json()
    return (data.get("result", 0) == 1)

async def rest_setex(key: str, ttl_sec: int, value: str) -> None:
    url = rest_url(f"setex/{urllib.parse.quote(key, safe='')}/{ttl_sec}/{urllib.parse.quote(value, safe='')}")
    r = await http_client.get(url, headers=rest_headers())
    r.raise_for_status()

# ---------- Memory facade (auto backend) ----------
_backend: Optional[str] = None  # 'resp' or 'rest'

def mem_key(user_id: int, fp: str) -> str:
    return f"{MEM_NAMESPACE}:u{user_id}:{fp}"

def pick_fingerprint(date_str: str, league: str, home: str, away: str, side: str) -> str:
    try:
        d = dt.datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        dpart = d.isoformat()
    except Exception:
        dpart = date_str[:10]
    raw = f"{dpart}|{league.lower()}|{home.lower()}|{away.lower()}|1x2|{side.lower()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

async def decide_backend():
    global _backend
    if REDIS_BACKEND in ("resp","rest"):
        _backend = REDIS_BACKEND
        return
    try:
        await get_redis_resp()
        _backend = "resp"
    except Exception as e:
        if REDIS_REST_URL and REDIS_REST_TOKEN:
            _backend = "rest"
        else:
            raise RuntimeError("No se pudo usar Redis RESP y no hay credenciales REST (REDIS_REST_URL/TOKEN)")

async def mem_exists(user_id: int, fp: str) -> bool:
    key = mem_key(user_id, fp)
    if _backend is None:
        await decide_backend()
    if _backend == "resp":
        r = await get_redis_resp()
        try:
            return (await r.exists(key)) == 1
        except Exception:
            if REDIS_REST_URL and REDIS_REST_TOKEN:
                _backend = "rest"
            else:
                raise
    if _backend == "rest":
        return await rest_exists(key)
    raise RuntimeError("Backend de memoria no inicializado")

async def mem_setex(user_id: int, fp: str, meta: dict, ttl_days: int):
    key = mem_key(user_id, fp)
    ttl = ttl_days * 24 * 3600
    value = json.dumps(meta)
    if _backend is None:
        await decide_backend()
    if _backend == "resp":
        r = await get_redis_resp()
        try:
            await r.set(key, value, ex=ttl)
            return
        except Exception:
            if REDIS_REST_URL and REDIS_REST_TOKEN:
                _backend = "rest"
            else:
                raise
    if _backend == "rest":
        await rest_setex(key, ttl, value)
        return
    raise RuntimeError("Backend de memoria no inicializado")

# ---------- Odds (async) ----------
async def fetch_h2h_odds_async(client: httpx.AsyncClient, sport_key: str) -> List[dict]:
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {"apiKey": ODDS_API_KEY, "regions": "eu,uk", "markets": "h2h", "oddsFormat": "decimal", "dateFormat": "iso"}
    r = await client.get(url, params=params)
    r.raise_for_status()
    return r.json()

@dataclass
class OutcomePrice:
    book: str
    outcome: str  # "1","X","2"
    price: float

@dataclass
class Match:
    sport_key: str
    league: str
    commence_time: str   # ISO
    home: str
    away: str
    prices: List[OutcomePrice]

def parse_matches(sport_key: str, data: List[dict]) -> List[Match]:
    out: List[Match] = []
    for ev in data:
        prices: List[OutcomePrice] = []
        for bm in ev.get("bookmakers", []):
            bname = bm.get("title", bm.get("key", "book"))
            o_book = {"1":None, "X":None, "2":None}
            for mk in bm.get("markets", []):
                if mk.get("key") != "h2h": continue
                for oc in mk in bm.get("markets", []):
                    if mk.get("key") != "h2h": continue
                for oc in mk.get("outcomes", []):
                    name = (oc.get("name") or "").lower()
                    price = oc.get("price", None)
                    if not price or price <= 1: continue
                    if name == ev["home_team"].lower():
                        o_book["1"] = max(o_book["1"] or 0, float(price))
                    elif name == ev["away_team"].lower():
                        o_book["2"] = max(o_book["2"] or 0, float(price))
                    elif name in ("draw","empate","x"):
                        o_book["X"] = max(o_book["X"] or 0, float(price))
            for side, pr in o_book.items():
                if pr: prices.append(OutcomePrice(bname, side, pr))
        out.append(Match(
            sport_key=sport_key,
            league=sport_key.replace("soccer_","").replace("_"," ").title(),
            commence_time=ev["commence_time"],
            home=ev["home_team"],
            away=ev["away_team"],
            prices=prices
        ))
    return out

async def fetch_all_matches_async() -> List[Match]:
    matches: List[Match] = []
    limits = httpx.Limits(max_connections=HTTP_CONCURRENCY, max_keepalive_connections=HTTP_CONCURRENCY)
    async with httpx.AsyncClient(limits=limits, timeout=HTTP_TIMEOUT) as client:
        tasks = [fetch_h2h_odds_async(client, lg) for lg in LEAGUES]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for lg, res in zip(LEAGUES, results):
            if isinstance(res, Exception):
                print(f"Error odds {lg}: {res}")
                continue
            matches.extend(parse_matches(lg, res))
    return matches

def devig_consensus(prices: List[OutcomePrice]) -> Tuple[Dict[str,float], Dict[str,float]]:
    best = {"1":None,"X":None,"2":None}
    by_book: Dict[str, Dict[str, Optional[float]]] = {}
    for p in prices:
        best[p.outcome] = p.price if (best[p.outcome] is None or p.price>best[p.outcome]) else best[p.outcome]
        by_book.setdefault(p.book, {"1":None,"X":None,"2":None})
        by_book[p.book][p.outcome] = max(by_book[p.book][p.outcome] or 0, p.price)
    triples = []
    for b,od in by_book.items():
        if not od["1"] or not od["X"] or not od["2"]: continue
        r1,rX,r2 = 1.0/od["1"], 1.0/od["X"], 1.0/od["2"]
        s = r1+rX+r2
        if s<=0: continue
        triples.append((r1/s, rX/s, r2/s))
    if not triples:
        return best, {"1":0.0,"X":0.0,"2":0.0}
    avg1 = sum(t[0] for t in triples)/len(triples)
    avgX = sum(t[1] for t in triples)/len(triples)
    avg2 = sum(t[2] for t in triples)/len(triples)
    return best, {"1":avg1,"X":avgX,"2":avg2}

def value_from_consensus(best_price: float, fair_prob: float, buffer: float) -> bool:
    if fair_prob<=0: return False
    fair_odds = 1.0/fair_prob
    return (best_price - fair_odds)/fair_odds >= buffer

def local_datetime_from_iso(iso: str) -> dt.datetime:
    try:
        return dt.datetime.fromisoformat(iso.replace("Z","+00:00")).astimezone(TZ)
    except Exception:
        return dt.datetime.now(TZ)

def local_date_from_iso(iso: str) -> dt.date:
    return local_datetime_from_iso(iso).date()

async def gather_picks_for_date_async(target_date: dt.date) -> List[dict]:
    matches = await fetch_all_matches_async()
    filtered = [m for m in matches if local_date_from_iso(m.commence_time) == target_date]
    picks = []
    for m in filtered:
        best, consensus = devig_consensus(m.prices)
        ratios = {}
        for side in ("1","X","2"):
            bp = best.get(side) or 0.0
            p  = consensus.get(side, 0.0)
            fair_odds = (1.0/p) if p>0 else math.inf
            ratios[side] = (bp / fair_odds) if fair_odds!=math.inf else 0.0
        side = max(ratios, key=ratios.get)
        bp = best.get(side) or 0.0
        p  = consensus.get(side, 0.0)
        if bp<=1 or p<=0 or not value_from_consensus(bp, p, VALUE_BUFFER):
            continue
        dstr = local_datetime_from_iso(m.commence_time).strftime("%Y-%m-%d %H:%M")
        picks.append({"league": m.league, "date": dstr, "home": m.home, "away": m.away, "side": side, "price": bp})
    picks.sort(key=lambda x: x["price"], reverse=True)
    return picks[:TOP_K_PER_DAY]

async def gather_picks_for_offsets_async(offsets: List[int]) -> Dict[dt.date, List[dict]]:
    today = dt.datetime.now(TZ).date()
    targets = { today + dt.timedelta(days=d) for d in offsets }
    matches = await fetch_all_matches_async()
    by_date: Dict[dt.date, List[dict]] = {d: [] for d in targets}
    for m in matches:
        d = local_date_from_iso(m.commence_time)
        if d in by_date:
            by_date[d].append(m)
    out: Dict[dt.date, List[dict]] = {}
    for d, arr in by_date.items():
        picks: List[dict] = []
        for m in arr:
            best, consensus = devig_consensus(m.prices)
            ratios = {}
            for side in ("1","X","2"):
                bp = best.get(side) or 0.0
                p  = consensus.get(side, 0.0)
                fair_odds = (1.0/p) if p>0 else math.inf
                ratios[side] = (bp / fair_odds) if fair_odds!=math.inf else 0.0
            side = max(ratios, key=ratios.get)
            bp = best.get(side) or 0.0
            p  = consensus.get(side, 0.0)
            if bp<=1 or p<=0 or not value_from_consensus(bp, p, VALUE_BUFFER):
                continue
            dstr = local_datetime_from_iso(m.commence_time).strftime("%Y-%m-%d %H:%M")
            picks.append({"league": m.league, "date": dstr, "home": m.home, "away": m.away, "side": side, "price": bp})
        picks.sort(key=lambda x: x["price"], reverse=True)
        out[d] = picks[:TOP_K_PER_DAY]
    return out

def fmt_pick(p: dict, idx: Optional[int]=None) -> str:
    label = {"1": f"1 — Gana {p['home']}", "X": "X — Empate", "2": f"2 — Gana {p['away']}"}
    head = "🧩 Apuesta" if idx is None else f"🧩 Apuesta {idx}"
    return (
        f"{head}\n"
        f"🏆 Liga: {p['league']}\n"
        f"📅 Fecha (hora): {p['date']}\n"
        f"🏟️ Partido: {p['home']} vs {p['away']}\n"
        f"🎯 Mercado / Tipo: 1X2\n"
        f"✅ Selección: {p['side']} — {label[p['side']].split('—')[-1].strip()}\n"
        f"💸 Cuota aprox.: {p['price']:.2f}"
    )

# ---------- Telegram handlers ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hola 👋 Soy tu bot de picks (memoria Upstash resp/rest).\n"
        "Comandos:\n"
        "/picks → *mañana* (por defecto)\n"
        "/today → *hoy*\n"
        "/tomorrow → *mañana*\n"
        "/day 20/08/25 → picks de ese día\n"
        "/week → próximos 7 días\n"
        "/meminfo → estado de la memoria (solo RESP)\n"
        "/forgetall → borrar memoria (solo RESP)"
    )

async def reply_picks(update: Update, day: dt.date, label: str):
    user_id = update.effective_user.id
    if not ODDS_API_KEY:
        await update.message.reply_text("Falta ODDS_API_KEY en el servicio.")
        return
    await update.message.reply_text(f"🔎 Buscando *picks para {label}*…")
    try:
        chosen = await gather_picks_for_date_async(day)
    except Exception as e:
        await update.message.reply_text(f"❌ Error al obtener cuotas: {e}")
        return
    if not chosen:
        await update.message.reply_text("No veo valor claro para esa fecha.")
        return
    sent = skipped = 0
    for pk in chosen:
        fp = pick_fingerprint(pk["date"], pk["league"], pk["home"], pk["away"], pk["side"])
        try:
            already = await mem_exists(user_id, fp)
        except Exception as e:
            await update.message.reply_text(f"⚠️ Memoria no disponible: {e}")
            already = False
        if already:
            skipped += 1; continue
        await update.message.reply_text(fmt_pick(pk, sent+1))
        try:
            await mem_setex(user_id, fp, pk, MEMORY_TTL_DAYS)
        except Exception as e:
            await update.message.reply_text(f"⚠️ No pude guardar memoria: {e}")
        sent += 1
    if sent == 0:
        msg = "No veo picks nuevos para ti (todo lo que salió ya te lo recomendé)."
        if skipped > 0: msg += f" (Omitidos {skipped} repetidos)"
        await update.message.reply_text(msg)
    elif skipped > 0:
        await update.message.reply_text(f"ℹ️ Omitidos {skipped} repetidos.")

async def picks_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target = dt.datetime.now(TZ).date() + dt.timedelta(days=DAY_OFFSET_DEFAULT)
    await reply_picks(update, target, target.strftime('%d/%m/%Y'))

async def today_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target = dt.datetime.now(TZ).date()
    await reply_picks(update, target, f"hoy ({target.strftime('%d/%m/%Y')})")

async def tomorrow_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target = dt.datetime.now(TZ).date() + dt.timedelta(days=1)
    await reply_picks(update, target, f"mañana ({target.strftime('%d/%m/%Y')})")

async def day_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Formato: /day dd/mm/aa (o dd/mm/aaaa). Ej: /day 20/08/25")
        return
    target = parse_local_date(" ".join(context.args))
    if not target:
        await update.message.reply_text("Fecha inválida. Usa dd/mm/aa o dd/mm/aaaa. Ej: 20/08/25")
        return
    await reply_picks(update, target, target.strftime('%d/%m/%Y'))

async def week_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not ODDS_API_KEY:
        await update.message.reply_text("Falta ODDS_API_KEY en el servicio.")
        return
    await update.message.reply_text("🔎 Buscando *picks para los próximos 7 días*…")
    try:
        grouped = await gather_picks_for_offsets_async(list(range(1, 8)))
    except Exception as e:
        await update.message.reply_text(f"❌ Error al obtener cuotas: {e}")
        return
    any_pick = any(len(v)>0 for v in grouped.values())
    if not any_pick:
        await update.message.reply_text("No veo valor claro en los próximos 7 días.")
        return
    user_id = update.effective_user.id
    for d in sorted(grouped.keys()):
        picks = grouped[d]
        if not picks: continue
        header = f"📅 {d.strftime('%A %d/%m/%Y')}"
        lines = [header]
        sent = skipped = 0
        for pk in picks:
            fp = pick_fingerprint(pk["date"], pk["league"], pk["home"], pk["away"], pk["side"])
            try:
                already = await mem_exists(user_id, fp)
            except Exception as e:
                await update.message.reply_text(f"⚠️ Memoria no disponible: {e}")
                already = False
            if already:
                skipped += 1; continue
            lines.append(fmt_pick(pk, sent+1))
            try:
                await mem_setex(user_id, fp, pk, MEMORY_TTL_DAYS)
            except Exception as e:
                lines.append(f"⚠️ No pude guardar memoria: {e}")
            sent += 1
        if sent == 0 and skipped > 0:
            lines.append("No hay picks *nuevos* para ti este día. (Omitidos repetidos)")
        elif skipped > 0:
            lines.append(f"ℹ️ Omitidos {skipped} repetidos.")
        await update.message.reply_text("\n\n".join(lines))

# Utilidades
DATE_PAT = re.compile(r"^\s*(\d{1,2})[/-](\d{1,2})[/-](\d{2}|\d{4})\s*$")
def parse_local_date(text: str) -> Optional[dt.date]:
    m = DATE_PAT.match(text or "")
    if not m: return None
    dd, mm, yy = int(m.group(1)), int(m.group(2)), m.group(3)
    yy_i = int(yy)
    if len(yy)==2: yy_i = 2000 + yy_i
    try: return dt.date(yy_i, mm, dd)
    except ValueError: return None

# Error handler
async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    try:
        if isinstance(update, Update) and update.effective_message:
            await update.effective_message.reply_text("❌ Ocurrió un error inesperado. Intenta de nuevo más tarde.")
    except Exception:
        pass
    print("ERROR:", context.error)

def main():
    if not TELEGRAM_BOT_TOKEN:
        print("Falta TELEGRAM_BOT_TOKEN"); return
    if not EXTERNAL_URL:
        print("Falta WEBHOOK_URL o RENDER_EXTERNAL_URL"); return

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("picks", picks_cmd))
    app.add_handler(CommandHandler("today", today_cmd))
    app.add_handler(CommandHandler("tomorrow", tomorrow_cmd))
    app.add_handler(CommandHandler("day", day_cmd))
    app.add_handler(CommandHandler("week", week_cmd))
    app.add_handler(CommandHandler("meminfo", lambda u,c: u.effective_message.reply_text("🧠 Memoria: usa RESP para ver resumen.")))
    app.add_handler(CommandHandler("forgetall", lambda u,c: u.effective_message.reply_text("🧹 Borrado masivo: usa RESP.")))
    app.add_error_handler(on_error)

    url_path = TELEGRAM_BOT_TOKEN
    webhook_url = EXTERNAL_URL.rstrip("/") + "/" + url_path
    masked = webhook_url.replace(TELEGRAM_BOT_TOKEN, "****TOKEN****")
    print(f"Webhook en {masked}")
    app.run_webhook(listen="0.0.0.0", port=PORT, url_path=url_path, webhook_url=webhook_url)

if __name__ == "__main__":
    main()
