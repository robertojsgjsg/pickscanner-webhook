
import os, math, datetime as dt, requests, re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from zoneinfo import ZoneInfo
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, JobQueue

# ---------- Config via env ----------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
BANKROLL = float(os.getenv("BANKROLL", "1000"))
KELLY_FRAC = float(os.getenv("KELLY_FRACTION", "0.25"))
CAP_PER_BET = float(os.getenv("CAP_PER_BET", "0.015"))
VALUE_BUFFER = float(os.getenv("VALUE_BUFFER", "0.01"))  # require >=1% over fair odds
LEAGUES = os.getenv("LEAGUES",
    "soccer_epl,soccer_spain_la_liga,soccer_italy_serie_a,soccer_germany_bundesliga,soccer_france_ligue_one"
).split(",")
TOP_K = int(os.getenv("TOP_K", "4"))
TOP_K_PER_DAY = int(os.getenv("TOP_K_PER_DAY", str(TOP_K)))
TZ = ZoneInfo(os.getenv("TZ", "Europe/Madrid"))
DAY_OFFSET_DEFAULT = int(os.getenv("DAY_OFFSET_DEFAULT", "1"))  # 0=today, 1=tomorrow

# Webhook-specific
PORT = int(os.getenv("PORT", "10000"))
EXTERNAL_URL = os.getenv("WEBHOOK_URL") or os.getenv("RENDER_EXTERNAL_URL")

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

# ---------- Data fetching ----------
def fetch_h2h_odds(sport_key: str) -> List[Match]:
    base = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "eu,uk",
        "markets": "h2h",
        "oddsFormat": "decimal",
        "dateFormat": "iso"
    }
    r = requests.get(base, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    out = []
    for ev in data:
        prices = []
        for bm in ev.get("bookmakers", []):
            bname = bm.get("title", bm.get("key", "book"))
            o_book = {"1":None, "X":None, "2":None}
            for mk in bm.get("markets", []):
                if mk.get("key") != "h2h":
                    continue
                for oc in mk.get("outcomes", []):
                    name = oc.get("name","").lower()
                    price = oc.get("price", None)
                    if not price or price <= 1:
                        continue
                    if name == ev["home_team"].lower():
                        o_book["1"] = max(o_book["1"] or 0, float(price))
                    elif name == ev["away_team"].lower():
                        o_book["2"] = max(o_book["2"] or 0, float(price))
                    elif name in ("draw","empate","x"):
                        o_book["X"] = max(o_book["X"] or 0, float(price))
            for side, pr in o_book.items():
                if pr:
                    prices.append(OutcomePrice(bname, side, pr))
        out.append(Match(
            sport_key=sport_key,
            league=sport_key.replace("soccer_","").replace("_"," ").title(),
            commence_time=ev["commence_time"],
            home=ev["home_team"],
            away=ev["away_team"],
            prices=prices
        ))
    return out

# ---------- Helpers ----------
def devig_consensus(prices: List[OutcomePrice]) -> Tuple[Dict[str,float], Dict[str,float]]:
    best = {"1":None,"X":None,"2":None}
    by_book = {}
    for p in prices:
        best[p.outcome] = p.price if (best[p.outcome] is None or p.price>best[p.outcome]) else best[p.outcome]
        by_book.setdefault(p.book, {"1":None,"X":None,"2":None})
        by_book[p.book][p.outcome] = max(by_book[p.book][p.outcome] or 0, p.price)
    triples = []
    for b,od in by_book.items():
        if not od["1"] or not od["X"] or not od["2"]:
            continue
        r1,rX,r2 = 1.0/od["1"], 1.0/od["X"], 1.0/od["2"]
        s = r1+rX+r2
        if s<=0:
            continue
        triples.append((r1/s, rX/s, r2/s))
    if not triples:
        return best, {"1":0.0,"X":0.0,"2":0.0}
    avg1 = sum(t[0] for t in triples)/len(triples)
    avgX = sum(t[1] for t in triples)/len(triples)
    avg2 = sum(t[2] for t in triples)/len(triples)
    return best, {"1":avg1,"X":avgX,"2":avg2}

def kelly_full(p: float, odds: float) -> float:
    if p<=0 or odds<=1: return 0.0
    b = odds-1; q = 1-p
    f = (b*p - q)/b
    return max(0.0, f)

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

def filter_matches_by_date(matches: List[Match], target: dt.date) -> List[Match]:
    return [m for m in matches if local_date_from_iso(m.commence_time) == target]

def match_to_pick(m: Match) -> Optional[dict]:
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
        return None
    f_full = kelly_full(p, bp)
    f_stake = min(CAP_PER_BET, KELLY_FRAC * f_full)
    stake_eur = BANKROLL * f_stake
    dstr = local_datetime_from_iso(m.commence_time).strftime("%Y-%m-%d %H:%M")
    return {
        "league": m.league,
        "date": dstr,
        "home": m.home,
        "away": m.away,
        "side": side,
        "price": bp,
        "kelly_full": f_full,
        "stake_frac": f_stake,
        "stake_eur": stake_eur
    }

def fmt_pick_simple(p: dict, idx: Optional[int]=None) -> str:
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

# ---------- Day parsing ----------
_DATE_PAT = re.compile(r"^\s*(\d{1,2})[/-](\d{1,2})[/-](\d{2}|\d{4})\s*$")
def parse_local_date(text: str) -> Optional[dt.date]:
    m = _DATE_PAT.match(text or "")
    if not m: return None
    dd, mm, yy = int(m.group(1)), int(m.group(2)), m.group(3)
    yy_i = int(yy)
    if len(yy)==2: yy_i = 2000 + yy_i
    try: return dt.date(yy_i, mm, dd)
    except ValueError: return None

# ---------- Core gatherers ----------
def fetch_all_matches() -> List[Match]:
    all_matches: List[Match] = []
    for lg in LEAGUES:
        try:
            all_matches += fetch_h2h_odds(lg)
        except Exception as e:
            print("Error odds", lg, e)
    return all_matches

def gather_picks_for_date(target_date: dt.date) -> List[dict]:
    matches = fetch_all_matches()
    filtered = filter_matches_by_date(matches, target_date)
    picks: List[dict] = []
    for m in filtered:
        pk = match_to_pick(m)
        if pk:
            picks.append(pk)
    picks.sort(key=lambda x: x["stake_frac"], reverse=True)
    return picks[:TOP_K_PER_DAY]

def gather_picks_for_offsets(offsets: List[int]) -> Dict[dt.date, List[dict]]:
    today = dt.datetime.now(TZ).date()
    targets = { today + dt.timedelta(days=d) for d in offsets }
    matches = fetch_all_matches()
    by_date: Dict[dt.date, List[Match]] = {d: [] for d in targets}
    for m in matches:
        d = local_date_from_iso(m.commence_time)
        if d in by_date:
            by_date[d].append(m)
    out: Dict[dt.date, List[dict]] = {}
    for d, arr in by_date.items():
        picks: List[dict] = []
        for m in arr:
            pk = match_to_pick(m)
            if pk:
                picks.append(pk)
        picks.sort(key=lambda x: x["stake_frac"], reverse=True)
        out[d] = picks[:TOP_K_PER_DAY]
    return out

# ---------- Telegram handlers ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hola 👋 Soy tu bot de picks automáticos.\n"
        "Comandos:\n"
        "/picks → picks de *mañana* (por defecto)\n"
        "/today → picks de *hoy*\n"
        "/tomorrow → picks de *mañana*\n"
        "/day 20/08/25 → picks de ese día (dd/mm/aa o dd/mm/aaaa)\n"
        "/week → picks de los próximos 7 días (mañana→+7)"
    )

async def picks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not ODDS_API_KEY:
        await update.message.reply_text("Falta ODDS_API_KEY. Configúrala y vuelve a intentar.")
        return
    target = dt.datetime.now(TZ).date() + dt.timedelta(days=DAY_OFFSET_DEFAULT)
    await update.message.reply_text(f"🔎 Buscando *picks para {target.strftime('%d/%m/%Y')}*…")
    chosen = gather_picks_for_date(target)
    if not chosen:
        await update.message.reply_text("No veo valor claro para esa fecha.")
        return
    for i,pk in enumerate(chosen, start=1):
        await update.message.reply_text(fmt_pick_simple(pk, i))

async def today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target = dt.datetime.now(TZ).date()
    await update.message.reply_text(f"🔎 Buscando *picks para hoy ({target.strftime('%d/%m/%Y')})*…")
    chosen = gather_picks_for_date(target)
    if not chosen:
        await update.message.reply_text("No veo valor claro hoy.")
        return
    for i,pk in enumerate(chosen, start=1):
        await update.message.reply_text(fmt_pick_simple(pk, i))

async def tomorrow(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target = dt.datetime.now(TZ).date() + dt.timedelta(days=1)
    await update.message.reply_text(f"🔎 Buscando *picks para mañana ({target.strftime('%d/%m/%Y')})*…")
    chosen = gather_picks_for_date(target)
    if not chosen:
        await update.message.reply_text("No veo valor claro para mañana.")
        return
    for i,pk in enumerate(chosen, start=1):
        await update.message.reply_text(fmt_pick_simple(pk, i))

async def day(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Formato: /day dd/mm/aa (o dd/mm/aaaa). Ej: /day 20/08/25")
        return
    target = parse_local_date(" ".join(context.args))
    if not target:
        await update.message.reply_text("Fecha inválida. Usa dd/mm/aa o dd/mm/aaaa. Ej: 20/08/25")
        return
    await update.message.reply_text(f"🔎 Buscando *picks para {target.strftime('%d/%m/%Y')}*…")
    chosen = gather_picks_for_date(target)
    if not chosen:
        await update.message.reply_text("No veo valor claro para esa fecha.")
        return
    for i,pk in enumerate(chosen, start=1):
        await update.message.reply_text(fmt_pick_simple(pk, i))

async def week(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔎 Buscando *picks para los próximos 7 días*…")
    offsets = list(range(1, 8))  # mañana → +7
    grouped = gather_picks_for_offsets(offsets)
    any_pick = any(len(v)>0 for v in grouped.values())
    if not any_pick:
        await update.message.reply_text("No veo valor claro en los próximos 7 días.")
        return
    for d in sorted(grouped.keys()):
        picks = grouped[d]
        if not picks:
            continue
        header = f"📅 {d.strftime('%A %d/%m/%Y')}"
        lines = [header]
        for i,pk in enumerate(picks, start=1):
            block = fmt_pick_simple(pk, i)
            lines.append(block)
        await update.message.reply_text("\n\n".join(lines))

def main():
    if not TELEGRAM_BOT_TOKEN:
        print("Falta TELEGRAM_BOT_TOKEN"); return
    if not EXTERNAL_URL:
        print("Falta WEBHOOK_URL o RENDER_EXTERNAL_URL"); return

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("picks", picks))
    app.add_handler(CommandHandler("today", today))
    app.add_handler(CommandHandler("tomorrow", tomorrow))
    app.add_handler(CommandHandler("day", day))
    app.add_handler(CommandHandler("week", week))

    # Webhook
    url_path = TELEGRAM_BOT_TOKEN
    webhook_url = EXTERNAL_URL.rstrip("/") + "/" + url_path
    print(f"Webhook en {webhook_url}")
    app.run_webhook(listen="0.0.0.0", port=int(os.getenv("PORT", "10000")), url_path=url_path, webhook_url=webhook_url)

if __name__ == "__main__":
    main()
