#!/usr/bin/env python3
"""
Персональный новостной дайджест.

Читает config/topics.json, для каждой темы, у которой подошёл срок
(frequency: weekly/monthly), дёргает Anthropic API (модель + web_search),
просит найти свежие новости, отфильтровать шум и не повторять уже виденные
ссылки. Результат шлёт в Telegram, состояние (даты запусков и виденные
ссылки) хранит в data/seen_urls.json — воркфлоу коммитит его обратно.

Защита от лишних трат:
  * состояние сохраняется сразу после КАЖДОЙ оплаченной темы, а не в конце —
    сбой посередине не заставит платить за те же темы повторно;
  * потолок поисков на один прогон (RUN_SEARCH_BUDGET);
  * ручной запуск без галочек ничего не тратит, если сроков нет;
    можно прогнать одну тему (ONLY_TOPIC) или все принудительно (FORCE_ALL);
  * в логе по каждой теме — поиски, токены и оценка стоимости.
"""
import html
import json
import os
import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import requests

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

# Параметры ручного запуска (кнопка Run workflow), для расписания пустые.
ONLY_TOPIC = os.environ.get("ONLY_TOPIC", "").strip()
FORCE_ALL = os.environ.get("FORCE_ALL", "").strip().lower() == "true"

MODEL = "claude-haiku-5-5"
SEARCH_TOOL_TYPE = "web_search_20250305"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

# Каждый поиск подтягивает в контекст страницы (порядка 20-40K токенов).
# Чем больше поисков в одном запросе, тем дороже: у Haiku 5.5 при запросе
# больше 100K токенов весь запрос тарифицируется в 5 раз дороже.
# Поэтому по умолчанию 2 поиска на тему; для важной темы можно поднять
# полем "max_searches" прямо в config/topics.json.
MAX_SEARCHES_PER_TOPIC = 2
RUN_SEARCH_BUDGET = 50          # потолок поисков за один прогон
SEEN_URLS_LIMIT_PER_TOPIC = 200  # чтобы файл состояния не рос бесконечно
RECENCY_DAYS = 14                # старше этого не включаем
FREQUENCY_DAYS = {"daily": 1, "weekly": 7, "monthly": 30}

# Оценка стоимости для лога. Цены токенов Haiku 5.5 взяты из сторонних
# обзоров — сверь с консолью Anthropic (Settings -> Usage). Поиск: $10/1000.
SEARCH_PRICE_USD = 0.01
PRICE_IN_PER_M = 0.10
PRICE_OUT_PER_M = 0.50
LONG_PROMPT_TOKENS = 100_000
LONG_PROMPT_MULT = 5

MSG_LIMIT = 3500  # символов в одном сообщении Telegram (лимит 4096, с запасом)

BASE_DIR = Path(__file__).resolve().parent.parent
TOPICS_FILE = BASE_DIR / "config" / "topics.json"
SEEN_FILE = BASE_DIR / "data" / "seen_urls.json"

SYSTEM_PROMPT = """Ты — персональный новостной аналитик.

Тебе дают описание темы: что именно важно отследить, ключевые сущности \
и источники, что считать шумом, и список уже виденных ссылок (их повторно \
включать нельзя).

Задача:
1. Используй web_search, чтобы найти РЕАЛЬНЫЕ новости/изменения — законы, \
решения регуляторов, официальные объявления. Не общие статьи-гайды/руководства.
2. ОБЯЗАТЕЛЬНО проверяй дату публикации каждого найденного материала. Тебе \
дают сегодняшнюю дату и дату отсечки — включай только то, что опубликовано \
НЕ РАНЬШЕ даты отсечки. Если дата публикации неизвестна или старше отсечки — \
не включай материал, даже если он идеально подходит по теме.
3. Отфильтруй всё, что попадает под "что игнорировать".
4. Не включай ссылки из списка уже виденных.
5. Если по теме ничего нового, свежего и релевантного не нашлось — верни \
пустой список items, не выдумывай новость и не include старые материалы.
6. ВСЕ текстовые поля ответа (title, summary) — строго на русском языке, \
даже если источник на английском или другом языке. Переводи смысл своими \
словами, не копируй иностранный текст как есть.

НЕ пиши рассуждений, пересказов и вступлений — твой ответ ограничен по длине. \
Всё, что ты выводишь текстом, — ТОЛЬКО один ЧИСТЫЙ JSON-объект, без \
markdown-разметки и без пояснений до или после него. Не более 6 новостей, \
summary — 1-2 коротких предложения:
{"items": [{"title": "...", "summary": "1-2 предложения по сути", \
"url": "...", "date": "дата публикации если известна, иначе пустая строка"}]}
"""


class FatalApiError(Exception):
    """Ошибка, после которой продолжать бессмысленно (ключ, баланс)."""


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------- состояние

def load_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_state(raw_state: dict) -> dict:
    """Приводит state к формату {topic_id: {"urls": [...], "last_run": "YYYY-MM-DD"|None}}.
    Старый формат (просто список ссылок) конвертируется на лету."""
    normalized = {}
    for topic_id, val in raw_state.items():
        if isinstance(val, list):
            normalized[topic_id] = {"urls": val, "last_run": None}
        elif isinstance(val, dict):
            normalized[topic_id] = {"urls": val.get("urls", []), "last_run": val.get("last_run")}
        else:
            normalized[topic_id] = {"urls": [], "last_run": None}
    return normalized


def is_due(frequency: str, last_run: str | None, today: date) -> bool:
    """Подошёл ли срок опроса темы (weekly/monthly) от даты последнего запуска."""
    if not last_run:
        return True
    days_needed = FREQUENCY_DAYS.get(frequency, 7)
    return (today - date.fromisoformat(last_run)).days >= days_needed


# ------------------------------------------------------------- запрос к API

def estimate_cost(in_tok: int, out_tok: int, searches: int) -> float:
    mult = LONG_PROMPT_MULT if in_tok > LONG_PROMPT_TOKENS else 1
    tokens_cost = (in_tok * PRICE_IN_PER_M + out_tok * PRICE_OUT_PER_M) / 1_000_000 * mult
    return searches * SEARCH_PRICE_USD + tokens_cost


def call_anthropic(body: dict) -> dict:
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    last_error = "неизвестная ошибка"
    for attempt in range(1, 4):
        try:
            resp = requests.post(ANTHROPIC_API_URL, headers=headers, json=body, timeout=180)
        except requests.RequestException as e:
            last_error = f"сеть: {e}"
            log(f"  попытка {attempt}/3: {last_error}")
            time.sleep(15)
            continue

        if resp.status_code == 200:
            return resp.json()

        detail = resp.text[:300]
        last_error = f"HTTP {resp.status_code}: {detail}"
        low = detail.lower()
        if resp.status_code in (401, 403) or (resp.status_code == 400 and "credit" in low):
            raise FatalApiError(last_error)
        if resp.status_code in (429, 500, 502, 503, 529):
            try:
                wait = min(float(resp.headers.get("retry-after", 20)), 90)
            except ValueError:
                wait = 20
            log(f"  попытка {attempt}/3: {last_error[:120]} — жду {wait:.0f} с")
            time.sleep(wait)
            continue
        raise RuntimeError(last_error)  # прочее (например, 400 по параметрам) повтором не лечится
    raise RuntimeError(f"исчерпаны повторы: {last_error}")


def ask_claude(topic: dict, seen_urls: list) -> tuple[dict, dict]:
    today = date.today().isoformat()
    cutoff = (date.today() - timedelta(days=RECENCY_DAYS)).isoformat()
    user_prompt = (
        f"Сегодняшняя дата: {today}. Дата отсечки по свежести: {cutoff} "
        f"(включай только материалы, опубликованные не раньше этой даты).\n"
        f"Тема: {topic['name']}\n"
        f"Что важно: {topic['what_matters']}\n"
        f"Ключевые сущности/источники: {topic['entities']}\n"
        f"Что игнорировать: {topic['ignore']}\n"
        f"Уже виденные ссылки (не повторять): "
        f"{json.dumps(seen_urls[-50:], ensure_ascii=False)}"
    )
    body = {
        "model": MODEL,
        "max_tokens": 4000,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_prompt}],
        "tools": [
            {
                "type": SEARCH_TOOL_TYPE,
                "name": "web_search",
                "max_uses": topic.get("max_searches", MAX_SEARCHES_PER_TOPIC),
            }
        ],
    }
    data = call_anthropic(body)

    usage_raw = data.get("usage") or {}
    usage = {
        "in": usage_raw.get("input_tokens", 0),
        "out": usage_raw.get("output_tokens", 0),
        "searches": (usage_raw.get("server_tool_use") or {}).get("web_search_requests", 0),
    }

    log(f"[{topic['id']}] stop_reason={data.get('stop_reason')}, блоков={[b.get('type') for b in data.get('content', [])]}")
    text_parts = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
    raw = "".join(text_parts).strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    # Модель сначала может рассуждать вслух и только потом выдать JSON —
    # вырезаем объект от первой "{" до последней "}".
    start, end = raw.find("{"), raw.rfind("}")
    candidate = raw[start:end + 1] if start != -1 and end > start else raw
    log(f"[{topic['id']}] ответ модели (начало): {raw[:500]}")
    if ONLY_TOPIC:
        send_safe(f"🔧 Диагностика [{topic['id']}]: stop={data.get('stop_reason')}, поисков={usage['searches']}, "
                  f"токенов вх/исх={usage['in']}/{usage['out']}\n\n{raw[:1500]}")
    try:
        result = json.loads(candidate)
    except json.JSONDecodeError:
        log(f"[{topic['id']}] не удалось распарсить JSON-ответ модели: {raw[:800]}")
        result = {"items": []}
    return result, usage


# ----------------------------------------------------------------- Telegram

ITEM_SEPARATOR = "\n┈┈┈┈┈┈┈┈┈┈\n"


def esc(s: str) -> str:
    return html.escape(s, quote=False)


def format_item(it: dict) -> str:
    title = (it.get("title") or "Без названия").strip()[:300]
    summary = (it.get("summary") or "").strip()[:1200]
    url = (it.get("url") or "").strip()
    date_str = (it.get("date") or "").strip()[:60]

    title_html = esc(title)
    if url:
        title_html = f'<a href="{html.escape(url, quote=True)}">{title_html}</a>'

    lines = [f"<b>{title_html}</b>"]
    if date_str:
        lines.append(f"<i>{esc(date_str)}</i>")
    if summary:
        lines.append(f"<blockquote>{esc(summary)}</blockquote>")
    return "\n".join(lines)


def build_topic_messages(topic_name: str, items: list) -> list[str]:
    """Режет тему на сообщения по границам новостей (а не по символам) —
    иначе разрез попадает внутрь HTML-тега и Telegram отвечает 400."""
    head = f"📌 <b>{esc(topic_name)}</b>"
    head_cont = f"📌 <b>{esc(topic_name)}</b> <i>(продолжение)</i>"
    messages, current, count = [], head + "\n\n", 0
    for it in items:
        block = format_item(it)
        sep = ITEM_SEPARATOR if count else ""
        if count and len(current) + len(sep) + len(block) > MSG_LIMIT:
            messages.append(current)
            current, count, sep = head_cont + "\n\n", 0, ""
        current += sep + block
        count += 1
    messages.append(current)
    return messages


def html_to_plain(text: str) -> str:
    text = re.sub(r'<a href="([^"]*)">(.*?)</a>', r"\2 (\1)", text, flags=re.S)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text)


def send_telegram(text: str, as_html: bool = True) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "disable_web_page_preview": True,
    }
    if as_html:
        payload["parse_mode"] = "HTML"
    r = requests.post(url, json=payload, timeout=30)
    if not r.ok:
        raise RuntimeError(f"Telegram {r.status_code}: {r.text[:200]}")


def send_safe(text: str) -> bool:
    """Сначала с форматированием, при отказе — простым текстом.
    Никогда не роняет прогон: возвращает True/False."""
    try:
        send_telegram(text)
        time.sleep(1)
        return True
    except Exception as e:
        log(f"Telegram не принял HTML-сообщение ({e}); пробую простым текстом")
    try:
        send_telegram(html_to_plain(text)[:4000], as_html=False)
        time.sleep(1)
        return True
    except Exception as e:
        log(f"Telegram не принял и простой текст: {e}")
        return False


def alert(text: str) -> None:
    """Короткое предупреждение в Telegram, чтобы сбой не оставался тихим."""
    try:
        send_telegram(f"⚠️ Дайджест: {text}"[:4000], as_html=False)
    except Exception as e:
        log(f"Не удалось отправить предупреждение в Telegram: {e}")


# --------------------------------------------------------------------- main

def main() -> None:
    topics = load_json(TOPICS_FILE, [])
    state = normalize_state(load_json(SEEN_FILE, {}))
    today = date.today()

    known_ids = [t["id"] for t in topics]
    if ONLY_TOPIC and ONLY_TOPIC not in known_ids:
        log(f"Тема '{ONLY_TOPIC}' не найдена. Доступные id: {', '.join(known_ids)}")
        sys.exit(1)

    forced = FORCE_ALL or bool(ONLY_TOPIC)
    totals = {"topics": 0, "searches": 0, "in": 0, "out": 0, "cost": 0.0}
    pending = []                 # (topic_id, topic_name, items, messages)
    skipped, postponed, failed = [], [], []

    for topic in topics:
        tid = topic["id"]
        if ONLY_TOPIC and tid != ONLY_TOPIC:
            continue
        topic_state = state.setdefault(tid, {"urls": [], "last_run": None})

        if not forced and not is_due(topic.get("frequency", "weekly"), topic_state["last_run"], today):
            skipped.append(tid)
            continue
        if totals["searches"] >= RUN_SEARCH_BUDGET:
            postponed.append(tid)
            continue

        try:
            result, usage = ask_claude(topic, topic_state["urls"])
        except FatalApiError as e:
            log(f"[{tid}] ФАТАЛЬНАЯ ошибка API, прогон остановлен: {e}")
            alert(f"остановлен из-за ошибки API (проверь ключ и баланс). {str(e)[:300]}")
            break
        except Exception as e:
            log(f"[{tid}] ошибка запроса, тема останется к опросу в следующий раз: {e}")
            failed.append(tid)
            continue

        cost = estimate_cost(usage["in"], usage["out"], usage["searches"])
        totals["topics"] += 1
        totals["searches"] += usage["searches"]
        totals["in"] += usage["in"]
        totals["out"] += usage["out"]
        totals["cost"] += cost
        warn = "  ⚠ вход >100K токенов: тариф x5" if usage["in"] > LONG_PROMPT_TOKENS else ""
        log(f"[{tid}] поисков {usage['searches']}, токены вход {usage['in']} / выход {usage['out']}, ≈${cost:.3f}{warn}")

        # Деньги потрачены — фиксируем это СРАЗУ, чтобы сбой дальше не заставил платить снова.
        topic_state["last_run"] = today.isoformat()
        save_json(SEEN_FILE, state)

        raw_items = result.get("items", []) if isinstance(result, dict) else []
        seen = set(topic_state["urls"])
        items = [it for it in raw_items if isinstance(it, dict) and it.get("url") not in seen]
        if items:
            pending.append((tid, topic["name"], items, build_topic_messages(topic["name"], items)))

    # Рассылка. Ссылки считаем «виденными» только после успешной отправки.
    delivered_topics = []
    if pending:
        if len(pending) > 1:
            names = "\n".join(f"• {esc(name)}" for _, name, _, _ in pending)
            send_safe(f"🗞 <b>Дайджест за {today.strftime('%d.%m.%Y')}</b>\n\n{names}")
        for tid, name, items, messages in pending:
            if all(send_safe(m) for m in messages):
                new_urls = [it["url"] for it in items if it.get("url")]
                st = state[tid]
                st["urls"] = (st["urls"] + new_urls)[-SEEN_URLS_LIMIT_PER_TOPIC:]
                save_json(SEEN_FILE, state)
                delivered_topics.append(tid)
            else:
                log(f"[{tid}] доставка не удалась. Новости для лога:")
                for it in items:
                    log(f"   - {it.get('title')} | {it.get('url')}")
        send_safe(
            f"ℹ️ Прогон: тем {totals['topics']}, поисков {totals['searches']}, "
            f"≈${totals['cost']:.2f} (оценка)"
        )

    log("")
    log(f"Итого: опрошено тем {totals['topics']}, поисков {totals['searches']}, "
        f"токены вход {totals['in']} / выход {totals['out']}, ≈${totals['cost']:.3f}")
    log(f"Пропущено (срок не подошёл): {len(skipped)}; "
        f"отложено из-за потолка поисков: {len(postponed)}; с ошибкой: {len(failed)}")
    if not pending:
        log("Новостей для отправки нет.")

    save_json(SEEN_FILE, state)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # неожиданный сбой: не молчим и сообщаем в Telegram
        log(f"НЕОЖИДАННАЯ ОШИБКА: {exc!r}")
        alert(f"неожиданная ошибка в скрипте: {exc!r}"[:300])
        raise
