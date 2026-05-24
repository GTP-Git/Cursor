"""
Royal Caribbean price scraper — Playwright data collector + SQLite persistence.
"""

from __future__ import annotations

import json
import os
import platform
import re
import sqlite3
import time
from contextlib import contextmanager
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from playwright.sync_api import Page, Response, sync_playwright

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "cruises.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS cruises (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ship_name TEXT NOT NULL,
    sailing_date TEXT NOT NULL,
    duration TEXT,
    departure_port TEXT,
    itinerary TEXT,
    url TEXT NOT NULL UNIQUE,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS pricing_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    cruise_id INTEGER NOT NULL,
    interior_price REAL,
    oceanview_price REAL,
    balcony_price REAL,
    suite_price REAL,
    voom_price REAL,
    FOREIGN KEY (cruise_id) REFERENCES cruises(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_pricing_cruise_time
    ON pricing_history (cruise_id, timestamp DESC);
"""

PRICE_COLUMNS = (
    "interior_price",
    "oceanview_price",
    "balcony_price",
    "suite_price",
    "voom_price",
)

CABIN_LABELS = {
    "interior_price": "Interior",
    "oceanview_price": "Ocean View",
    "balcony_price": "Balcony",
    "suite_price": "Suite",
    "voom_price": "Voom (Internet)",
}

# Lowest → highest stateroom tier (used for inversion detection).
CABIN_TIER_ORDER: tuple[tuple[str, str], ...] = (
    ("interior_price", "Interior"),
    ("oceanview_price", "Ocean View"),
    ("balcony_price", "Balcony"),
    ("suite_price", "Suite"),
)

DESKTOP_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def init_database(db_path: Path | None = None) -> None:
    """Create SQLite schema if the database file does not exist yet."""
    path = db_path or DB_PATH
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.commit()


@contextmanager
def get_connection(db_path: Path | None = None):
    path = db_path or DB_PATH
    init_database(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_night_count(duration: str | None = None, url: str | None = None) -> int:
    """Parse cruise length in nights from duration text or URL slug."""
    if duration:
        match = re.search(r"(\d+)", str(duration))
        if match:
            return max(1, int(match.group(1)))
    if url:
        match = re.search(r"(\d+)-night", url.lower())
        if match:
            return max(1, int(match.group(1)))
    return 7


def parse_price(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None
    text = str(value)
    match = re.search(r"[\d,]+(?:\.\d{2})?", text.replace(",", ""))
    if not match:
        return None
    try:
        amount = float(match.group().replace(",", ""))
        return amount if amount > 0 else None
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Playwright scraping
# ---------------------------------------------------------------------------

CABIN_KEYWORDS: dict[str, list[str]] = {
    "interior_price": ["interior", "inside"],
    "oceanview_price": ["ocean view", "oceanview", "outside"],
    "balcony_price": ["balcony"],
    "suite_price": ["suite", "royal suite", "grand suite"],
}

VOOM_KEYWORDS = ["voom", "surf + stream", "surf & stream", "surf and stream", "internet package"]

CHROMIUM_LAUNCH_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--no-sandbox",
]


class PlaywrightSetupError(RuntimeError):
    """Playwright browsers missing or wrong CPU architecture for this Python."""


def _playwright_cache_dir() -> Path:
    override = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if override:
        return Path(override)
    return Path.home() / "Library/Caches/ms-playwright"


def discover_chromium_executables() -> list[Path]:
    """Return installed Chromium binaries (newest builds first)."""
    root = _playwright_cache_dir()
    if not root.exists():
        return []
    patterns = (
        "chromium-*/chrome-mac-*/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",
        "chromium_headless_shell-*/chrome-headless-shell-mac-*/chrome-headless-shell",
    )
    found: list[Path] = []
    for pattern in patterns:
        found.extend(root.glob(pattern))
    return sorted(
        [p for p in found if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def check_playwright_ready() -> tuple[bool, str]:
    """
    Verify Playwright browser binaries exist for the current Python architecture.
    Returns (ready, human-readable message).
    """
    machine = platform.machine().lower()
    expected = "arm64" if machine in ("arm64", "aarch64") else "x64"
    executables = discover_chromium_executables()

    if not executables:
        return False, (
            "Playwright browsers are not installed.\n\n"
            "Run in your project folder:\n"
            "  cd ~/royal-caribbean-tracker\n"
            "  python3 -m playwright install chromium"
        )

    matching = [p for p in executables if f"mac-{expected}" in str(p)]
    if matching:
        return True, f"Playwright is ready ({expected} browser found)."

    other_arch = "arm64" if expected == "x64" else "x64"
    return False, (
        f"Python is running as **{expected}**, but only **{other_arch}** Playwright "
        f"browsers are installed. The scraper cannot launch.\n\n"
        "**Fix (recommended):** use a native arm64 virtualenv:\n"
        "```\n"
        "cd ~/royal-caribbean-tracker\n"
        "python3 -m venv .venv\n"
        "source .venv/bin/activate\n"
        "pip install -r requirements.txt\n"
        "python -m playwright install chromium\n"
        "streamlit run app.py\n"
        "```\n\n"
        f"**Or** install browsers for your current Python arch:\n"
        f"```\n"
        f"arch -{expected} python3 -m playwright install chromium\n"
        f"```"
    )


def _launch_chromium(playwright):
    """Launch Chromium, auto-picking an installed binary when the default path fails."""
    machine = platform.machine().lower()
    expected = "arm64" if machine in ("arm64", "aarch64") else "x64"
    executables = discover_chromium_executables()
    ordered = sorted(
        executables,
        key=lambda p: (
            expected not in str(p),
            "headless_shell" in str(p),
            "Google Chrome for Testing" not in str(p),
        ),
    )

    errors: list[str] = []
    for executable in ordered:
        try:
            return playwright.chromium.launch(
                headless=True,
                executable_path=str(executable),
                args=CHROMIUM_LAUNCH_ARGS,
            )
        except Exception as exc:
            errors.append(f"{executable.name}: {exc}")

    try:
        return playwright.chromium.launch(headless=True, args=CHROMIUM_LAUNCH_ARGS)
    except Exception as exc:
        errors.append(f"default launch: {exc}")

    _, setup_message = check_playwright_ready()
    detail = "; ".join(errors[-2:]) if errors else str(exc)
    raise PlaywrightSetupError(f"{setup_message}\n\nTechnical detail: {detail}")


def _browser_context(playwright):
    browser = _launch_chromium(playwright)
    context = browser.new_context(
        user_agent=DESKTOP_USER_AGENT,
        viewport={"width": 1440, "height": 900},
        locale="en-US",
        timezone_id="America/New_York",
        extra_http_headers={
            "Accept-Language": "en-US,en;q=0.9",
            "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"macOS"',
        },
    )
    context.add_init_script(
        "Object.defineProperty(navigator, 'webdriver', { get: () => undefined });"
    )
    return browser, context


def _metadata_from_url(url: str) -> dict[str, str | None]:
    """Pull sail date and hints from the booking/itinerary URL."""
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    sailing_date = None
    for key in ("sailDate", "sail-date", "sail_date"):
        if query.get(key):
            sailing_date = query[key][0]
            break

    path_lower = parsed.path.lower()
    ship_name = None
    ship_patterns = (
        (r"mariner", "Mariner of the Seas"),
        (r"icon-of-the-seas|icon", "Icon of the Seas"),
        (r"wonder-of-the-seas|wonder", "Wonder of the Seas"),
        (r"symphony", "Symphony of the Seas"),
        (r"harmony", "Harmony of the Seas"),
        (r"oasis", "Oasis of the Seas"),
        (r"allure", "Allure of the Seas"),
    )
    for needle, label in ship_patterns:
        if needle in path_lower:
            ship_name = label
            break

    duration = None
    nights_match = re.search(r"(\d+)-night", path_lower)
    if nights_match:
        duration = f"{nights_match.group(1)} nights"

    departure_port = None
    if "galveston" in path_lower:
        departure_port = "Galveston, Texas"

    return {
        "sailing_date": sailing_date,
        "ship_name": ship_name,
        "duration": duration,
        "departure_port": departure_port,
    }


def _dismiss_overlays(page: Page) -> None:
    selectors = [
        "button:has-text('Accept')",
        "button:has-text('Accept All')",
        "button:has-text('I Agree')",
        "button:has-text('Got it')",
        "button:has-text('Continue')",
        "[data-testid='cookie-accept']",
        "#onetrust-accept-btn-handler",
    ]
    for selector in selectors:
        try:
            btn = page.locator(selector).first
            if btn.is_visible(timeout=1500):
                btn.click(timeout=2000)
                page.wait_for_timeout(500)
        except Exception:
            continue


def _collect_api_payloads(responses: list[Response]) -> list[dict]:
    payloads: list[dict] = []
    for response in responses:
        url = response.url.lower()
        if not any(
            token in url
            for token in (
                "commerce-api",
                "pricing",
                "cruise",
                "itinerary",
                "stateroom",
                "sailing",
                "addon",
                "add-on",
                "product",
                "catalog",
                "internet",
                "wifi",
                "voom",
            )
        ):
            continue
        try:
            if "application/json" not in (response.headers.get("content-type") or ""):
                continue
            data = response.json()
            if isinstance(data, dict):
                payloads.append(data)
        except Exception:
            continue
    return payloads


def _walk_json(node: Any, path: str = "") -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            key_path = f"{path}.{key}" if path else key
            found.append((key_path.lower(), value))
            found.extend(_walk_json(value, key_path))
    elif isinstance(node, list):
        for index, item in enumerate(node):
            found.extend(_walk_json(item, f"{path}[{index}]"))
    return found


def _open_pricing_tab(page: Page) -> None:
    """Royal Caribbean lists cabin prices on the Rooms tab."""
    for label in ("Rooms", "Staterooms", "Select room type"):
        try:
            tab = page.get_by_role("button", name=re.compile(label, re.I)).first
            if tab.is_visible(timeout=2500):
                tab.click(timeout=3000)
                page.wait_for_timeout(2500)
                return
        except Exception:
            continue


# Royal Caribbean shows different prices under each stateroom category tab.
ROOM_CATEGORY_TABS: tuple[tuple[str, str], ...] = (
    ("Interior Rooms", "interior_price"),
    ("Ocean View Rooms", "oceanview_price"),
    ("Balcony Rooms", "balcony_price"),
    ("Suite", "suite_price"),
)

_EXTRACT_CATEGORY_TAB_PRICES_JS = """
async () => {
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const tabMap = [
    ['Interior Rooms', 'interior_price'],
    ['Ocean View Rooms', 'oceanview_price'],
    ['Balcony Rooms', 'balcony_price'],
    ['Suite', 'suite_price'],
  ];

  const findRoomsRoot = () => {
    const heading = Array.from(document.querySelectorAll('h5, h4, h3')).find((h) =>
      /available rooms/i.test(h.textContent || '')
    );
    let node = heading ? heading.parentElement : null;
    for (let i = 0; i < 8 && node; i += 1) {
      if (/starting from/i.test(node.innerText || '')) return node;
      node = node.parentElement;
    }
    return heading?.parentElement || document.body;
  };

  const startingPrices = (root) => {
    const text = root?.innerText || '';
    const amounts = [];
    for (const match of text.matchAll(
      /Starting from:\\s*\\$\\s*([\\d,]+(?:\\.\\d{2})?)/gi
    )) {
      const value = parseFloat(match[1].replace(/,/g, ''));
      if (value >= 300 && value <= 50000) amounts.push(value);
    }
    return amounts.length ? Math.min(...amounts) : null;
  };

  const roomsRoot = findRoomsRoot();
  const result = {};

  for (const [tabLabel, field] of tabMap) {
    const tab = Array.from(document.querySelectorAll('[role="tab"]')).find(
      (el) => (el.textContent || '').trim() === tabLabel
    );
    if (tab) {
      tab.click();
      await sleep(3000);
    }

    const panelId = tab?.getAttribute('aria-controls');
    const panel = panelId ? document.getElementById(panelId) : null;
    const scoped = panel && (panel.innerText || '').includes('Starting from')
      ? panel
      : roomsRoot;

    result[field] = startingPrices(scoped);
  }

  return result;
}
"""


def _extract_prices_by_category_tabs(page: Page) -> dict[str, float | None]:
    """Click each RC category tab and parse 'Starting from' fares in Available Rooms."""
    prices = {col: None for col in PRICE_COLUMNS if col != "voom_price"}
    _open_pricing_tab(page)
    page.wait_for_timeout(2000)

    try:
        raw = page.evaluate(_EXTRACT_CATEGORY_TAB_PRICES_JS)
        if isinstance(raw, dict):
            for column in prices:
                prices[column] = parse_price(raw.get(column))
        if _prices_all_identical(prices):
            return {col: None for col in prices}
    except Exception:
        pass
    return prices


def _prices_all_identical(prices: dict[str, float | None]) -> bool:
    cabin_cols = [c for c in PRICE_COLUMNS if c != "voom_price"]
    values = [prices.get(c) for c in cabin_cols if prices.get(c) is not None]
    return len(values) >= 2 and len(set(values)) == 1


def _extract_stateroom_card_prices(page: Page) -> dict[str, float | None]:
    """Parse per-tier prices from Royal Caribbean stateroom cards on the Rooms tab."""
    script = """
    () => {
      const money = (text) => {
        const matches = String(text || '').match(/\\$\\s*([\\d,]+(?:\\.\\d{2})?)/g);
        if (!matches) return null;
        const values = matches.map((m) =>
          parseFloat(m.replace(/[$,\\s]/g, ''))
        ).filter((n) => n >= 100);
        return values.length ? Math.min(...values) : null;
      };

      const tiers = [
        ['interior_price', /(we choose your interior|promenade view interior|connecting promenade interior|\\binterior\\b|\\binside\\b)/i],
        ['oceanview_price', /(ocean\\s*view|oceanview|\\boutside\\b)/i],
        ['balcony_price', /(spa balcony|boardwalk balcony|\\bbalcony\\b)/i],
        ['suite_price', /(royal suite|grand suite|\\bsuite\\b)/i],
      ];

      const result = {
        interior_price: null,
        oceanview_price: null,
        balcony_price: null,
        suite_price: null,
      };

      const candidates = Array.from(
        document.querySelectorAll(
          'button, a, [role="button"], article, section, li, div'
        )
      );

      for (const [field, pattern] of tiers) {
        for (const el of candidates) {
          const label = (
            el.getAttribute('aria-label') ||
            el.textContent ||
            ''
          ).trim();
          if (!label || label.length > 180) continue;
          if (!pattern.test(label)) continue;

          const container =
            el.closest(
              'article, section, li, [class*="card"], [class*="room"], [class*="stateroom"]'
            ) || el.parentElement;
          const price = money(container ? container.innerText : label);
          if (price) {
            result[field] = price;
            break;
          }
        }
      }

      return result;
    }
    """
    prices = {col: None for col in PRICE_COLUMNS}
    try:
        raw = page.evaluate(script)
        for column in PRICE_COLUMNS:
            if column == "voom_price":
                continue
            prices[column] = parse_price(raw.get(column))
    except Exception:
        pass
    return prices


def _extract_prices_from_page_text(page: Page) -> dict[str, float | None]:
    """Last-resort regex — only fills tiers still missing after card parsing."""
    prices = {col: None for col in PRICE_COLUMNS}
    try:
        body = page.inner_text("body", timeout=10_000)
    except Exception:
        return prices

    tier_patterns = {
        "interior_price": r"(?:we choose your )?interior|inside",
        "oceanview_price": r"ocean\s*view|oceanview|outside",
        "balcony_price": r"balcony",
        "suite_price": r"\bsuite\b",
    }
    for column, tier_re in tier_patterns.items():
        if prices.get(column):
            continue
        for match in re.finditer(
            rf"({tier_re}).{{0,120}}?\$(\d[\d,]*(?:\.\d{{2}})?)",
            body,
            re.IGNORECASE | re.DOTALL,
        ):
            prices[column] = parse_price(match.group(2))
            break
    return prices


def _api_column_for_key(key_path: str) -> str | None:
    """Map commerce-api JSON paths to a single cabin tier (strict matching)."""
    key = key_path.lower()
    if "price" not in key and "fare" not in key and "amount" not in key:
        return None

    rules: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("interior_price", ("interior", "inside", ".i.", "_i_", "price_usd_i", "roomtypeid_interior")),
        ("oceanview_price", ("oceanview", "ocean view", "outside", ".o.", "_o_", "price_usd_o", "roomtypeid_oceanview")),
        ("balcony_price", ("balcony", ".b.", "_b_", "price_usd_b", "roomtypeid_balcony")),
        ("suite_price", ("suite", ".d.", "_d_", "price_usd_d", "roomtypeid_suite")),
    )
    for column, hints in rules:
        if any(hint in key for hint in hints):
            return column
    return None


def _extract_from_api_payloads(payloads: list[dict]) -> dict[str, float | None]:
    prices = {col: None for col in PRICE_COLUMNS}
    for payload in payloads:
        for key_path, value in _walk_json(payload):
            if not isinstance(value, (int, float, str)):
                continue
            amount = parse_price(value)
            if amount is None:
                continue

            column = _api_column_for_key(key_path)
            if column:
                existing = prices[column]
                if existing is None or amount < existing:
                    prices[column] = amount
                continue

            if prices["voom_price"] is None and any(kw in key_path for kw in VOOM_KEYWORDS):
                if "price" in key_path or "amount" in key_path or "daily" in key_path:
                    prices["voom_price"] = amount
    return prices


def _booking_params_from_url(url: str) -> dict[str, str]:
    """Build room-selection query parameters from an itinerary URL."""
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    package = (query.get("packageCode") or query.get("packagecode") or [None])[0]
    if not package:
        slug_match = re.search(
            r"-on-[a-z0-9-]+-([A-Z]{2}\d{2}[A-Z0-9]+)",
            parsed.path,
            re.I,
        )
        if slug_match:
            package = slug_match.group(1)
        else:
            match = re.search(r"([A-Z]{2}\d{2}[A-Z0-9]+)", parsed.path, re.I)
            package = match.group(1) if match else ""

    sail_date = ""
    for key in ("sailDate", "sail-date", "sail_date"):
        if query.get(key):
            sail_date = query[key][0]
            break

    ship_code = package[:2] if package and len(package) >= 2 else "MA"
    return {
        "packageCode": package,
        "shipCode": ship_code,
        "sailDate": sail_date,
        "country": (query.get("country") or ["USA"])[0],
        "currencyCode": (query.get("currencyCode") or query.get("currency") or ["USD"])[0],
        "selectedCurrencyCode": "USD",
    }


def _booking_guest_query(params: dict[str, str]) -> dict[str, str]:
    """Guest/room defaults required by the booking add-ons step."""
    return {
        **params,
        "roomIndex": "0",
        "r0a": "2",
        "r0c": "0",
        "r0d": "INTERIOR",
    }


def _normalize_voom_daily(amount: float | None, nights: int) -> float | None:
    """
    RC may return per-device per-day ($15–$35) or voyage total ($60–$140).
    Normalize to a daily rate for storage.
    """
    if amount is None or amount <= 0:
        return None
    nights = max(1, nights)
    if amount > 80:
        amount = round(amount / nights, 2)
    if amount < 8 or amount > 80:
        return None
    return amount


def _page_has_voom_content(page: Page) -> bool:
    try:
        body = page.inner_text("body", timeout=8_000).lower()
    except Exception:
        return False
    return (
        ("surf" in body and "stream" in body)
        or "voom" in body
        or ("internet" in body and "device" in body)
        or ("wi-fi" in body and "/day" in body)
    )


def _find_voom_in_node(node: Any) -> float | None:
    """Locate Surf + Stream / Voom per-day price inside commerce API JSON."""
    if isinstance(node, dict):
        text_blob = " ".join(
            str(node.get(field, ""))
            for field in ("title", "name", "description", "productName", "displayName", "code")
        ).lower()
        if any(token in text_blob for token in ("surf + stream", "surf & stream", "surf and stream", "voom")):
            for key, value in node.items():
                key_lower = key.lower()
                if any(token in key_lower for token in ("price", "fare", "amount", "daily")):
                    amount = parse_price(value)
                    if amount is not None and amount < 500:
                        return amount
        for value in node.values():
            found = _find_voom_in_node(value)
            if found is not None:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _find_voom_in_node(item)
            if found is not None:
                return found
    return None


def _extract_voom_from_api_payloads(payloads: list[dict]) -> float | None:
    for payload in payloads:
        price = _find_voom_in_node(payload)
        if price is not None:
            return price
    return None


_EXTRACT_VOOM_TEXT_JS = """
() => {
  const parseMoney = (raw) => {
    if (raw == null) return null;
    const value = parseFloat(String(raw).replace(/,/g, ''));
    return Number.isFinite(value) ? value : null;
  };

  const pickFromText = (text) => {
    const patterns = [
      /surf\\s*[+&]\\s*stream[\\s\\S]{0,500}?\\$\\s*([\\d,]+(?:\\.\\d{2})?)(?:\\s*\\/\\s*(?:device\\s*)?day)?/i,
      /voom\\s*(?:surf\\s*[+&]\\s*stream)?[\\s\\S]{0,300}?\\$\\s*([\\d,]+(?:\\.\\d{2})?)(?:\\s*\\/\\s*(?:device\\s*)?day)?/i,
      /surf\\s*\\+\\s*stream[^$]{0,200}?\\$\\s*([\\d,]+(?:\\.\\d{2})?)/i,
      /internet[^$]{0,160}?\\$\\s*([\\d,]+(?:\\.\\d{2})?)\\s*\\/\\s*day/i,
    ];
    for (const pattern of patterns) {
      const match = text.match(pattern);
      if (match) return parseMoney(match[1]);
    }
    return null;
  };

  const bodyText = document.body.innerText || '';
  let best = pickFromText(bodyText);

  const cards = Array.from(
    document.querySelectorAll('article, section, li, div, button, label')
  ).filter((el) => {
    const t = (el.innerText || '').toLowerCase();
    return (
      (t.includes('surf') && t.includes('stream')) ||
      t.includes('voom') ||
      (t.includes('internet') && t.includes('device'))
    );
  });

  for (const card of cards) {
    const t = card.innerText || '';
    if (!/\\$\\s*\\d/.test(t)) continue;
    const amount = pickFromText(t);
    if (amount != null && (best == null || amount > best)) {
      best = amount;
    }
  }

  return best;
}
"""


def _extract_voom_from_html(html: str) -> float | None:
    """Search serialized page HTML/JSON for Surf + Stream per-day pricing."""
    patterns = (
        r"surf\s*[+&]\s*stream[^\"]{0,400}?\"(?:price|amount|dailyPrice)\"?\s*:\s*(\d+(?:\.\d{2})?)",
        r"\"(?:price|amount|dailyPrice)\"?\s*:\s*(\d+(?:\.\d{2})?)[^\"]{0,200}?surf\s*[+&]\s*stream",
        r"voom[^\"]{0,200}?\"(?:price|amount)\"?\s*:\s*(\d+(?:\.\d{2})?)",
    )
    for pattern in patterns:
        match = re.search(pattern, html, re.I)
        if match:
            return _normalize_voom_daily(parse_price(match.group(1)), 4)
    return None


def _advance_booking_for_voom(page: Page, itinerary_url: str) -> bool:
    """
    Walk Royal Caribbean's Book-now funnel to checkout/add-ons.
    Internet (Voom) is only sold on some sailings; many only offer dining/gratuities here.
    """
    try:
        page.goto(itinerary_url, wait_until="domcontentloaded", timeout=60_000)
        _dismiss_overlays(page)
        page.wait_for_timeout(2000)

        for link_label in ("Book now", "Book Now"):
            try:
                link = page.get_by_role("link", name=re.compile(link_label, re.I)).first
                if link.is_visible(timeout=3000):
                    link.click(timeout=15_000)
                    page.wait_for_timeout(4000)
                    break
            except Exception:
                continue

        if "rooms-and-guests" in page.url:
            try:
                page.get_by_role(
                    "button", name=re.compile("Continue to Room Selection", re.I)
                ).first.click(timeout=15_000)
                page.wait_for_timeout(5000)
            except Exception:
                pass

        try:
            page.locator("button").filter(
                has_text=re.compile("InteriorFrom", re.I)
            ).first.click(timeout=12_000)
            page.wait_for_timeout(4000)
        except Exception:
            pass

        try:
            page.get_by_text(re.compile(r"We choose your", re.I)).first.click(
                timeout=12_000
            )
            page.wait_for_timeout(6000)
        except Exception:
            pass

        return "checkout/add-ons" in page.url or _page_has_voom_content(page)
    except Exception:
        return False


def _read_voom_from_page(
    page: Page, captured_responses: list[Response], nights: int
) -> float | None:
    """Parse Voom / Surf + Stream from the current page and captured API payloads."""
    api_price = _normalize_voom_daily(
        _extract_voom_from_api_payloads(_collect_api_payloads(captured_responses)),
        nights,
    )
    if api_price is not None:
        return api_price

    try:
        daily = page.evaluate(_EXTRACT_VOOM_TEXT_JS)
    except Exception:
        daily = None
    parsed = _normalize_voom_daily(parse_price(daily), nights)
    if parsed is not None:
        return parsed

    try:
        html = page.content()
        return _extract_voom_from_html(html)
    except Exception:
        return None


def _scrape_voom_via_booking_flow(
    page: Page,
    itinerary_url: str,
    captured_responses: list[Response],
    nights: int = 7,
) -> float | None:
    """
    Voom (Surf + Stream) is sold in the booking funnel or Cruise Planner, not on the
    itinerary Rooms tab. Walk Book now → room pick → checkout/add-ons when possible.
    """
    price = _normalize_voom_daily(
        _extract_voom_from_api_payloads(_collect_api_payloads(captured_responses)),
        nights,
    )
    if price is not None:
        return price

    if not _advance_booking_for_voom(page, itinerary_url):
        return _normalize_voom_daily(
            _extract_voom_from_api_payloads(_collect_api_payloads(captured_responses)),
            nights,
        )

    found = _read_voom_from_page(page, captured_responses, nights)
    if found is not None:
        return found

    return _normalize_voom_daily(
        _extract_voom_from_api_payloads(_collect_api_payloads(captured_responses)),
        nights,
    )


def _extract_from_dom(page: Page) -> dict[str, float | None]:
    script = """
    () => {
      const money = (text) => {
        if (!text) return null;
        const m = String(text).replace(/,/g, '').match(/\\$?\\s*(\\d+(?:\\.\\d{2})?)/);
        return m ? parseFloat(m[1]) : null;
      };

      const result = {
        interior_price: null,
        oceanview_price: null,
        balcony_price: null,
        suite_price: null,
        voom_price: null,
        ship_name: null,
        sailing_date: null,
        duration: null,
        departure_port: null,
        itinerary: null,
      };

      const keywords = {
        interior_price: ['interior', 'inside'],
        oceanview_price: ['ocean view', 'oceanview', 'outside'],
        balcony_price: ['balcony'],
        suite_price: ['suite'],
        voom_price: ['voom', 'surf + stream', 'surf & stream', 'internet'],
      };

      const title = document.querySelector('h1');
      const h1Text = title ? title.innerText.trim() : '';
      if (h1Text.includes('•')) {
        const parts = h1Text.split('•').map((p) => p.trim()).filter(Boolean);
        if (parts.length >= 1) {
          const last = parts[parts.length - 1];
          if (last.toLowerCase().includes('seas') || last.length < 60) {
            result.ship_name = last;
          }
        }
        if (parts.length >= 2) result.departure_port = parts[parts.length - 2];
        if (parts.length >= 1) {
          const nights = parts[0].match(/(\\d+)\\s*Nights?/i);
          if (nights) result.duration = `${nights[1]} nights`;
        }
      } else if (h1Text) {
        result.ship_name = h1Text;
      }

      const shipMatch = document.body.innerText.match(
        /([A-Z][\\w\\s]+?\\s+of\\s+the\\s+Seas)/i
      );
      if (shipMatch) result.ship_name = shipMatch[1].trim();

      const meta = (name) =>
        document.querySelector(`meta[property="${name}"], meta[name="${name}"]`)?.content;

      const cards = Array.from(
        document.querySelectorAll(
          '[class*="stateroom"], [class*="cabin"], [class*="room-type"], [data-testid*="room"], article, section'
        )
      );

      const scanBlock = (el, label) => {
        const text = (el.innerText || '').toLowerCase();
        const price = money(el.innerText);
        if (!price) return;
        for (const [field, words] of Object.entries(keywords)) {
          if (result[field] !== null) continue;
          if (words.some((w) => text.includes(w))) {
            result[field] = price;
          }
        }
      };

      cards.forEach((el) => scanBlock(el, 'card'));

      const lines = document.body.innerText.split('\\n').map((l) => l.trim()).filter(Boolean);
      for (let i = 0; i < lines.length; i++) {
        const line = lines[i].toLowerCase();
        const price = money(lines[i]) || money(lines[i + 1] || '');
        if (!price) continue;
        for (const [field, words] of Object.entries(keywords)) {
          if (result[field] !== null) continue;
          if (words.some((w) => line.includes(w))) {
            result[field] = price;
          }
        }
      }

      const ld = Array.from(document.querySelectorAll('script[type="application/ld+json"]'));
      for (const node of ld) {
        try {
          const data = JSON.parse(node.textContent);
          const items = Array.isArray(data) ? data : [data];
          for (const item of items) {
            if (item.name && !result.ship_name) result.ship_name = item.name;
            if (item.startDate && !result.sailing_date) result.sailing_date = item.startDate;
            if (item.description && !result.itinerary) result.itinerary = item.description;
            if (item.offers && item.offers.price) {
              const p = money(String(item.offers.price));
              if (p && !result.interior_price) result.interior_price = p;
            }
          }
        } catch (_) {}
      }

      const bodyText = document.body.innerText;
      const sailMatch = bodyText.match(/Sail(?:ing)? Date[:\\s]+([^\\n]+)/i);
      if (sailMatch) result.sailing_date = sailMatch[1].trim();
      const nightsMatch = bodyText.match(/(\\d+)\\s*(?:Nights?|Days?)/i);
      if (nightsMatch) result.duration = `${nightsMatch[1]} nights`;
      const portMatch = bodyText.match(/(?:From|Departing|Departure)[:\\s]+([^\\n]+)/i);
      if (portMatch) result.departure_port = portMatch[1].trim();

      return result;
    }
    """
    try:
        raw = page.evaluate(script)
        return {k: parse_price(v) if k.endswith("_price") else v for k, v in raw.items()}
    except Exception:
        return {col: None for col in PRICE_COLUMNS}


def is_itinerary_url(url: str) -> bool:
    """True when the URL points at an RC itinerary page (supports multi-date discovery)."""
    return "/itinerary/" in urlparse(url).path.lower()


def _itinerary_url_for_sail_date(
    base_url: str,
    sail_date: str,
    *,
    package_code: str | None = None,
    group_id: str | None = None,
) -> str:
    """Build an itinerary URL for a specific sail date."""
    parsed = urlparse(base_url)
    query = parse_qs(parsed.query)
    query["sailDate"] = [sail_date]
    if package_code:
        query["packageCode"] = [package_code]
    if group_id:
        query["groupId"] = [group_id]
    flat = {key: values[0] for key, values in query.items() if values}
    return urlunparse(parsed._replace(query=urlencode(flat)))


def _parse_sailings_response(data: Any) -> list[dict[str, Any]]:
    """Extract open sail dates from the RC itinerary sailings API payload."""
    if isinstance(data, dict):
        sailings = data.get("sailings") or []
    elif isinstance(data, list):
        sailings = data
    else:
        return []

    results: list[dict[str, Any]] = []
    for item in sailings:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or "OPEN").upper()
        if status not in {"OPEN", "AVAILABLE"}:
            continue
        sail_date = item.get("sailDate") or item.get("startDate")
        if not sail_date:
            continue
        results.append(
            {
                "sailing_date": str(sail_date),
                "package_code": item.get("packageCode"),
                "status": status,
            }
        )
    return results


def discover_itinerary_sail_dates(url: str, timeout_ms: int = 60_000) -> dict[str, Any]:
    """
    Load an RC itinerary page and return every open sail date with a bookable URL.
    Royal Caribbean prices vary by departure date; each sail date is tracked separately.
    """
    captured_payload: dict[str, Any] | None = None
    api_request_url: str | None = None

    with sync_playwright() as playwright:
        browser, context = _browser_context(playwright)
        page = context.new_page()

        def on_response(response: Response) -> None:
            nonlocal captured_payload, api_request_url
            if "itinerary/api/v1/sailings" not in response.url or response.status != 200:
                return
            api_request_url = response.url
            try:
                captured_payload = response.json()
            except Exception:
                pass

        page.on("response", on_response)

        try:
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            _dismiss_overlays(page)
            page.wait_for_timeout(2500)
            try:
                page.wait_for_load_state("networkidle", timeout=15_000)
            except Exception:
                pass
            _dismiss_overlays(page)
            page.wait_for_timeout(1500)

            canonical = page.url or url
            url_meta = _metadata_from_url(canonical)
            dom_data = _extract_from_dom(page)
            booking_params = _booking_params_from_url(canonical)
            package_code = booking_params.get("packageCode") or ""

            group_id = None
            if api_request_url:
                group_id = parse_qs(urlparse(api_request_url).query).get("groupId", [None])[0]
            if not group_id:
                group_id = parse_qs(urlparse(canonical).query).get("groupId", [None])[0]

            sail_dates: list[dict[str, Any]] = []
            for entry in _parse_sailings_response(captured_payload):
                sail_url = _itinerary_url_for_sail_date(
                    canonical,
                    entry["sailing_date"],
                    package_code=str(entry.get("package_code") or package_code or ""),
                    group_id=group_id,
                )
                sail_dates.append(
                    {
                        "sailing_date": entry["sailing_date"],
                        "url": sail_url,
                        "package_code": entry.get("package_code") or package_code,
                        "group_id": group_id,
                    }
                )

            if not sail_dates:
                sail_dates.append(
                    {
                        "sailing_date": (
                            dom_data.get("sailing_date")
                            or url_meta.get("sailing_date")
                            or "TBD"
                        ),
                        "url": canonical,
                        "package_code": package_code,
                        "group_id": group_id,
                    }
                )

            return {
                "ship_name": dom_data.get("ship_name") or url_meta.get("ship_name"),
                "duration": dom_data.get("duration") or url_meta.get("duration"),
                "departure_port": dom_data.get("departure_port") or url_meta.get("departure_port"),
                "itinerary": dom_data.get("itinerary"),
                "package_code": package_code,
                "group_id": group_id,
                "sail_dates": sail_dates,
            }
        finally:
            context.close()
            browser.close()


def cruise_url_is_tracked(conn: sqlite3.Connection, url: str) -> bool:
    """Return True if this exact URL is already in the cruises table."""
    row = conn.execute("SELECT 1 FROM cruises WHERE url = ? LIMIT 1", (url,)).fetchone()
    return row is not None


def scrape_cruise_url(url: str, timeout_ms: int = 60_000) -> dict[str, Any]:
    """
    Scrape a Royal Caribbean URL for cabin tiers, Voom pricing, and cruise metadata.
    Returns a dictionary ready for persistence or UI display.
    """
    captured_responses: list[Response] = []

    with sync_playwright() as playwright:
        browser, context = _browser_context(playwright)
        page = context.new_page()

        def on_response(response: Response) -> None:
            captured_responses.append(response)

        page.on("response", on_response)

        try:
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            _dismiss_overlays(page)
            page.wait_for_timeout(2500)
            try:
                page.wait_for_load_state("networkidle", timeout=15_000)
            except Exception:
                pass
            _dismiss_overlays(page)
            page.wait_for_timeout(1500)

            url_meta = _metadata_from_url(url)
            dom_data = _extract_from_dom(page)
            tab_prices = _extract_prices_by_category_tabs(page)
            api_prices = _extract_from_api_payloads(_collect_api_payloads(captured_responses))

            prices = {col: None for col in PRICE_COLUMNS}
            for column in PRICE_COLUMNS:
                if column == "voom_price":
                    continue
                value = tab_prices.get(column)
                if value is None and not _prices_all_identical(api_prices):
                    value = api_prices.get(column)
                prices[column] = value

            nights = parse_night_count(
                dom_data.get("duration") or url_meta.get("duration"), url
            )
            prices["voom_price"] = _scrape_voom_via_booking_flow(
                page, url, captured_responses, nights=nights
            )

            canonical_url = page.url or url
            return {
                "url": canonical_url,
                "ship_name": (
                    dom_data.get("ship_name")
                    or url_meta.get("ship_name")
                    or "Unknown Ship"
                ),
                "sailing_date": (
                    dom_data.get("sailing_date")
                    or url_meta.get("sailing_date")
                    or "TBD"
                ),
                "duration": dom_data.get("duration") or url_meta.get("duration"),
                "departure_port": dom_data.get("departure_port") or url_meta.get("departure_port"),
                "itinerary": dom_data.get("itinerary"),
                **prices,
                "scraped_at": utc_now_iso(),
            }
        finally:
            context.close()
            browser.close()


# ---------------------------------------------------------------------------
# Persistence + price-change detection
# ---------------------------------------------------------------------------


def get_cruise_by_id(conn: sqlite3.Connection, cruise_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM cruises WHERE id = ?", (cruise_id,)).fetchone()


def get_all_cruises(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM cruises ORDER BY sailing_date, ship_name").fetchall()


def get_latest_pricing(conn: sqlite3.Connection, cruise_id: int) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT * FROM pricing_history
        WHERE cruise_id = ?
        ORDER BY timestamp DESC
        LIMIT 1
        """,
        (cruise_id,),
    ).fetchone()


def get_pricing_history(conn: sqlite3.Connection, cruise_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT * FROM pricing_history
        WHERE cruise_id = ?
        ORDER BY timestamp ASC
        """,
        (cruise_id,),
    ).fetchall()


def delete_cruise(conn: sqlite3.Connection, cruise_id: int) -> bool:
    """Remove a cruise and its pricing history (ON DELETE CASCADE)."""
    cursor = conn.execute("DELETE FROM cruises WHERE id = ?", (cruise_id,))
    return cursor.rowcount > 0


def set_manual_voom_price(
    conn: sqlite3.Connection, cruise_id: int, daily_price: float
) -> None:
    """Record a manual Voom per-device daily price (e.g. from Cruise Planner)."""
    latest = get_latest_pricing(conn, cruise_id)
    conn.execute(
        """
        INSERT INTO pricing_history (
            timestamp, cruise_id,
            interior_price, oceanview_price, balcony_price, suite_price, voom_price
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            utc_now_iso(),
            cruise_id,
            latest["interior_price"] if latest else None,
            latest["oceanview_price"] if latest else None,
            latest["balcony_price"] if latest else None,
            latest["suite_price"] if latest else None,
            daily_price,
        ),
    )


def add_cruise(conn: sqlite3.Connection, scraped: dict[str, Any]) -> int:
    conn.execute(
        """
        INSERT INTO cruises (ship_name, sailing_date, duration, departure_port, itinerary, url)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            scraped.get("ship_name") or "Unknown Ship",
            scraped.get("sailing_date") or "TBD",
            scraped.get("duration"),
            scraped.get("departure_port"),
            scraped.get("itinerary"),
            scraped["url"],
        ),
    )
    return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])


def insert_pricing_snapshot(conn: sqlite3.Connection, cruise_id: int, scraped: dict[str, Any]) -> int:
    row_id = conn.execute(
        """
        INSERT INTO pricing_history (
            timestamp, cruise_id,
            interior_price, oceanview_price, balcony_price, suite_price, voom_price
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            scraped.get("scraped_at") or utc_now_iso(),
            cruise_id,
            scraped.get("interior_price"),
            scraped.get("oceanview_price"),
            scraped.get("balcony_price"),
            scraped.get("suite_price"),
            scraped.get("voom_price"),
        ),
    ).lastrowid
    return int(row_id)


def detect_cabin_inversions(prices: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Flag when a higher stateroom tier is priced at or below a lower tier.
    Example: Balcony <= Interior.
    """
    inversions: list[dict[str, Any]] = []
    for i, (lower_col, lower_label) in enumerate(CABIN_TIER_ORDER):
        lower_val = prices.get(lower_col)
        if lower_val is None:
            continue
        lower_f = float(lower_val)
        for higher_col, higher_label in CABIN_TIER_ORDER[i + 1 :]:
            higher_val = prices.get(higher_col)
            if higher_val is None:
                continue
            higher_f = float(higher_val)
            if higher_f <= lower_f + 0.01:
                inversions.append(
                    {
                        "lower_tier": lower_label,
                        "lower_field": lower_col,
                        "lower_price": lower_f,
                        "higher_tier": higher_label,
                        "higher_field": higher_col,
                        "higher_price": higher_f,
                        "delta": round(lower_f - higher_f, 2),
                    }
                )
    return inversions


def compare_to_previous(
    previous: sqlite3.Row | None, current: dict[str, Any]
) -> list[dict[str, Any]]:
    """Return structured alerts when prices move vs. the last snapshot."""
    if previous is None:
        return []

    alerts: list[dict[str, Any]] = []
    for column in PRICE_COLUMNS:
        old_val = previous[column]
        new_val = current.get(column)
        if old_val is None or new_val is None:
            continue
        old_f, new_f = float(old_val), float(new_val)
        if abs(new_f - old_f) < 0.01:
            continue
        direction = "decrease" if new_f < old_f else "increase"
        alerts.append(
            {
                "cruise_id": previous["cruise_id"],
                "field": column,
                "label": CABIN_LABELS[column],
                "previous": old_f,
                "current": new_f,
                "delta": round(new_f - old_f, 2),
                "direction": direction,
            }
        )
    return alerts


def scrape_and_store_cruise(
    conn: sqlite3.Connection, cruise_id: int, url: str
) -> tuple[dict, list[dict], list[dict]]:
    previous = get_latest_pricing(conn, cruise_id)
    scraped = scrape_cruise_url(url)
    insert_pricing_snapshot(conn, cruise_id, scraped)
    alerts = compare_to_previous(previous, scraped)
    inversions = detect_cabin_inversions(scraped)
    return scraped, alerts, inversions


def run_full_scrape(db_path: Path | None = None) -> dict[str, Any]:
    """
    Scrape every tracked cruise, persist snapshots, and return aggregate results.
    """
    path = db_path or DB_PATH
    init_database(path)
    summary: dict[str, Any] = {
        "scraped": [],
        "alerts": [],
        "inversions": [],
        "errors": [],
    }

    with get_connection(path) as conn:
        cruises = get_all_cruises(conn)
        for cruise in cruises:
            try:
                scraped, alerts, inversions = scrape_and_store_cruise(
                    conn, cruise["id"], cruise["url"]
                )
                summary["scraped"].append(
                    {
                        "cruise_id": cruise["id"],
                        "ship_name": cruise["ship_name"],
                        "data": scraped,
                    }
                )
                for alert in alerts:
                    alert["ship_name"] = cruise["ship_name"]
                    summary["alerts"].append(alert)
                for inv in inversions:
                    inv["ship_name"] = cruise["ship_name"]
                    inv["cruise_id"] = cruise["id"]
                    summary["inversions"].append(inv)
            except Exception as exc:
                summary["errors"].append(
                    {"cruise_id": cruise["id"], "ship_name": cruise["ship_name"], "error": str(exc)}
                )
            time.sleep(1.5)

    return summary


if __name__ == "__main__":
    init_database()
    result = run_full_scrape()
    print(json.dumps(result, indent=2, default=str))
