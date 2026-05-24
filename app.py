"""
Royal Caribbean Cruise Tracker — Streamlit local dashboard.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from predictions import (
    CABIN_FORECAST_COLUMNS,
    build_forecast_chart_frame,
    forecast_all_cabins,
)
from scraper import (
    CABIN_LABELS,
    DB_PATH,
    PRICE_COLUMNS,
    PlaywrightSetupError,
    add_cruise,
    check_playwright_ready,
    cruise_url_is_tracked,
    delete_cruise,
    detect_cabin_inversions,
    discover_itinerary_sail_dates,
    filter_sail_dates_by_month,
    get_all_cruises,
    is_itinerary_url,
    get_connection,
    get_cruise_by_id,
    get_latest_pricing,
    get_pricing_history,
    init_database,
    insert_pricing_snapshot,
    parse_night_count,
    run_full_scrape,
    scrape_cruise_url,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Royal Caribbean Cruise Tracker",
    page_icon="🛳️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .price-down {
        background-color: #d4edda;
        border-left: 6px solid #28a745;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 6px;
        color: #155724;
        font-weight: 600;
    }
    .price-up {
        background-color: #f8d7da;
        border-left: 6px solid #dc3545;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 6px;
        color: #721c24;
        font-weight: 600;
    }
    .price-neutral {
        background-color: #fff3cd;
        border-left: 6px solid #fd7e14;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 6px;
        color: #856404;
    }
    .price-inversion {
        background-color: #e8daef;
        border-left: 6px solid #8e44ad;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 6px;
        color: #4a235a;
        font-weight: 600;
    }
    /* Tracked cruises: pinned header row above scrollable body */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.tracked-cruises-table-marker)
        > div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"]:first-child {
        position: sticky;
        top: 0;
        z-index: 2;
        background: var(--secondary-background-color, #f0f2f6);
        border-bottom: 1px solid rgba(49, 51, 63, 0.2);
        padding-bottom: 0.35rem;
        margin-bottom: 0.15rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _format_money(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return f"${float(value):,.2f}"


def _format_total_per_night(total: Any, nights: int) -> str:
    """Format cabin fare as $664/$166 (total per person / per night)."""
    if total is None or (isinstance(total, float) and pd.isna(total)) or nights < 1:
        return "—"
    total_f = float(total)
    per_night = round(total_f / nights)
    return f"${total_f:,.0f}/${per_night:,.0f}"


def _format_sail_date_label(sail_date: str) -> str:
    try:
        return datetime.strptime(sail_date[:10], "%Y-%m-%d").strftime("%a %b %d, %Y")
    except ValueError:
        return sail_date


def _month_label(month_key: str) -> str:
    try:
        return datetime.strptime(month_key, "%Y-%m").strftime("%B %Y")
    except ValueError:
        return month_key


def _sail_date_from_url(url: str) -> str | None:
    from urllib.parse import parse_qs, urlparse

    query = parse_qs(urlparse(url).query)
    for key in ("sailDate", "sail-date", "sail_date"):
        if query.get(key):
            return query[key][0][:10]
    return None


def _apply_itinerary_metadata(scraped: dict[str, Any], discovery: dict[str, Any]) -> None:
    if not scraped.get("ship_name") or scraped["ship_name"] == "Unknown Ship":
        if discovery.get("ship_name"):
            scraped["ship_name"] = discovery["ship_name"]
    if not scraped.get("duration") and discovery.get("duration"):
        scraped["duration"] = discovery["duration"]
    if not scraped.get("departure_port") and discovery.get("departure_port"):
        scraped["departure_port"] = discovery["departure_port"]
    if not scraped.get("itinerary") and discovery.get("itinerary"):
        scraped["itinerary"] = discovery["itinerary"]


def _add_tracked_sail_dates(
    targets: list[dict[str, Any]],
    discovery: dict[str, Any] | None,
    *,
    progress_label: str = "Checking prices",
) -> tuple[int, int, list[str], str]:
    """Scrape and persist selected sail dates. Returns added, skipped, dates, ship name."""
    added = 0
    skipped = 0
    added_dates: list[str] = []
    ship_label = (discovery or {}).get("ship_name") or "cruise"
    progress = st.progress(0.0, text="Starting price checks…")

    for index, target in enumerate(targets):
        sail_url = str(target["url"])
        sail_label = str(target.get("sailing_date") or sail_url)
        progress.progress(
            index / max(len(targets), 1),
            text=f"{progress_label} for {sail_label} ({index + 1}/{len(targets)})…",
        )
        scraped = scrape_cruise_url(sail_url)
        if discovery:
            _apply_itinerary_metadata(scraped, discovery)
        if scraped.get("ship_name"):
            ship_label = scraped["ship_name"]

        with get_connection() as conn:
            if cruise_url_is_tracked(conn, scraped["url"]) or cruise_url_is_tracked(conn, sail_url):
                skipped += 1
                continue
            cruise_id = add_cruise(conn, scraped)
            insert_pricing_snapshot(conn, cruise_id, scraped)
            added += 1
            added_dates.append(str(scraped.get("sailing_date") or sail_label))

    progress.progress(1.0, text="Done.")
    progress.empty()
    return added, skipped, added_dates, ship_label


def _pricing_to_dataframe(rows: list[sqlite3.Row]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    # Streamlit charts are happiest with timezone-naive datetimes
    if hasattr(df["timestamp"].dtype, "tz") and df["timestamp"].dtype.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    return df.sort_values("timestamp")


def render_price_alerts(alerts: list[dict[str, Any]]) -> None:
    if not alerts:
        return

    st.subheader("Price change alerts")
    for alert in alerts:
        label = alert["label"]
        ship = alert.get("ship_name", "Cruise")
        prev = _format_money(alert["previous"])
        curr = _format_money(alert["current"])
        delta = alert["delta"]
        if alert["direction"] == "decrease":
            message = (
                f"🟢 <b>{ship}</b> — <b>{label}</b> dropped from {prev} to {curr} "
                f"(<b>${abs(delta):,.2f}</b> savings)"
            )
            st.markdown(f'<div class="price-down">{message}</div>', unsafe_allow_html=True)
        else:
            message = (
                f"🔴 <b>{ship}</b> — <b>{label}</b> increased from {prev} to {curr} "
                f"(+${delta:,.2f})"
            )
            st.markdown(f'<div class="price-up">{message}</div>', unsafe_allow_html=True)


def render_inversion_alerts(inversions: list[dict[str, Any]]) -> None:
    if not inversions:
        return

    st.subheader("Cabin tier inversion alerts")
    st.caption(
        "A higher tier is priced at or below a lower tier — often a promo or data glitch worth checking."
    )
    for inv in inversions:
        ship = inv.get("ship_name", "Cruise")
        lower = _format_money(inv["lower_price"])
        higher = _format_money(inv["higher_price"])
        if inv["higher_price"] < inv["lower_price"]:
            detail = f"{inv['higher_tier']} is <b>${inv['delta']:,.2f}</b> cheaper than {inv['lower_tier']}"
        else:
            detail = f"{inv['higher_tier']} matches {inv['lower_tier']} at the same price"
        message = (
            f"🟣 <b>{ship}</b> — <b>{inv['higher_tier']}</b> ({higher}) vs "
            f"<b>{inv['lower_tier']}</b> ({lower}): {detail}"
        )
        st.markdown(f'<div class="price-inversion">{message}</div>', unsafe_allow_html=True)


def collect_current_inversions() -> list[dict[str, Any]]:
    """Scan latest snapshots for active tier inversions across all cruises."""
    active: list[dict[str, Any]] = []
    with get_connection() as conn:
        for cruise in get_all_cruises(conn):
            latest = get_latest_pricing(conn, cruise["id"])
            if not latest:
                continue
            prices = {col: latest[col] for col in PRICE_COLUMNS if col != "voom_price"}
            for inv in detect_cabin_inversions(prices):
                inv["ship_name"] = cruise["ship_name"]
                inv["cruise_id"] = cruise["id"]
                active.append(inv)
    return active


def render_price_predictions(cruise_id: int) -> None:
    with get_connection() as conn:
        cruise = get_cruise_by_id(conn, cruise_id)
        history = get_pricing_history(conn, cruise_id)

    if not cruise or not history:
        return

    forecasts = forecast_all_cabins(history, cruise)
    st.subheader("AI price forecast")
    st.caption(
        "Tree models (Random Forest, XGBoost, LightGBM) trained on calendar features "
        "(days until sailing, weekends, US holidays, peak season), price lags, and "
        "cross-tier prices. Best model is chosen by time-series cross-validation MAE."
    )

    if not forecasts:
        st.info(
            "Not enough history yet for forecasting. After at least 6 snapshots per cruise, "
            "7-day and 14-day projections will appear here."
        )
        return

    nights = parse_night_count(cruise["duration"], cruise["url"])
    rows = []
    for column in CABIN_FORECAST_COLUMNS:
        if column not in forecasts:
            continue
        f = forecasts[column]
        rows.append(
            {
                "Cabin": CABIN_LABELS[column],
                "Current": _format_total_per_night(f["current"], nights),
                "7-day forecast": _format_total_per_night(f["day_7"], nights),
                "14-day forecast": _format_total_per_night(f["day_14"], nights),
                "Trend": f["trend"],
                "Model": f["model"],
                "CV MAE": f"${f['cv_mae']:,.0f}",
                "Confidence": f["confidence"],
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    forecast_tabs = st.tabs([CABIN_LABELS[c] for c in CABIN_FORECAST_COLUMNS])
    for tab, column in zip(forecast_tabs, CABIN_FORECAST_COLUMNS):
        with tab:
            chart = build_forecast_chart_frame(history, cruise, column)
            if chart is None:
                st.write(f"No {CABIN_LABELS[column]} history yet.")
                continue
            st.line_chart(chart, height=280)
            if column in forecasts:
                f = forecasts[column]
                top_feats = ", ".join(f["top_features"].keys()) if f.get("top_features") else "—"
                st.caption(
                    f"Projected {f['date_7']}: {_format_total_per_night(f['day_7'], nights)} · "
                    f"Projected {f['date_14']}: {_format_total_per_night(f['day_14'], nights)} · "
                    f"Model: **{f['model']}** · CV MAE ${f['cv_mae']:,.0f} · "
                    f"Confidence: {f['confidence']} · Top features: {top_feats}"
                )


CRUISE_TABLE_COLUMNS = (2.2, 1.4, 1, 1, 1, 1, 1, 1.4, 0.35)
CRUISE_TABLE_HEADERS = (
    "Ship",
    "Sailing Date",
    "Duration",
    "Interior",
    "Ocean View",
    "Balcony",
    "Suite",
    "Last Check",
    "",
)
CRUISE_TABLE_SCROLL_HEIGHT = 520
CRUISE_TABLE_SCROLL_THRESHOLD = 6


def _render_cruise_table_header() -> None:
    header_cols = st.columns(CRUISE_TABLE_COLUMNS)
    for col, label in zip(header_cols, CRUISE_TABLE_HEADERS):
        col.markdown(f"**{label}**")


def render_cruise_dashboard() -> None:
    with get_connection() as conn:
        cruises = get_all_cruises(conn)

    if not cruises:
        st.info("No cruises tracked yet. Add a Royal Caribbean URL in the sidebar.")
        return

    if "pending_delete_cruise_id" not in st.session_state:
        st.session_state.pending_delete_cruise_id = None

    st.caption(
        "Cabin prices are **per person** (total/night). "
        "Click **×** to stop tracking a sail date."
    )

    with st.container(border=True):
        st.markdown('<div class="tracked-cruises-table-marker"></div>', unsafe_allow_html=True)
        _render_cruise_table_header()

        use_scroll = len(cruises) > CRUISE_TABLE_SCROLL_THRESHOLD
        if use_scroll:
            list_container: Any = st.container(height=CRUISE_TABLE_SCROLL_HEIGHT)
        else:
            list_container = st.container()

        with list_container:
            for cruise in cruises:
                with get_connection() as conn:
                    latest = get_latest_pricing(conn, cruise["id"])
                nights = parse_night_count(cruise["duration"], cruise["url"])
                cruise_id = int(cruise["id"])

                row_cols = st.columns(CRUISE_TABLE_COLUMNS)
                row_cols[0].write(cruise["ship_name"])
                row_cols[1].write(cruise["sailing_date"])
                row_cols[2].write(cruise["duration"] or "—")
                row_cols[3].write(
                    _format_total_per_night(latest["interior_price"] if latest else None, nights)
                )
                row_cols[4].write(
                    _format_total_per_night(latest["oceanview_price"] if latest else None, nights)
                )
                row_cols[5].write(
                    _format_total_per_night(latest["balcony_price"] if latest else None, nights)
                )
                row_cols[6].write(
                    _format_total_per_night(latest["suite_price"] if latest else None, nights)
                )
                row_cols[7].write(
                    latest["timestamp"][:19].replace("T", " ")
                    if latest and latest["timestamp"]
                    else "Never"
                )
                with row_cols[8]:
                    if st.button(
                        "×",
                        key=f"remove_cruise_{cruise_id}",
                        help="Stop tracking this sail date",
                    ):
                        st.session_state.pending_delete_cruise_id = cruise_id
                        st.rerun()

                if st.session_state.pending_delete_cruise_id == cruise_id:
                    st.warning(
                        f"Remove **{cruise['ship_name']}** sailing **{cruise['sailing_date']}**? "
                        "All price history for this departure will be deleted."
                    )
                    confirm_col, cancel_col = st.columns(2)
                    if confirm_col.button(
                        "Yes, remove this sail date",
                        key=f"confirm_remove_{cruise_id}",
                        type="primary",
                    ):
                        with get_connection() as conn:
                            delete_cruise(conn, cruise_id)
                        st.session_state.pending_delete_cruise_id = None
                        st.session_state.pop("last_scrape_summary", None)
                        st.session_state["last_delete_success"] = (
                            f"Removed **{cruise['ship_name']}** sailing **{cruise['sailing_date']}**."
                        )
                        st.rerun()
                    if cancel_col.button("Cancel", key=f"cancel_remove_{cruise_id}"):
                        st.session_state.pending_delete_cruise_id = None
                        st.rerun()

                st.divider()


def render_price_history(cruise_id: int) -> None:
    with get_connection() as conn:
        cruise = get_cruise_by_id(conn, cruise_id)
        history = get_pricing_history(conn, cruise_id)

    if not cruise:
        st.warning("Cruise not found.")
        return

    st.subheader(f"{cruise['ship_name']} — price history")
    st.caption(
        f"Sailing {cruise['sailing_date']} · {cruise['duration'] or 'Duration TBD'} · "
        f"From {cruise['departure_port'] or 'TBD'}"
    )
    if cruise["itinerary"]:
        st.write(cruise["itinerary"])
    st.link_button("Open Royal Caribbean page", cruise["url"])

    df = _pricing_to_dataframe(history)
    if df.empty:
        st.info("No price history yet. Run **Check Prices Now** to capture the first snapshot.")
        return

    chart_labels = {
        "interior_price": "Interior",
        "oceanview_price": "Ocean View",
        "balcony_price": "Balcony",
        "suite_price": "Suite",
    }

    tabs = st.tabs(list(chart_labels.values()))
    for tab, (column, label) in zip(tabs, chart_labels.items()):
        with tab:
            series = df[["timestamp", column]].dropna(subset=[column])
            if series.empty:
                st.write(f"No {label} prices recorded yet.")
                continue
            chart_df = series.set_index("timestamp")[[column]].copy()
            chart_df.columns = [label]
            st.line_chart(chart_df, height=320)

    nights = parse_night_count(cruise["duration"], cruise["url"])
    with st.expander("Raw pricing history"):
        display = df.copy()
        display["timestamp"] = display["timestamp"].dt.strftime("%Y-%m-%d %H:%M UTC")
        for col in PRICE_COLUMNS:
            if col == "voom_price":
                continue
            display[col] = display[col].apply(
                lambda v: _format_total_per_night(v, nights) if pd.notna(v) else "—"
            )
        st.dataframe(display, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Sidebar — add cruise + manual scrape
# ---------------------------------------------------------------------------

init_database()

if "last_error" not in st.session_state:
    st.session_state["last_error"] = None

st.title("🛳️ Royal Caribbean Cruise Tracker")
st.caption(f"Local SQLite database: `{DB_PATH}`")

playwright_ok, playwright_message = check_playwright_ready()
if not playwright_ok:
    st.error(playwright_message)
elif st.session_state.get("show_playwright_ok"):
    st.success(playwright_message)

if st.session_state.get("last_error"):
    st.error(f"**Last operation failed:** {st.session_state['last_error']}")

with st.sidebar:
    st.header("Track a new cruise")
    new_url = st.text_input(
        "Royal Caribbean URL",
        placeholder="https://www.royalcaribbean.com/...",
        key="new_cruise_url",
    )
    url = new_url.strip()
    is_itinerary = bool(url) and is_itinerary_url(url)

    if url and url != st.session_state.get("itinerary_discovery_url", ""):
        st.session_state.pop("itinerary_discovery", None)

    if is_itinerary:
        st.caption(
            "Itinerary link detected. Find available departures, filter by month, "
            "then choose which sail dates to track (~30–90s per date)."
        )
        if st.button("Find sail dates", use_container_width=True, disabled=not playwright_ok):
            st.session_state["last_error"] = None
            if not playwright_ok:
                st.error("Fix Playwright setup (see main page) before adding cruises.")
            else:
                try:
                    with st.spinner("Loading available sail dates…"):
                        st.session_state["itinerary_discovery"] = discover_itinerary_sail_dates(url)
                        st.session_state["itinerary_discovery_url"] = url
                except PlaywrightSetupError as exc:
                    st.session_state["last_error"] = str(exc)
                    st.error(str(exc))
                except Exception as exc:
                    st.session_state["last_error"] = f"Could not load sail dates: {exc}"
                    st.error(st.session_state["last_error"])

        discovery = st.session_state.get("itinerary_discovery")
        if discovery and st.session_state.get("itinerary_discovery_url") == url:
            sail_dates: list[dict[str, Any]] = discovery.get("sail_dates") or []
            ship_name = discovery.get("ship_name") or "this itinerary"
            st.success(f"**{ship_name}** — {len(sail_dates)} open departure(s)")

            month_keys = sorted(
                {str(entry["sailing_date"])[:7] for entry in sail_dates if entry.get("sailing_date")}
            )
            month_options = {"all": "All months"}
            for month_key in month_keys:
                count = sum(
                    1 for entry in sail_dates if str(entry.get("sailing_date", "")).startswith(month_key)
                )
                month_options[month_key] = f"{_month_label(month_key)} ({count})"

            selected_month = st.selectbox(
                "Filter by month",
                options=list(month_options.keys()),
                format_func=lambda key: month_options[key],
                key="sail_date_month_filter",
            )
            filtered_dates = filter_sail_dates_by_month(sail_dates, selected_month)

            date_labels = {
                _format_sail_date_label(str(entry["sailing_date"])): entry
                for entry in filtered_dates
            }
            multiselect_key = f"selected_sail_dates_{selected_month}"
            if multiselect_key not in st.session_state:
                url_sail_date = _sail_date_from_url(url)
                st.session_state[multiselect_key] = [
                    label
                    for label, entry in date_labels.items()
                    if str(entry.get("sailing_date", ""))[:10] == url_sail_date
                ]

            if st.checkbox(
                "Select all dates in this month",
                value=False,
                key=f"select_all_sail_dates_{selected_month}",
            ):
                st.session_state[multiselect_key] = list(date_labels.keys())

            selected_labels = st.multiselect(
                "Departure dates to track",
                options=list(date_labels.keys()),
                key=multiselect_key,
            )

            if st.button("Add selected sail dates", type="primary", use_container_width=True):
                st.session_state["last_error"] = None
                if not selected_labels:
                    msg = "Select at least one departure date."
                    st.session_state["last_error"] = msg
                    st.warning(msg)
                elif not playwright_ok:
                    st.session_state["last_error"] = playwright_message
                    st.error("Fix Playwright setup (see main page) before adding cruises.")
                else:
                    try:
                        targets = [date_labels[label] for label in selected_labels]
                        added, skipped, added_dates, ship_label = _add_tracked_sail_dates(
                            targets, discovery
                        )
                        if added:
                            st.session_state["last_error"] = None
                            date_summary = ", ".join(added_dates[:4])
                            if len(added_dates) > 4:
                                date_summary += f", +{len(added_dates) - 4} more"
                            st.session_state["last_add_success"] = (
                                f"Now tracking **{ship_label}** on **{added} sail date(s)** "
                                f"({date_summary})."
                            )
                            if skipped:
                                st.session_state["last_add_success"] += (
                                    f" Skipped {skipped} date(s) already being tracked."
                                )
                            st.rerun()
                        elif skipped:
                            msg = "All selected sail dates are already being tracked."
                            st.session_state["last_error"] = msg
                            st.warning(msg)
                    except PlaywrightSetupError as exc:
                        st.session_state["last_error"] = str(exc)
                        st.error(str(exc))
                    except Exception as exc:
                        msg = f"Could not add cruise: {exc}"
                        st.session_state["last_error"] = msg
                        st.error(msg)
    else:
        st.caption("Paste a Royal Caribbean URL, then click **Add cruise** (~30–90 seconds).")

        if st.button("Add cruise", type="primary", use_container_width=True):
            st.session_state["last_error"] = None
            if not url:
                msg = "Enter a valid Royal Caribbean URL."
                st.session_state["last_error"] = msg
                st.error(msg)
            elif "royalcaribbean.com" not in url.lower():
                msg = "URL should be from royalcaribbean.com."
                st.session_state["last_error"] = msg
                st.warning(msg)
            elif not playwright_ok:
                st.session_state["last_error"] = playwright_message
                st.error("Fix Playwright setup (see main page) before adding cruises.")
            else:
                try:
                    added, skipped, added_dates, ship_label = _add_tracked_sail_dates(
                        [{"url": url, "sailing_date": "selected date"}],
                        None,
                    )
                    if added:
                        st.session_state["last_error"] = None
                        st.session_state["last_add_success"] = (
                            f"Now tracking **{ship_label}** (sailing {added_dates[0]})."
                        )
                        if skipped:
                            st.session_state["last_add_success"] += " That cruise was already tracked."
                        st.rerun()
                    elif skipped:
                        msg = "That cruise is already being tracked."
                        st.session_state["last_error"] = msg
                        st.warning(msg)
                except PlaywrightSetupError as exc:
                    st.session_state["last_error"] = str(exc)
                    st.error(str(exc))
                except Exception as exc:
                    msg = f"Could not add cruise: {exc}"
                    st.session_state["last_error"] = msg
                    st.error(msg)

    st.divider()
    st.header("Price check")
    if st.button("Check Prices Now", use_container_width=True):
        st.session_state["run_scrape"] = True

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

if add_success := st.session_state.pop("last_add_success", None):
    st.success(add_success)

if delete_success := st.session_state.pop("last_delete_success", None):
    st.success(delete_success)

if st.session_state.pop("run_scrape", False):
    st.session_state["last_error"] = None
    if not playwright_ok:
        st.session_state["last_error"] = playwright_message
        st.error(playwright_message)
    else:
        with st.spinner("Running Playwright scraper for all tracked cruises…"):
            try:
                summary = run_full_scrape()
                st.session_state["last_scrape_summary"] = summary
                st.session_state["last_scrape_at"] = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            except PlaywrightSetupError as exc:
                st.session_state["last_error"] = str(exc)
                st.error(str(exc))
            except Exception as exc:
                msg = f"Scrape failed: {exc}"
                st.session_state["last_error"] = msg
                st.error(msg)

summary = st.session_state.get("last_scrape_summary")
if summary:
    scraped_at = st.session_state.get("last_scrape_at", "")
    st.success(f"Last price check completed at {scraped_at}.")
    scraped_count = len(summary.get("scraped", []))
    if scraped_count == 0 and not summary.get("errors"):
        st.warning("No tracked cruises were updated. Add a cruise first (sidebar).")
    render_price_alerts(summary.get("alerts", []))
    render_inversion_alerts(summary.get("inversions", []))
    if summary.get("errors"):
        for err in summary["errors"]:
            st.markdown(
                f'<div class="price-neutral">⚠️ <b>{err["ship_name"]}</b>: {err["error"]}</div>',
                unsafe_allow_html=True,
            )

st.header("Tracked cruises")
render_cruise_dashboard()

active_inversions = collect_current_inversions()
if active_inversions:
    st.header("Active tier inversions")
    render_inversion_alerts(active_inversions)

st.header("Cruise detail & trends")
with get_connection() as conn:
    cruises = get_all_cruises(conn)

if cruises:
    options = {
        f"{c['ship_name']} — {c['sailing_date']} (ID {c['id']})": c["id"] for c in cruises
    }
    choice = st.selectbox("Select a cruise", list(options.keys()))
    render_price_predictions(options[choice])
    render_price_history(options[choice])
else:
    st.write("Add a cruise URL in the sidebar to begin tracking.")
