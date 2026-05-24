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
    get_all_cruises,
    is_itinerary_url,
    set_manual_voom_price,
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


def _format_voom_price(daily: Any, nights: int) -> str:
    """Format Voom as $88/$22 (voyage total per device / per day)."""
    if daily is None or (isinstance(daily, float) and pd.isna(daily)) or nights < 1:
        return "—"
    daily_f = float(daily)
    voyage_total = round(daily_f * nights)
    return f"${voyage_total:,.0f}/${daily_f:,.0f}"


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


def render_cruise_dashboard() -> None:
    with get_connection() as conn:
        cruises = get_all_cruises(conn)

    if not cruises:
        st.info("No cruises tracked yet. Add a Royal Caribbean URL in the sidebar.")
        return

    records = []
    for cruise in cruises:
        with get_connection() as conn:
            latest = get_latest_pricing(conn, cruise["id"])
        nights = parse_night_count(cruise["duration"], cruise["url"])
        records.append(
            {
                "ID": cruise["id"],
                "Ship": cruise["ship_name"],
                "Sailing Date": cruise["sailing_date"],
                "Duration": cruise["duration"] or "—",
                "Departure": cruise["departure_port"] or "—",
                "Interior": _format_total_per_night(
                    latest["interior_price"] if latest else None, nights
                ),
                "Ocean View": _format_total_per_night(
                    latest["oceanview_price"] if latest else None, nights
                ),
                "Balcony": _format_total_per_night(
                    latest["balcony_price"] if latest else None, nights
                ),
                "Suite": _format_total_per_night(
                    latest["suite_price"] if latest else None, nights
                ),
                "Voom (device)": _format_voom_price(
                    latest["voom_price"] if latest else None, nights
                ),
                "Last Check": (
                    latest["timestamp"][:19].replace("T", " ")
                    if latest and latest["timestamp"]
                    else "Never"
                ),
            }
        )

    st.caption(
        "Cabin prices are **per person** (total/night). "
        "Voom is **per device** (voyage total/day). Royal Caribbean often only lists "
        "internet in **Cruise Planner** after booking—use the sidebar to enter it manually "
        "if auto-scrape shows —."
    )
    st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)


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
        "voom_price": "Voom (Internet)",
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
                display[col] = display[col].apply(
                    lambda v: _format_voom_price(v, nights) if pd.notna(v) else "—"
                )
            else:
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
    )
    track_all_dates = st.checkbox(
        "Track all available sail dates for this itinerary",
        value=True,
        help=(
            "Royal Caribbean prices change by departure date. "
            "When enabled, every open sail date for the itinerary is tracked separately."
        ),
    )
    st.caption(
        "Paste an itinerary URL, then click **Add cruise**. "
        "Tracking all sail dates runs one price check per departure (~30–90s each)."
    )

    if st.button("Add cruise", type="primary", use_container_width=True):
        st.session_state["last_error"] = None
        if not new_url.strip():
            msg = "Enter a valid Royal Caribbean URL."
            st.session_state["last_error"] = msg
            st.error(msg)
        elif "royalcaribbean.com" not in new_url.lower():
            msg = "URL should be from royalcaribbean.com."
            st.session_state["last_error"] = msg
            st.warning(msg)
        elif not playwright_ok:
            st.session_state["last_error"] = playwright_message
            st.error("Fix Playwright setup (see main page) before adding cruises.")
        else:
            url = new_url.strip()
            use_all_dates = track_all_dates and is_itinerary_url(url)
            try:
                targets: list[dict[str, str]] = []
                ship_label = "cruise"
                discovery: dict[str, Any] = {}

                if use_all_dates:
                    with st.spinner("Finding available sail dates for this itinerary…"):
                        discovery = discover_itinerary_sail_dates(url)
                    targets = discovery.get("sail_dates") or []
                    ship_label = discovery.get("ship_name") or ship_label
                    if not targets:
                        msg = "No open sail dates were found for that itinerary."
                        st.session_state["last_error"] = msg
                        st.warning(msg)
                else:
                    targets = [{"url": url, "sailing_date": "selected date"}]

                if targets:
                    progress = st.progress(0.0, text="Starting price checks…")
                    added = 0
                    skipped = 0
                    total_prices = 0
                    added_dates: list[str] = []

                    for index, target in enumerate(targets):
                        sail_url = target["url"]
                        sail_label = target.get("sailing_date") or sail_url
                        progress.progress(
                            index / max(len(targets), 1),
                            text=f"Checking prices for {sail_label} ({index + 1}/{len(targets)})…",
                        )
                        scraped = scrape_cruise_url(sail_url)
                        if not scraped.get("ship_name") or scraped["ship_name"] == "Unknown Ship":
                            if use_all_dates and discovery.get("ship_name"):
                                scraped["ship_name"] = discovery["ship_name"]
                        if not scraped.get("duration") and use_all_dates and discovery.get("duration"):
                            scraped["duration"] = discovery["duration"]
                        if not scraped.get("departure_port") and use_all_dates:
                            scraped["departure_port"] = discovery.get("departure_port")
                        if not scraped.get("itinerary") and use_all_dates:
                            scraped["itinerary"] = discovery.get("itinerary")

                        with get_connection() as conn:
                            if cruise_url_is_tracked(conn, scraped["url"]) or cruise_url_is_tracked(
                                conn, sail_url
                            ):
                                skipped += 1
                                continue
                            cruise_id = add_cruise(conn, scraped)
                            insert_pricing_snapshot(conn, cruise_id, scraped)
                            added += 1
                            added_dates.append(str(scraped.get("sailing_date") or sail_label))
                            total_prices += sum(
                                1 for col in PRICE_COLUMNS if scraped.get(col)
                            )

                    progress.progress(1.0, text="Done.")
                    progress.empty()

                    if added:
                        st.session_state["last_error"] = None
                        if use_all_dates and len(targets) > 1:
                            date_summary = ", ".join(added_dates[:4])
                            if len(added_dates) > 4:
                                date_summary += f", +{len(added_dates) - 4} more"
                            st.session_state["last_add_success"] = (
                                f"Now tracking **{ship_label}** on **{added} sail dates** "
                                f"({date_summary})."
                            )
                        else:
                            st.session_state["last_add_success"] = (
                                f"Now tracking **{scraped['ship_name']}** "
                                f"(sailing {scraped['sailing_date']})."
                            )
                        if skipped:
                            st.session_state["last_add_success"] += (
                                f" Skipped {skipped} date(s) already being tracked."
                            )
                    elif skipped:
                        msg = "All sail dates for that itinerary are already being tracked."
                        st.session_state["last_error"] = msg
                        st.warning(msg)
                    else:
                        msg = "No cruises were added."
                        st.session_state["last_error"] = msg
                        st.warning(msg)

                if st.session_state.get("last_add_success"):
                    st.rerun()
            except PlaywrightSetupError as exc:
                st.session_state["last_error"] = str(exc)
                st.error(str(exc))
            except Exception as exc:
                msg = f"Could not add cruise: {exc}"
                st.session_state["last_error"] = msg
                st.error(msg)

    st.divider()
    st.header("Remove tracked cruise")
    with get_connection() as conn:
        tracked = get_all_cruises(conn)

    if tracked:
        delete_options = {
            f"{c['ship_name']} — {c['sailing_date']} (ID {c['id']})": c["id"]
            for c in tracked
        }
        delete_label = st.selectbox(
            "Select cruise to delete",
            list(delete_options.keys()),
            key="delete_cruise_select",
        )
        confirm_delete = st.checkbox(
            "Yes, remove this cruise and all of its price history",
            key="confirm_delete_cruise",
        )
        if st.button(
            "Delete cruise",
            type="secondary",
            use_container_width=True,
            disabled=not confirm_delete,
        ):
            cruise_id = delete_options[delete_label]
            with get_connection() as conn:
                removed = delete_cruise(conn, cruise_id)
            if removed:
                st.session_state.pop("last_scrape_summary", None)
                st.session_state["last_delete_success"] = (
                    f"Removed **{delete_label.split(' (ID')[0]}** from tracking."
                )
                st.rerun()
            else:
                st.session_state["last_error"] = "Could not delete that cruise."
    else:
        st.caption("No cruises to remove yet.")

    st.divider()
    st.header("Manual Voom price")
    if tracked:
        voom_options = {
            f"{c['ship_name']} — {c['sailing_date']} (ID {c['id']})": c["id"]
            for c in tracked
        }
        voom_label = st.selectbox(
            "Cruise for Voom entry",
            list(voom_options.keys()),
            key="manual_voom_select",
        )
        manual_voom = st.number_input(
            "Surf + Stream $/device/day",
            min_value=0.0,
            step=0.01,
            format="%.2f",
            help="From Cruise Planner → Internet. Example: 19.99",
        )
        if st.button("Save Voom price", use_container_width=True):
            if manual_voom <= 0:
                st.warning("Enter a price greater than zero.")
            else:
                cruise_id = voom_options[voom_label]
                with get_connection() as conn:
                    set_manual_voom_price(conn, cruise_id, float(manual_voom))
                st.session_state["last_voom_success"] = (
                    f"Saved Voom **${manual_voom:,.2f}/device/day** for "
                    f"{voom_label.split(' (ID')[0]}."
                )
                st.rerun()
    else:
        st.caption("Add a cruise first, then enter Voom from Cruise Planner if needed.")

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

if voom_success := st.session_state.pop("last_voom_success", None):
    st.success(voom_success)

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
