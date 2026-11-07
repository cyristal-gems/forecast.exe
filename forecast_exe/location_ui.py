"""Permission-based browser location and per-browser city preferences."""

from pathlib import Path
import uuid

import streamlit as st

from forecast_exe.data import NYC, geocode
from forecast_exe.locations import nearby_city, validate_location
from forecast_exe.comparisons import add_to_comparison

HTML = """<div class="location-tools">
<button data-use type="button" aria-label="Use my location" title="Use my location"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="19" height="19" aria-hidden="true"><path d="M21 3L3 10.5L10.5 13.5L13.5 21Z" fill="currentColor" stroke="currentColor" stroke-linejoin="round"/></svg></button>
<div class="aux"><input data-remember type="checkbox" checked><button data-forget type="button">Forget saved city</button></div>
<div data-status class="sr-only" role="status" aria-live="polite"></div></div>"""

CSS = """.location-tools {font:13px system-ui;color:#afc1d7;}
button {display:flex;align-items:center;justify-content:center;height:42px;width:100%;padding:0;border:1px solid #c8e6fc2e;border-radius:999px;
background:#1b30455c;color:#e8f5ff;font:26px system-ui;cursor:pointer;}
button:hover {background:#35576d77;border-color:#cee8f888;} button:disabled {opacity:.5;cursor:wait;}
.aux {display:none;}.sr-only {position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0,0,0,0);}
button:focus-visible {outline:2px solid #8df3d5;outline-offset:3px;}"""

BRIDGE = st.components.v2.component(
    "forecast_city_preferences",
    html=HTML,
    css=CSS,
    js=Path(__file__).with_name("location_bridge.js").read_text(),
)


def select_location(state, city, remember=None, restored=False):
    """Apply a validated city while invalidating city-specific UI snapshots."""
    city = validate_location(city)
    state["active_location"] = city
    state["demo_mode"] = False
    state["forecast_source"] = "External forecast"
    state.pop("dataset", None)
    state.pop("external", None)
    state.pop("location_error", None)
    if remember is not None:
        state["remember_city"] = bool(remember)
    if not restored:
        state["preference_write"] = dict(
            city=city, remember=state.get("remember_city", True), revision=uuid.uuid4().hex
        )


def process_event(state, event, resolver=nearby_city):
    """Interpret component events; no permission request or lookup occurs on page load."""
    if not isinstance(event, dict) or not event.get("id") or state.get("last_location_event") == event["id"]:
        return False
    state["last_location_event"] = event["id"]
    state["city_preferences_ready"] = True
    kind = event.get("kind")
    try:
        if kind == "ready":
            if event.get("storage_error"):
                state["storage_notice"] = (
                    "Browser storage is unavailable. City selection works for this visit, but cannot be remembered."
                )
        elif kind == "storage-error":
            state["storage_notice"] = (
                "Browser storage is unavailable. City selection works for this visit, but cannot be remembered."
            )
        elif kind == "detected":
            city = resolver(event.get("latitude"), event.get("longitude"))
            select_location(state, city, remember=event.get("remember", True))
            state["location_notice"] = (
                "Using nearby " + city["name"] + ". You can change it with city search."
            )
        elif kind == "restore":
            select_location(state, event.get("city"), remember=True, restored=True)
            state["location_notice"] = "Restored your preferred city in this browser."
        elif kind == "remember":
            state["remember_city"] = bool(event.get("remember"))
            state["preference_write"] = dict(
                city=state.get("active_location"), remember=state["remember_city"], revision=uuid.uuid4().hex
            )
        elif kind == "forget":
            if event.get("storage_error"):
                state["storage_notice"] = (
                    "The browser could not clear saved preferences. Clear site storage in your browser settings."
                )
            state.pop("active_location", None)
            state.pop("dataset", None)
            state.pop("location_error", None)
            state["preference_write"] = dict(city=None, remember=False, revision=uuid.uuid4().hex)
            state["location_notice"] = "Saved city forgotten. Select a city or use location again."
        elif kind == "error":
            state["location_error"] = str(event.get("message", "Location unavailable. Use city search."))[
                :250
            ]
        else:
            return False
    except (ValueError, TypeError, KeyError, RuntimeError, OSError) as exc:
        state["location_error"] = f"Location could not be resolved: {exc}. Please use city search."
    return True


def choose_selected():
    select_location(st.session_state, st.session_state["location_choice"])
    st.session_state["location_notice"] = "Using " + st.session_state["active_location"]["name"]


def location_button():
    """Header arrow: restore city preferences without requesting device location."""
    result = BRIDGE(
        key="city_bridge",
        data=dict(
            write=st.session_state.get("preference_write"),
            remember=st.session_state.get("remember_city", True),
        ),
        on_event_change=lambda: None,
    )
    if process_event(st.session_state, result.get("event")):
        st.rerun()
    return validate_location(st.session_state.get("active_location", NYC))


def remember_changed():
    process_event(
        st.session_state,
        dict(id=uuid.uuid4().hex, kind="remember", remember=st.session_state.get("remember_city", True)),
    )


def forget_saved_city():
    process_event(st.session_state, dict(id=uuid.uuid4().hex, kind="forget"))
    st.session_state["forecast_source"] = "External forecast"


def choose_comparison():
    add_to_comparison(st.session_state, st.session_state["location_choice"])


def search_controls():
    """Manual fallback and remembered-city controls, all inside the header gear."""
    st.markdown("#### City search")
    query = st.text_input("Find a city", "New York")
    if st.button("Search cities"):
        try:
            st.session_state["locations"] = geocode(query)
            st.session_state.pop("location_choice", None)
        except Exception as exc:
            st.error(f"City search failed: {exc}. Please retry.")
    locations = st.session_state.get("locations", [NYC])
    if locations:
        st.selectbox("Location", locations, format_func=lambda city: city["name"], key="location_choice")
        st.button("Use selected city", on_click=choose_selected)
        st.button("Add to comparison", on_click=choose_comparison)
    else:
        st.info("No matches. Try a city and country.")
    if "remember_city" not in st.session_state:
        st.session_state["remember_city"] = True
    st.checkbox(
        "Remember this city in this browser", key="remember_city", on_change=remember_changed
    )
    st.button("Forget saved city", on_click=forget_saved_city)
    st.caption(
        "The triangular location arrow requests browser location only when clicked. Lookup uses a nearby city center; "
        "manual search works if permission is declined. Only the selected city is saved in this browser."
    )
    if "active_location" not in st.session_state:
        st.caption("Default: New York City. Choose a city or use the location arrow to change it.")
