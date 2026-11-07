"""Decorative weather-responsive themes driven only by validated external estimates."""

from datetime import UTC, datetime
import streamlit as st


def location_identity(location):
    return f"{location['latitude']:.4f},{location['longitude']:.4f}:{location['timezone']}"


def weather_theme(code, is_day):
    """Classify provider WMO codes; never infer current conditions from ML outputs."""
    code = int(code)
    if code in (0, 1):
        return "sunny" if is_day else "clear-night"
    if code in (2, 3):
        return "cloudy" if is_day else "cloudy-night"
    if code in (45, 48):
        return "fog"
    if code in (51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82):
        return "rain"
    if code in (71, 73, 75, 77, 85, 86):
        return "snow"
    if code in (95, 96, 99):
        return "storm"
    return "neutral"


def theme_css(theme):
    palette = {
        "sunny": "rgba(240,189,104,.32)",
        "clear-night": "rgba(80,116,180,.16)",
        "cloudy": "rgba(154,188,207,.17)",
        "cloudy-night": "rgba(59,83,128,.14)",
        "fog": "rgba(184,201,209,.27)",
        "rain": "rgba(67,109,145,.23)",
        "snow": "rgba(183,215,233,.28)",
        "storm": "rgba(104,83,170,.23)",
        "neutral": "rgba(96,144,170,.12)",
    }
    glow = palette.get(theme, palette["neutral"])
    overlay, animation = "none", "none"
    if theme == "rain":
        overlay = (
            "repeating-linear-gradient(112deg,transparent 0 28px,#b8d7eb16 29px 30px,transparent 31px 65px)"
        )
        animation = "weather-rain 1.8s linear infinite"
    elif theme == "snow":
        overlay = "radial-gradient(circle at 15px 25px,#edf7ff55 1px,transparent 2px),radial-gradient(circle at 70px 85px,#edf7ff33 1.5px,transparent 2.5px)"
        animation = "weather-snow 18s linear infinite"
    elif theme == "clear-night":
        overlay = "radial-gradient(circle at 12% 18%,#eaf4ff66 1px,transparent 2px),radial-gradient(circle at 45% 8%,#eaf4ff44 1px,transparent 2px),radial-gradient(circle at 76% 26%,#eaf4ff44 1px,transparent 2px)"
    elif theme == "fog":
        overlay = "linear-gradient(0deg,#c2d1d51c,transparent 75%)"
    elif theme == "storm":
        overlay = "radial-gradient(ellipse at 80% 12%,#8475bd22,transparent 60%)"
        animation = "weather-storm 14s ease-in-out infinite"
    size = "100% 100%" if theme in ["clear-night", "fog", "storm"] else "140px 180px"
    return f"""[data-testid="stAppViewContainer"] {{
background-image:radial-gradient(ellipse at 85% 8%,{glow},transparent 60%),
linear-gradient(90deg,#050d19bb,#050d1922),url("app/static/atmosphere.png");
background-position:center top;background-size:cover;background-attachment:fixed;}}
[data-testid="stAppViewContainer"]::before {{content:"";position:fixed;inset:0;pointer-events:none;
z-index:0;background-image:{overlay};background-size:{size};animation:{animation};}}
[data-testid="stMain"] {{position:relative;z-index:1;}}
@keyframes weather-rain {{from{{background-position:0 -180px;}}to{{background-position:-55px 180px;}}}}
@keyframes weather-snow {{from{{background-position:0 -180px;}}to{{background-position:40px 180px;}}}}
@keyframes weather-storm {{0%,100%{{opacity:.45;}}50%{{opacity:1;}}}}
@media(prefers-reduced-motion:reduce){{[data-testid="stAppViewContainer"]::before{{animation:none!important;}}}}
"""


def apply_background(location, demo=False, snapshot=None):
    """Retain a city's recent external theme across tabs; fall back when unknown/stale."""
    if snapshot is not None and not demo:
        st.session_state["weather_theme"] = dict(
            location=location_identity(location),
            valid_at_utc=snapshot["valid_at_utc"],
            code=snapshot["current"]["weather_code"],
            is_day=bool(snapshot["current"]["is_day"]),
        )
    saved = st.session_state.get("weather_theme")
    theme = "neutral"
    if not demo and saved and saved["location"] == location_identity(location):
        age = (datetime.now(UTC) - datetime.fromisoformat(saved["valid_at_utc"])).total_seconds()
        if -3600 <= age <= 3 * 3600:
            theme = weather_theme(saved["code"], saved["is_day"])
    st.markdown("<style>" + theme_css(theme) + "</style>", unsafe_allow_html=True)
    return theme
