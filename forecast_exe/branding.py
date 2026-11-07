"""Shared icon-only contact links; never send messages automatically."""

import streamlit as st

from forecast_exe import __version__


def contact_dock():
    linkedin = '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="3" width="18" height="18" rx="3" fill="none" stroke="currentColor" stroke-width="1.6"/><circle cx="7.5" cy="8" r="1.2" fill="currentColor"/><path d="M7.5 11v6M11.5 17v-6m0 2c0-2.5 5-3 5 0v4" fill="none" stroke="currentColor" stroke-width="1.9"/></svg>'
    github = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8.8 20.5c-4 1-4-2-5.7-2.5m11.5 4v-3.3c0-1-.3-1.8-.8-2.2 2.8-.3 5.7-1.4 5.7-6.3 0-1.4-.5-2.5-1.3-3.4.1-.4.6-1.7-.1-3.4 0 0-1-.3-3.5 1.3a12 12 0 0 0-6.4 0C5.8 3.1 4.8 3.4 4.8 3.4c-.7 1.7-.3 3-.1 3.4-.8.9-1.3 2-1.3 3.4 0 4.9 2.9 6 5.7 6.3-.4.4-.8 1.1-.8 2.2V22" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>'
    email = '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="5" width="18" height="14" rx="3" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="m4 7 8 6 8-6" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>'
    st.markdown(
        '<nav class="contact-dock" aria-label="Contact">'
        '<a href="https://www.linkedin.com/in/cyristalj" target="_blank" rel="noopener noreferrer" aria-label="LinkedIn profile" title="LinkedIn">'
        + linkedin
        + "</a>"
        '<a href="https://github.com/cyristal-gems" target="_blank" rel="noopener noreferrer" aria-label="GitHub profile" title="GitHub">'
        + github
        + "</a>"
        '<a href="mailto:cyrisjoseph@outlook.com" aria-label="Email Cyristal Joseph" title="Email">'
        + email
        + "</a></nav>",
        unsafe_allow_html=True,
    )
    st.caption(f"Forecast.exe · v{__version__}")

