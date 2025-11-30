from __future__ import annotations

"""Utilities for opening Stim circuits in the Shatter web visualiser.

Primary entry point:
    prompt_open_shatter(stim_path: str) -> None

This reads the given .stim file, URL-encodes its content, and prompts the user
to open the Shatter web app with the circuit embedded in the URL fragment.
"""

from pathlib import Path
from urllib.parse import quote
import webbrowser


_SHATTER_BASE = "https://stasiu51.github.io/Shatter/#circuit="


def build_shatter_url_from_text(stim_text: str) -> str:
    """Return a Shatter URL with the given Stim text embedded in the hash.

    The Stim text is URL-encoded verbatim. For large circuits, note that browser
    URL length limits may apply.
    """
    return _SHATTER_BASE + quote(stim_text, safe="")


def prompt_open_shatter(stim_path: str) -> None:
    """Prompt the user and open Shatter in the system browser for the given .stim.

    Prints a message:
      "Press enter to open Shatter to visualise the circuit; or ctrl-c to cancel"
    If the user presses Enter, opens the default web browser at the Shatter URL
    with the circuit embedded. Ctrl-C cancels without action.
    """
    p = Path(stim_path)
    try:
        text = p.read_text()
    except Exception as e:
        print(f"[warn] Could not read stim file '{stim_path}': {e}")
        return
    url = build_shatter_url_from_text(text)
    try:
        input("Press enter to open Shatter to visualise the circuit; or ctrl-c to cancel: ")
    except KeyboardInterrupt:
        print("\n[cancelled] Not opening Shatter.")
        return
    try:
        webbrowser.open(url)
        print("[ok] Opened Shatter in your default browser.")
    except Exception as e:
        print(f"[warn] Failed to open browser: {e}")

