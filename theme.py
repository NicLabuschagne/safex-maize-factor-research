"""
theme.py — shared One Dark Pro theme for matplotlib and seaborn.

Usage
-----
    from theme import set_theme
    set_theme()
"""

from __future__ import annotations

from typing import Any, Dict, List

import matplotlib.pyplot as plt
import seaborn as sns

# One Dark Pro Core System Colors
BG = "#282c34"         # Classic One Dark charcoal canvas
FG = "#abb2bf"         # Soft light gray for labels and titles
EDGE = "#3e4451"       # Subtle panel border mapping
GRID = "#3e4451"       # Distinct grid structure tone

# Balanced Syntax Accent Spectrum
PALETTE: List[str] = [
    "#61afef",  # Bright Sky Blue
    "#98c379",  # Soft Emerald Green
    "#e5c07b",  # Warm Autumn Yellow
    "#d19a66",  # Light Terracotta Orange
    "#c678dd",  # Deep Vibrant Purple
    "#e06c75",  # Soft Coral Red
]

RC_PARAMS: Dict[str, Any] = {
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "axes.edgecolor": EDGE,
    "axes.labelcolor": FG,
    "text.color": FG,
    "xtick.color": FG,
    "ytick.color": FG,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",  # Crisp, solid grid line structure matching standard code themes
    "font.size": 9,
}


def set_theme() -> Dict[str, Any]:
    """
    Apply the shared One Dark Pro theme to matplotlib and seaborn.

    Returns
    -------
    dict with keys: rc_params (dict), palette (list)
    """
    sns.set_theme(style="darkgrid", rc=RC_PARAMS)
    plt.rcParams.update(RC_PARAMS)
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=PALETTE)
    sns.set_palette(PALETTE)

    return {"rc_params": RC_PARAMS, "palette": PALETTE}
