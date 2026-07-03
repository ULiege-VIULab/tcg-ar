"""
Branded Qt stylesheet for the PTCG-AR GUI.

A single QSS string applied once in ``inference.main`` via ``app.setStyleSheet``,
giving the whole app a clean, Pokemon-broadcast-inspired look: a deep blue
palette, rounded cards/buttons, gradient headers and restyled scrollbars,
replacing the scattered ``Times`` fonts. Widgets opt into specific styling by
object name where needed; everything else inherits the base theme.
"""

# Palette (kept in sync with the broadcast overlay in caster_module.py).
BRAND = "#266ec8"
BRAND_DARK = "#18468c"
BG = "#0f1724"
PANEL = "#16203a"
PANEL_LIGHT = "#1f2c4a"
INK = "#e7eef8"
SUBTLE = "#9fb2cc"
BORDER = "#2a3b5c"

STYLESHEET = f"""
* {{
    font-family: "Segoe UI", "Arial", sans-serif;
    font-size: 14px;
    color: {INK};
}}

QMainWindow, QWidget {{
    background-color: {BG};
}}

QMenuBar {{
    background-color: {BRAND_DARK};
    color: {INK};
    padding: 2px;
}}
QMenuBar::item {{
    background: transparent;
    padding: 6px 12px;
    border-radius: 6px;
}}
QMenuBar::item:selected {{
    background-color: {BRAND};
}}
QMenu {{
    background-color: {PANEL};
    border: 1px solid {BORDER};
    padding: 4px;
}}
QMenu::item {{
    padding: 6px 24px 6px 12px;
    border-radius: 4px;
}}
QMenu::item:selected {{
    background-color: {BRAND};
}}
QMenu::separator {{
    height: 1px;
    background: {BORDER};
    margin: 4px 8px;
}}

QPushButton {{
    background-color: {BRAND};
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: bold;
}}
QPushButton:hover {{
    background-color: #357ad6;
}}
QPushButton:pressed {{
    background-color: {BRAND_DARK};
}}
QPushButton:disabled {{
    background-color: #33415c;
    color: {SUBTLE};
}}

QGroupBox {{
    background-color: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 10px;
    margin-top: 14px;
    padding: 10px;
    font-weight: bold;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 2px 10px;
    color: white;
    background-color: {BRAND};
    border-radius: 6px;
}}

QLineEdit, QSpinBox, QComboBox {{
    background-color: {PANEL_LIGHT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 5px 8px;
    selection-background-color: {BRAND};
}}
QLineEdit:focus, QSpinBox:focus, QComboBox:focus {{
    border: 1px solid {BRAND};
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox QAbstractItemView {{
    background-color: {PANEL};
    border: 1px solid {BORDER};
    selection-background-color: {BRAND};
}}

QCheckBox {{
    spacing: 8px;
}}
QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 1px solid {BORDER};
    background: {PANEL_LIGHT};
}}
QCheckBox::indicator:checked {{
    background: {BRAND};
    border: 1px solid {BRAND};
}}

QLabel {{
    background: transparent;
}}

QScrollArea {{
    border: none;
}}
QScrollBar:vertical {{
    background: {PANEL};
    width: 12px;
    margin: 0;
    border-radius: 6px;
}}
QScrollBar::handle:vertical {{
    background: {BRAND};
    min-height: 30px;
    border-radius: 6px;
}}
QScrollBar::handle:vertical:hover {{
    background: #357ad6;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: {PANEL};
    height: 12px;
    margin: 0;
    border-radius: 6px;
}}
QScrollBar::handle:horizontal {{
    background: {BRAND};
    min-width: 30px;
    border-radius: 6px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

QProgressBar {{
    background-color: {PANEL_LIGHT};
    border: 1px solid {BORDER};
    border-radius: 8px;
    text-align: center;
    color: {INK};
}}
QProgressBar::chunk {{
    background-color: {BRAND};
    border-radius: 7px;
}}

QToolTip {{
    background-color: {PANEL};
    color: {INK};
    border: 1px solid {BRAND};
    padding: 4px;
}}

QStatusBar {{
    background-color: {BRAND_DARK};
    color: {INK};
}}
"""
