"""CSS for the benchmark app: black buttons as in the Uitnodigingsregel app."""

APP_CSS = """
<style>
[data-testid="stToolbar"]    { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
.block-container             { padding-top: 2rem; max-width: 900px; }

button[kind="primary"],
[data-testid="stBaseButton-primary"],
[data-testid="stDownloadButton"] button {
    background-color: #1a1a1a !important;
    color: white !important;
    border-radius: 50px !important;
    font-weight: 700 !important;
    border: none !important;
}
[data-testid="stBaseButton-primary"]:disabled { background-color: #aaa !important; }
</style>
"""
