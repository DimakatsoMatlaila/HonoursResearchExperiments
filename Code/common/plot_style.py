import matplotlib

matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap

PAPER = "#EFF2F6"
PANEL = "#F7F4EC"
INK = "#2B3441"
PALETTE = ["#2A7F9E", "#E07A3F", "#4C9F70", "#B5484B", "#7B6BB5", "#C9A227", "#3D5A80", "#8C6D5A"]

CM_SEQ = LinearSegmentedColormap.from_list("seq", ["#F1E9D2", "#7FB7BE", "#1F5F7A"])
CM_GREEN = LinearSegmentedColormap.from_list("sel", ["#F1E9D2", "#8CC084", "#2E6B4F"])

matplotlib.rcParams.update({
    "figure.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "axes.facecolor": PANEL,
    "axes.edgecolor": "#9AA5B1",
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.grid": True,
    "grid.color": "#D5DAE0",
    "grid.linewidth": 0.7,
    "axes.axisbelow": True,
    "legend.facecolor": PANEL,
    "legend.edgecolor": "#C5CCD4",
    "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
})
