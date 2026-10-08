r"""House style + deterministic audit for publication-grade figures.

Usage:
    from figstyle import apply_house_style, save, panel_label, audit
    plt = apply_house_style()
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Nature column widths (mm). Science/Cell are close enough to reuse these.
NATURE_SINGLE_MM = 89.0
NATURE_DOUBLE_MM = 183.0
MAX_HEIGHT_MM = 247.0


def mm(value: float) -> float:
    """Millimetres to inches, for figsize=(mm(89), mm(60))."""
    return value / 25.4


def apply_house_style(styles=("science", "nature"), fontsize: float = 7.0,
                      fonttype: int = 42, use_tex: bool = False):
    """Apply the house standard and return pyplot.

    import scienceplots FIRST -- it is what registers the "science"/"nature"
    style names. Without it plt.style.use([...]) raises OSError.
    """
    import scienceplots  # noqa: F401  (registers the styles)
    plt.style.use(list(styles))
    plt.rcParams.update({
        # Type 3 (matplotlib default) is bitmap-ish and rejected by many
        # publishers. 42 = embed TrueType.
        "pdf.fonttype": fonttype,
        "ps.fonttype": fonttype,
        # Keep SVG text as text so the file stays editable in Illustrator/Inkscape.
        "svg.fonttype": "none",
        "font.size": fontsize,
        "axes.titlesize": fontsize,
        "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize - 1,
        "ytick.labelsize": fontsize - 1,
        "legend.fontsize": fontsize - 1,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "axes.grid": False,
        "legend.frameon": False,
        "text.usetex": use_tex,
    })
    return plt


def panel_label(ax, text: str, loc: str = "left", bold: bool = True, **kwargs):
    """Label a panel via a title slot.

    Traps worth remembering:
      * ax.texts does NOT contain this text.
      * ax.get_title() returns only the CENTRED title.
      * ax.get_title(loc="left") is the only way to read it back.
    A QA pass that checks only get_title()/texts silently misses every
    left-aligned panel label -- the exact failure this house style prevents.

    Convention: Nature-class journals label panels with BOLD LOWER-CASE letters
    and no parentheses -- "a", not "(a)". Pass bold=False and text="(a)" for
    journals that want the parenthesised form.
    """
    weight = "bold" if bold else plt.rcParams["axes.titleweight"]
    ax.set_title(text, loc=loc, fontsize=plt.rcParams["font.size"],
                 fontweight=weight, **kwargs)


def save(fig, stem: str, dpi: int = 600) -> dict:
    """Write the three-file deliverable: vector PDF, high-dpi PNG, editable SVG."""
    written = {}
    for ext in ("pdf", "png", "svg"):
        path = f"{stem}.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        written[ext] = path
    return written


def audit(fig, max_width_mm: float = NATURE_DOUBLE_MM, verbose: bool = True) -> dict:
    """Deterministic pre-flight checks. No vision model involved.

    The authoritative number is the TIGHT size: matplotlib writes with
    bbox_inches="tight", so the file that lands on the editor's desk is the
    tight bbox, not the nominal figsize. A figure configured at 89 mm can
    silently ship at 104 mm because one long label pushed it out.

    Also reports the left/right panel labels that a naive title scan misses.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    titles = []
    for i, ax in enumerate(fig.axes):
        for loc in ("left", "center", "right"):
            text = ax.get_title(loc=loc)
            if text:
                titles.append({"axis": i, "loc": loc, "text": text})

    sizes = []
    for ax in fig.axes:
        candidates = list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label]
        candidates += list(ax.get_xticklabels()) + list(ax.get_yticklabels())
        if ax.get_legend() is not None:
            candidates += list(ax.get_legend().get_texts())
        for t in candidates:
            if not t.get_text().strip():
                continue
            try:
                bb = t.get_window_extent(renderer=renderer)
                sizes.append(float(t.get_fontsize()))
            except Exception:
                continue
            # Text positions are only authoritative after get_tightbbox(), so no
            # per-text overflow heuristic is reported here: it produced false
            # positives on ordinary tick labels. The tight-size comparison below
            # is the measurement that actually predicts the shipped dimensions.

    tight = fig.get_tightbbox(renderer)
    tight_w_mm, tight_h_mm = tight.width * 25.4, tight.height * 25.4
    width_in, height_in = fig.get_size_inches()

    report = {
        "nominal_mm": [round(float(width_in) * 25.4, 1), round(float(height_in) * 25.4, 1)],
        "tight_mm": [round(tight_w_mm, 1), round(tight_h_mm, 1)],
        "tight_growth_mm": round(tight_w_mm - float(width_in) * 25.4, 1),
        "tight_within_max": tight_w_mm <= max_width_mm + 0.5,
        "max_width_mm": max_width_mm,
        "panel_titles": titles,
        "left_titles_found": sum(1 for t in titles if t["loc"] == "left"),
        "min_font_pt": min(sizes) if sizes else None,
        "fonts_under_5pt": sorted({s for s in sizes if s < 5}),
        "pdf_fonttype": int(plt.rcParams["pdf.fonttype"]),
    }
    report["ok"] = (report["tight_within_max"]
                    and not report["fonts_under_5pt"]
                    and report["pdf_fonttype"] == 42)

    if verbose:
        print(f"[audit] nominal {report['nominal_mm'][0]}x{report['nominal_mm'][1]} mm "
              f"-> tight {report['tight_mm'][0]}x{report['tight_mm'][1]} mm "
              f"(growth {report['tight_growth_mm']:+} mm)")
        print(f"[audit] tight width <= {max_width_mm} mm: {report['tight_within_max']}")
        print(f"[audit] panel titles: {len(titles)} "
              f"(left={report['left_titles_found']}, centre/right="
              f"{len(titles) - report['left_titles_found']})")
        print(f"[audit] min font {report['min_font_pt']} pt | "
              f"under 5pt: {report['fonts_under_5pt'] or 'none'}")
        print(f"[audit] pdf.fonttype={report['pdf_fonttype']} (42 = TrueType embedded)")
        print(f"[audit] OK = {report['ok']}")
    return report
