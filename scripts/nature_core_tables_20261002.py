"""Render all five manuscript tables as editable, publication-scale vector plates.

The manuscript is the sole data source. Display wrapping never changes cell text.
No statistical estimates, labels, or comparisons are calculated in this renderer.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "plot_inputs" / "core_tables.json"
OUT = ROOT / "generated" / "nature_visual_redesign_20261002" / "tables"
WIDTH_MM = 180.0
MARGIN_MM = 3.0
FONT_SIZE = 7.0
LINE_HEIGHT_PT = 9.6
CELL_PAD_X_MM = 1.5
CELL_PAD_Y_MM = 1.75
TITLE_SIZE = 8.0


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_source() -> dict:
    return {int(key):value for key,value in json.loads(SOURCE.read_text(encoding='utf-8')).items()}


def wrap_measured(text: str, max_width_mm: float, renderer, bold=False) -> str:
    """Greedy line wrap using the actual selected Arial renderer, not characters."""
    props = FontProperties(family="Arial", size=FONT_SIZE, weight="bold" if bold else "normal")
    limit_px = max_width_mm / 25.4 * 300
    output = []
    for paragraph in str(text).split("\n"):
        words = paragraph.split(" ")
        line = ""
        for word in words:
            candidate = word if not line else f"{line} {word}"
            width, _, _ = renderer.get_text_width_height_descent(candidate, props, ismath=False)
            if width <= limit_px or not line:
                line = candidate
            else:
                output.append(line)
                line = word
        output.append(line)
    wrapped = "\n".join(output)
    # Preserve every source token, including punctuation and numerical tokens.
    assert re.sub(r"\s+", " ", wrapped).strip() == re.sub(r"\s+", " ", text).strip()
    for line in output:
        width, _, _ = renderer.get_text_width_height_descent(line, props, ismath=False)
        if width > limit_px + .5:
            raise ValueError(f"Unbreakable word wider than cell: {line!r}")
    return wrapped


def line_height_mm(text: str) -> float:
    return len(text.splitlines()) * LINE_HEIGHT_PT / 72 * 25.4


def prepare_block(block: dict, widths_mm: list[float], renderer, headers=None) -> dict:
    assert abs(sum(widths_mm) - (WIDTH_MM - 2 * MARGIN_MM)) < .01
    headers = headers or block["headers"]
    assert len(headers) == len(widths_mm)
    wrapped_headers = [wrap_measured(value, width - 2 * CELL_PAD_X_MM, renderer, True)
                       for value, width in zip(headers, widths_mm)]
    wrapped_rows = [[wrap_measured(value, width - 2 * CELL_PAD_X_MM, renderer)
                     for value, width in zip(row, widths_mm)] for row in block["rows"]]
    header_height = max(map(line_height_mm, wrapped_headers)) + 2 * CELL_PAD_Y_MM
    row_heights = [max(map(line_height_mm, row)) + 2 * CELL_PAD_Y_MM for row in wrapped_rows]
    return {"headers": wrapped_headers, "rows": wrapped_rows, "widths_mm": widths_mm,
            "header_height": header_height, "row_heights": row_heights,
            "height": header_height + sum(row_heights)}


def local_content_audit(fig, texts: list[dict], rules: list[dict]) -> dict:
    """Deterministic final-draw text, cell and rule collision check at 300 dpi."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    text_overlap = []
    rule_overlap = []
    overflow = []
    boxes = []
    for record in texts:
        artist = record["artist"]
        box = artist.get_window_extent(renderer)
        boxes.append((record, box))
        if not (bounds.contains(box.x0, box.y0) and bounds.contains(box.x1, box.y1)):
            overflow.append(record["id"])
        if "cell" in record:
            x0, x1, y0, y1 = record["cell"]
            # Cell coordinates are transformed from mm geometry below.
            if box.x0 < x0 - .5 or box.x1 > x1 + .5 or box.y0 < y0 - .5 or box.y1 > y1 + .5:
                overflow.append(record["id"] + ":cell")
    for index, (record, box) in enumerate(boxes):
        for other, other_box in boxes[index + 1:]:
            if box.overlaps(other_box):
                text_overlap.append([record["id"], other["id"]])
        for rule in rules:
            rule_box = rule["artist"].get_window_extent(renderer)
            # Matplotlib's horizontal-line rectangle has zero height, so test
            # stroke extent explicitly, with a 0.25 pt guard margin.
            pad = (rule["linewidth"] / 2 + .25) / 72 * fig.dpi
            if (box.x1 > rule_box.x0 and box.x0 < rule_box.x1 and
                    box.y0 < rule_box.y0 + pad and box.y1 > rule_box.y0 - pad):
                rule_overlap.append([record["id"], rule["id"]])
    assert not text_overlap, text_overlap
    assert not rule_overlap, rule_overlap
    assert not overflow, overflow
    return {"text_text_intersections": len(text_overlap),
            "text_horizontal_rule_intersections": len(rule_overlap),
            "text_or_cell_overflow": len(overflow), "text_artists_checked": len(boxes),
            "horizontal_rules_checked": len(rules), "checked_at_dpi": fig.dpi}


def draw_table(number: int, source: dict, blocks: list[dict], title: str,
               panels: list[str], numeric_cols: list[set[int]]) -> dict:
    from nature_visual_style_20261002 import apply_style, export_figure, figure

    apply_style()
    measure = figure(width_mm=WIDTH_MM, height_mm=100)
    measure.set_dpi(300)
    measure.canvas.draw()
    renderer = measure.canvas.get_renderer()
    layouts = [prepare_block(block, widths, renderer, headers)
               for block, (widths, headers) in zip(source["blocks"], blocks)]
    plt.close(measure)
    title_h = 8.5
    panel_h = 6.0 if len(layouts) > 1 else 0
    gap_h = 8.0 if len(layouts) > 1 else 0
    height = MARGIN_MM * 2 + title_h + sum(item["height"] for item in layouts) + panel_h * len(layouts) + gap_h
    fig = figure(width_mm=WIDTH_MM, height_mm=height)
    fig.set_dpi(300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH_MM)
    ax.set_ylim(height, 0)
    ax.set_axis_off()
    records, rules = [], []

    def put(label, x, y, text, align="left", bold=False, size=FONT_SIZE, cell=None):
        artist = ax.text(x, y, text, fontsize=size, family="Arial", weight="bold" if bold else "normal",
                         ha=align, va="top", color="#202020", linespacing=LINE_HEIGHT_PT / size,
                         transform=ax.transData)
        item = {"id": label, "artist": artist}
        if cell:
            low = ax.transData.transform([cell[0], cell[3]])
            high = ax.transData.transform([cell[1], cell[2]])
            item["cell"] = (low[0], high[0], low[1], high[1])
        records.append(item)

    def rule(label, y, linewidth):
        artist, = ax.plot([MARGIN_MM, WIDTH_MM - MARGIN_MM], [y, y], color="#202020",
                          linewidth=linewidth, solid_capstyle="butt")
        rules.append({"id": label, "artist": artist, "linewidth": linewidth})

    put("title", MARGIN_MM, MARGIN_MM, f"Table {number} | {title}", bold=True, size=TITLE_SIZE)
    y = MARGIN_MM + title_h
    for panel, layout in enumerate(layouts):
        if len(layouts) > 1:
            put(f"panel_{panel}", MARGIN_MM, y, panels[panel], bold=True, size=TITLE_SIZE)
            y += panel_h
        rule(f"panel_{panel}:top", y, .65)
        heights = [layout["header_height"]] + layout["row_heights"]
        rows = [layout["headers"]] + layout["rows"]
        for row_index, (row, row_h) in enumerate(zip(rows, heights)):
            x = MARGIN_MM
            for col, (value, width) in enumerate(zip(row, layout["widths_mm"])):
                numeric = col in numeric_cols[panel]
                align = "right" if numeric else "left"
                tx = x + width - CELL_PAD_X_MM if numeric else x + CELL_PAD_X_MM
                put(f"panel_{panel}:r{row_index}:c{col}", tx, y + CELL_PAD_Y_MM,
                    value, align=align, bold=row_index == 0,
                    cell=(x, x + width, y, y + row_h))
                x += width
            y += row_h
            if row_index == 0:
                rule(f"panel_{panel}:header", y, .45)
        rule(f"panel_{panel}:bottom", y, .65)
        if panel < len(layouts) - 1:
            y += gap_h
    local_qa = local_content_audit(fig, records, rules)
    result = export_figure(fig, OUT, f"table{number}_nature", source_note=f"Released manuscript Table {number} cells",
                           values={"source_blocks": source["blocks"]}, check=True)
    result["table_geometry_qa"] = local_qa
    result["width_mm"] = WIDTH_MM
    result["height_mm"] = height
    result["font_pt"] = FONT_SIZE
    result["minimum_font_pt"] = FONT_SIZE
    plt.close(fig)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tables = parse_source()
    # Column labels may be condensed; exact original headers remain in CSV.
    descriptions = {
        1: {"title": "Electricity measurements and monitoring boundaries", "panels": [""],
            "layouts": [([29, 49, 96], ["Area", "Provider and native resolution", "Accounting boundary"])], "numeric": [set()]},
        2: {"title": "Original-region q90 review allocation", "panels": [""],
            "layouts": [([46, 20, 25, 26, 25, 32], ["Rule", "Native\nslots", "Event hours\nselected", "False-alert\nfraction", "Native early\nspells", "Expected early\ncoverage¹"])],
            "numeric": [{1, 2, 3, 4, 5}]},
        3: {"title": "q90 joint probability loss", "panels": [""],
            "layouts": [([22, 70, 41, 41], ["Model", "Specification", "Development\nBrier loss", "Reanalysis\nBrier loss"])], "numeric": [{2, 3}]},
        4: {"title": "Local-weather Danish transfer at q90", "panels": ["a  Joint probability loss", "b  Fixed-rule review allocation"],
            "layouts": [([24, 75, 75], ["Model", "DE–LU / FR / DK1\nBrier loss", "DE–LU / FR / DK2\nBrier loss"]),
                        ([13, 15, 22, 36, 36, 17, 35], ["New\narea", "Rule", "Native\nslots", "Event hits\n(DK-pivotal)", "Native early spells\n(fraction)", "Common\nB", "Matched early\ncoverage"])],
            "numeric": [{1, 2}, {2, 3, 4, 5, 6}]},
        5: {"title": "Sensitivity of the main interpretation", "panels": [""],
            "layouts": [([30, 42, 85, 17], ["Check", "Specification", "Result and scope", "Detail"])], "numeric": [set()]},
    }
    notes = {
        1: "Outcomes are hourly MW on a common UTC clock, 2019–2025. The primary event is simultaneous exceedance of source-specific seasonal q90 monitoring levels in at least two study areas; q95 is secondary. Seasonal levels do not harmonise physical accounting boundaries. Danish fitting uses hour-end + 15-day availability. Original quarterly fits use origin − 7 days (6.5 days before first issue), daily recalibration uses issue − 7 days, and new weekly maps/t selection uses a strict issue − 7-day cutoff. These delays define a historical replay; historical publication vintages were not recovered.",
        2: "The score domain comprises 725 complete 2024–2025 days for Germany–Luxembourg / France / Belgium. Native counts precede exact expected coverage after uniform, label-blind retention to 572 slots (¹). A native early spell is covered by a selected event target hour within the first six elapsed hours after realised onset. P0 requires two selections per complete day. P1 gates at probability 0.25 and permits zero selections; P2 gates at 0.19, applies risk-rise priority and a six-hour cooldown. Forecast-complete days lacking later labels are allocated but excluded from this score domain. A selected slot is a delivery-hour window, not an hour of staff work.",
        3: "Development comprises 715 complete 2022–2023 days; reanalysis comprises 725 complete 2024–2025 days. Lower Brier loss is better. All M0–M3 rows share physical thresholds, weather, samples and correlation matrices. F0 is the original marginal stream with daily beta recalibration; F1 adds the conditional map, without implying certified calibration. M2 and M3 share the selected ν = 3 t dependence. Independence and direct logistic rows are earlier fitted references.",
        4: "Panel a uses the same 17,400 scoring hours in each combination. Panel b reports native selected slots, detected event hours and early spells, followed by exact expected early coverage at the combination-specific common total B. Parenthesised event hits are Danish-pivotal subsets: Denmark exceeds and exactly one of Germany–Luxembourg / France exceeds. The other joint-event component comprises 924 hours with both reused areas exceeding, irrespective of Denmark. For DK1 the native early denominator is 71 spells and B = 432; for DK2 it is 75 and B = 492. Definitions and fixed review rules are the same as Table 2. Danish observations and local weather provide partial spatial transfer because German/French components were already inspected.",
        5: "Full contrasts are retained in the cited supporting sections. Sensitivity rows are descriptive unless an interval or test is explicitly specified. pp denotes percentage points. Model and policy contrasts use named finite families; the Holm check does not cover every sensitivity result or retrospectively confer preplanned-primary status. Intervals remain pointwise. Fixed transfer does not establish optimal rule parameters or neighbourhood robustness. No statistical comparison or sensitivity has been removed from this table.",
    }
    records = []
    fidelity = []
    caption_lines = ["# Nature-style core table plates", "", "The manuscript remains the scientific source. Vector tables use editable Arial text at 7 pt within a 180 mm publication width. Separate captions and notes preserve scope without placing dense prose beneath the plate.", ""]
    for number, source in tables.items():
        exported_csv = []
        for block_index, block in enumerate(source["blocks"]):
            suffix = "" if len(source["blocks"]) == 1 else chr(97 + block_index)
            path = OUT / f"table{number}{suffix}_source.csv"
            with path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(block["headers"])
                writer.writerows(block["rows"])
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                back = list(csv.reader(handle))
            assert back == [block["headers"]] + block["rows"], (number, suffix)
            fidelity.append({"table": number, "panel": suffix or None, "file": str(path.relative_to(OUT)),
                             "source_rows": len(block["rows"]), "source_columns": len(block["headers"]),
                             "exact_cell_readback": True, "sha256": digest(path)})
            exported_csv.append(str(path.relative_to(OUT)))
        config = descriptions[number]
        output = draw_table(number, source, config["layouts"], config["title"], config["panels"], config["numeric"])
        output["table_number"] = number
        output["csv_files"] = exported_csv
        records.append(output)
        caption_lines += [f"## Table {number}", "", source["caption"], "", f"Notes. {notes[number]}", ""]
    (OUT / "table_captions_and_notes.md").write_text("\n".join(caption_lines), encoding="utf-8")
    manifest = {"status": "rendered_pending_visual_review", "source_manuscript": str(SOURCE),
                "source_sha256": digest(SOURCE), "source_table_numbers": [1, 2, 3, 4, 5],
                "width_mm": WIDTH_MM, "body_font": "Arial", "body_font_pt": FONT_SIZE,
                "vector_formats": ["pdf", "svg"], "png_dpi": 1200, "preview_dpi": 300,
                "horizontal_rules_only": True, "vertical_rules": False,
                "cell_fidelity": fidelity, "table_exports": records,
                "notes_sha256": digest(OUT / "table_captions_and_notes.md")}
    (OUT / "tables_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"tables": len(records), "csv_blocks": len(fidelity), "source_sha256": digest(SOURCE),
                      "minimum_font_pt": FONT_SIZE, "manifest": str(OUT / "tables_manifest.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
