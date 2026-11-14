"""
Fill slides 3 and 4 of the Part C deck from the benchmark CSV written by bench_rf.py.

    python fill_deck.py parallel_bench/compute_timing.csv PartC_Parallel_Computing_120s.pptx PartC_filled.pptx
    python fill_deck.py parallel_bench/compute_timing.csv in.pptx out.pptx --use best

Slide 3: speedup chart (ideal + Random Forest + K-Means), results table, footnote, speaker notes;
         the orange TO FILL box is deleted.
Slide 4: the hypothetical "4.7x" box is replaced by the parallel fraction f fitted (Amdahl's law)
         to the measured Random Forest speedups; speaker notes updated.
Read the generated speaker notes afterwards and edit them so they say what you want to say.
"""
import argparse

import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.chart.data import XyChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_MARKER_STYLE
from pptx.enum.dml import MSO_LINE
from pptx.util import Pt

RED, BLUE, GREY = RGBColor(0xB6, 0x00, 0x07), RGBColor(0x15, 0x60, 0x82), RGBColor(0x6B, 0x6B, 0x6B)
STAGES = [("rf", "Random Forest", RED), ("kmeans", "K-Means", BLUE)]

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("deck")
ap.add_argument("out")
ap.add_argument("--use", choices=["mean", "best"], default="mean",
                help="speedup based on mean times (default) or on best runs")
a = ap.parse_args()

df = pd.read_csv(a.csv)
col = "speedup_mean" if a.use == "mean" else "speedup_best"
stages_present = [s for s in STAGES if s[0] in set(df.stage)]
workers = sorted(int(w) for w in df["workers"].unique())


def get(stage, p):
    r = df[(df.stage == stage) & (df.workers == p)]
    return float(r[col].iloc[0]) if len(r) else None


def fit_amdahl(stage):
    """Least-squares parallel fraction f in S(p) = 1 / ((1 - f) + f / p)."""
    ps = np.array([p for p in workers if get(stage, p) is not None], float)
    ss = np.array([get(stage, int(p)) for p in ps])
    grid = np.linspace(0, 1, 1001)
    err = [np.sum((1 / ((1 - f) + f / ps) - ss) ** 2) for f in grid]
    return float(grid[int(np.argmin(err))])


prs = Presentation(a.deck)
s3, s4 = prs.slides[2], prs.slides[3]
n3 = {sh.name: sh for sh in s3.shapes}
n4 = {sh.name: sh for sh in s4.shapes}

# ---------------- slide 3: chart ----------------
cd = XyChartData()
ideal = cd.add_series("Ideal (linear)")
for p in workers:
    ideal.add_data_point(p, p)
for key, label, _ in stages_present:
    ser = cd.add_series(label)
    for p in workers:
        v = get(key, p)
        if v is not None:
            ser.add_data_point(p, round(v, 3))
chart = n3["Chart speedup"].chart
chart.replace_data(cd)
sers = list(chart.plots[0].series)
sers[0].format.line.color.rgb = GREY
sers[0].format.line.dash_style = MSO_LINE.DASH
sers[0].format.line.width = Pt(2)
sers[0].marker.style = XL_MARKER_STYLE.NONE
sers[0].smooth = False
for ser, (_, _, c) in zip(sers[1:], stages_present):
    ser.format.line.color.rgb = c
    ser.format.line.width = Pt(3)
    ser.smooth = False
    ser.marker.style = XL_MARKER_STYLE.CIRCLE
    ser.marker.size = 8
    ser.marker.format.fill.solid()
    ser.marker.format.fill.fore_color.rgb = c
    ser.marker.format.line.color.rgb = c
chart.category_axis.maximum_scale = max(workers)
chart.value_axis.maximum_scale = max(workers)

# ---------------- slide 3: table ----------------
tbl = n3["Table results"].table
for r, (key, label, _) in enumerate(STAGES, start=1):
    s4_, s8_ = get(key, 4), get(key, 8)
    vals = [label,
            f"{s4_:.2f}×" if s4_ is not None else "n/a",
            f"{s8_:.2f}×" if s8_ is not None else "n/a",
            f"{s8_ / 8 * 100:.0f}%" if s8_ is not None else "n/a"]
    for c, v in enumerate(vals):
        tbl.cell(r, c).text_frame.paragraphs[0].runs[0].text = v

if "Fill-in tag" in n3:
    el = n3["Fill-in tag"]._element
    el.getparent().remove(el)
n_rows = int(df[df.stage == "rf"]["n_rows"].iloc[0])
reps = int(df[df.stage == "rf"]["reps"].iloc[0])
for sh in s3.shapes:
    if sh.has_text_frame and sh.text_frame.text.startswith("Plan:"):
        sh.text_frame.paragraphs[0].runs[0].text = (
            f"Apple M1 Pro (6P + 2E cores); Random Forest on {n_rows:,} rows (20 trees, depth 20, seed 42); "
            f"{a.use}-based, {reps} interleaved runs per setting.")


def fx(key, p):
    v = get(key, p)
    return f"{v:.1f}" if v is not None else "n/a"


rf8 = get("rf", 8)
eff = f"{rf8 / 8 * 100:.0f}" if rf8 is not None else "n/a"
s3.notes_slide.notes_text_frame.text = (
    "Now the compute-heavy stage: Random Forest with 20 trees. "
    f"With 8 workers, training is {fx('rf', 8)} times faster, which is {eff} percent efficiency. "
    "Why not 8 times? With 20 trees, the busiest worker builds 3, so the limit is about 6.7. "
    "[Edit to match the story in your data.]")

# ---------------- slide 4: fitted Amdahl fraction ----------------
f = fit_amdahl("rf") if "rf" in set(df.stage) else None
if f is not None:
    n4["Example number"].text_frame.paragraphs[0].runs[0].text = f"f ≈ {f:.2f}"
    for sh in s4.shapes:
        if sh.has_text_frame and sh.text_frame.text.startswith("upper bound at 8 workers"):
            sh.text_frame.paragraphs[0].runs[0].text = "parallel fraction fitted to our Random Forest speedups"
    s4.notes_slide.notes_text_frame.text = (
        "Amdahl's law says the serial part limits the gain: "
        f"fitted to our Random Forest results, about {f * 100:.0f} percent of the work is parallel. "
        "Uneven tree splits, shared memory bandwidth and two slower efficiency cores likely add to it. "
        "In short: parallelism pays off for independent, compute-heavy tasks, but not for I/O-bound ones. Thank you.")

prs.save(a.out)
print("saved", a.out)
if f is not None:
    print(f"Fitted Amdahl parallel fraction (Random Forest, {a.use}-based): f = {f:.3f}")
print("\nSlide 3 notes:\n", s3.notes_slide.notes_text_frame.text)
print("\nSlide 4 notes:\n", s4.notes_slide.notes_text_frame.text)
