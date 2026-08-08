"""Build a short, humanized methods/status paper for hydro-tl-ews.

Honest about compute: full CAMELS pretrain and multi-target study are designed
but not completed here. Numbers come from the mini sample pipeline (and smoke
test only as a packaging check). Schematic figures are reused from ``figures/``.

Usage::

    python scripts/paper/build_short_paper.py

Output::

    docs/Short_Paper_Hydro_TL_EWS.pdf
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figures"
SAMPLE = ROOT / "results" / "sample"
SMOKE = ROOT / "results" / "smoke"
OUT = ROOT / "docs" / "Short_Paper_Hydro_TL_EWS.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Fonts (same approach as the full manuscript builder)
# ---------------------------------------------------------------------------
FONT_DIR = Path("/tmp/hydro_fonts")
FONT_DIR.mkdir(exist_ok=True)
FONT_URLS = {
    "DMSans": "https://github.com/google/fonts/raw/main/ofl/dmsans/DMSans%5Bopsz%2Cwght%5D.ttf",
    "DMSans-Bold": "https://github.com/google/fonts/raw/main/ofl/dmsans/DMSans%5Bopsz%2Cwght%5D.ttf",
    "SourceSans": "https://github.com/google/fonts/raw/main/ofl/sourcesans3/SourceSans3%5Bwght%5D.ttf",
    "SourceSans-Bold": "https://github.com/google/fonts/raw/main/ofl/sourcesans3/SourceSans3%5Bwght%5D.ttf",
    "SourceSans-Italic": "https://github.com/google/fonts/raw/main/ofl/sourcesans3/SourceSans3-Italic%5Bwght%5D.ttf",
}


def _ensure_fonts() -> None:
    for name, url in FONT_URLS.items():
        path = FONT_DIR / f"{name}.ttf"
        if not path.exists():
            try:
                urllib.request.urlretrieve(url, path)
            except Exception:
                pass


_ensure_fonts()
REGISTERED: list[str] = []
for name in FONT_URLS:
    p = FONT_DIR / f"{name}.ttf"
    if p.exists():
        try:
            pdfmetrics.registerFont(TTFont(name, str(p)))
            REGISTERED.append(name)
        except Exception:
            pass

HEADER = "DMSans-Bold" if "DMSans-Bold" in REGISTERED else "Helvetica-Bold"
BODY = "SourceSans" if "SourceSans" in REGISTERED else "Helvetica"
BODY_B = "SourceSans-Bold" if "SourceSans-Bold" in REGISTERED else "Helvetica-Bold"
BODY_I = "SourceSans-Italic" if "SourceSans-Italic" in REGISTERED else "Helvetica-Oblique"

PRIMARY = HexColor("#01696F")
PRIMARY_DARK = HexColor("#0C4E54")
ACCENT = HexColor("#A84B2F")
INK = HexColor("#28251D")
MUTED = HexColor("#7A7974")
FAINT = HexColor("#D4D1CA")
SURFACE = HexColor("#F9F8F5")

ss = getSampleStyleSheet()
S_TITLE = ParagraphStyle(
    "T", parent=ss["Title"], fontName=HEADER, fontSize=16, leading=20,
    textColor=INK, spaceAfter=8, alignment=0,
)
S_SUB = ParagraphStyle(
    "Sub", parent=ss["Normal"], fontName=BODY_I, fontSize=10, leading=13,
    textColor=MUTED, spaceAfter=12,
)
S_AUTHOR = ParagraphStyle(
    "Auth", parent=ss["Normal"], fontName=BODY, fontSize=10, leading=13,
    textColor=INK, spaceAfter=2,
)
S_AFFIL = ParagraphStyle(
    "Aff", parent=ss["Normal"], fontName=BODY_I, fontSize=9, leading=12,
    textColor=MUTED, spaceAfter=14,
)
S_H1 = ParagraphStyle(
    "H1s", parent=ss["Heading1"], fontName=HEADER, fontSize=12.5, leading=16,
    textColor=PRIMARY_DARK, spaceBefore=14, spaceAfter=6,
)
S_H2 = ParagraphStyle(
    "H2s", parent=ss["Heading2"], fontName=HEADER, fontSize=10.5, leading=13,
    textColor=PRIMARY_DARK, spaceBefore=8, spaceAfter=3,
)
S_BODY = ParagraphStyle(
    "Bd", parent=ss["Normal"], fontName=BODY, fontSize=9.8, leading=13.5,
    textColor=INK, alignment=4, spaceAfter=6,
)
S_NOTE = ParagraphStyle(
    "Note", parent=S_BODY, fontName=BODY_I, fontSize=9.2, leading=12.5,
    textColor=ACCENT, backColor=SURFACE, borderPadding=8, spaceAfter=10,
)
S_ABS_L = ParagraphStyle(
    "AbsL", parent=ss["Normal"], fontName=BODY_B, fontSize=10, leading=13,
    textColor=PRIMARY_DARK, spaceAfter=3,
)
S_ABS = ParagraphStyle(
    "Abs", parent=S_BODY, fontSize=9.4, leading=12.8,
    leftIndent=10, rightIndent=10, spaceAfter=8,
)
S_CAP = ParagraphStyle(
    "Cap", parent=ss["Normal"], fontName=BODY_I, fontSize=8.5, leading=11,
    textColor=MUTED, alignment=1, spaceAfter=10, spaceBefore=2,
)
S_KW = ParagraphStyle(
    "Kw", parent=ss["Normal"], fontName=BODY_I, fontSize=9, leading=12,
    textColor=MUTED, spaceAfter=10,
)
S_REF = ParagraphStyle(
    "Ref", parent=ss["Normal"], fontName=BODY, fontSize=8.5, leading=11.5,
    textColor=INK, leftIndent=14, firstLineIndent=-14, spaceAfter=3,
)


def P(text: str, style=S_BODY) -> Paragraph:
    return Paragraph(text, style)


def H1(text: str) -> Paragraph:
    return Paragraph(text, S_H1)


def H2(text: str) -> Paragraph:
    return Paragraph(text, S_H2)


def _aspect(path: Path) -> float:
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        return im.height / im.width


def figure(name: str, caption: str, width: float = 6.2 * inch):
    path = FIG / name
    if not path.exists():
        return P(f"[Missing figure: {name}]", S_CAP)
    img = Image(str(path), width=width, height=width * _aspect(path))
    return KeepTogether([img, P(caption, S_CAP)])


def styled_table(headers, rows, col_widths=None):
    data = [[Paragraph(f"<b>{h}</b>", ParagraphStyle(
        "th", fontName=BODY_B, fontSize=8, textColor=HexColor("#FFFFFF"), leading=10
    )) for h in headers]]
    cell = ParagraphStyle("td", fontName=BODY, fontSize=8, leading=10, textColor=INK)
    for row in rows:
        data.append([Paragraph(str(c), cell) for c in row])
    t = Table(data, colWidths=col_widths, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY_DARK),
        ("BACKGROUND", (0, 1), (-1, -1), SURFACE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [SURFACE, HexColor("#FFFFFF")]),
        ("GRID", (0, 0), (-1, -1), 0.4, FAINT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def header_footer(c, doc):
    c.saveState()
    c.setFont(BODY, 8)
    c.setFillColor(MUTED)
    c.drawString(72, 30, "Short paper · hydro-tl-ews · compute-limited status")
    c.drawRightString(LETTER[0] - 72, 30, f"{doc.page}")
    c.setStrokeColor(FAINT)
    c.setLineWidth(0.5)
    c.line(72, 42, LETTER[0] - 72, 42)
    c.restoreState()


def first_page(c, doc):
    header_footer(c, doc)
    c.saveState()
    c.setFillColor(PRIMARY)
    c.rect(72, LETTER[1] - 56, 48, 3.5, fill=1, stroke=0)
    c.setFont(HEADER, 8)
    c.setFillColor(PRIMARY_DARK)
    c.drawString(72, LETTER[1] - 48, "SHORT PAPER · METHODS &amp; STATUS NOTE")
    c.restoreState()


def _fmt(x, nd=2):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if v != v:
        return "—"
    return f"{v:.{nd}f}"


def build_story():
    sm = json.loads((SAMPLE / "metrics.json").read_text())
    cont = sm["continuous"]
    ew = sm["early_warning"]
    smoke = None
    if (SMOKE / "summary.json").exists():
        smoke = json.loads((SMOKE / "summary.json").read_text())

    s = []
    s.append(P(
        "Transfer Learning for Hydrological Early Warning in Data-Scarce Basins: "
        "A Short Methods Note",
        S_TITLE,
    ))
    s.append(P(
        "What this project is trying to do, how it is built, what goes in and out, "
        "and what we can and cannot claim yet.",
        S_SUB,
    ))
    s.append(P("Krish Sharma", S_AUTHOR))
    s.append(P("Independent research note · August 2026", S_AFFIL))

    s.append(P("Abstract", S_ABS_L))
    s.append(P(
        "Most rivers on Earth are poorly gauged. That makes local flood and drought "
        "models hard to trust: if you only have a couple of years of streamflow, "
        "training a neural network from scratch is almost a non-starter. This project "
        "asks a practical question: can we pre-train an Entity-Aware LSTM on many "
        "CAMELS-US donor catchments, then fine-tune it on a short target warmup and "
        "evaluate it the way an operational desk would — rolling forward through "
        "time, not with a lucky random split? The full pipeline is implemented "
        "end-to-end in code. What is <i>not</i> finished is the expensive continental "
        "pretrain and the seven-basin multi-regime study. Those stages need days of "
        "GPU/Apple-Silicon time and the full ~14&nbsp;GB CAMELS extract. This short "
        "paper therefore explains the objective, data, methodology, and I/O contract "
        "in plain language, shows schematic figures from the design, and reports "
        "only the numbers we actually have: a few-minute sample pipeline on a "
        "synthetic mini-CAMELS archive that exercises the real loaders and trainers. "
        "Those sample metrics are for pipeline validation, not scientific claims.",
        S_ABS,
    ))
    s.append(P(
        "Keywords: transfer learning · EA-LSTM · CAMELS · walk-forward validation · "
        "flood early warning · data-scarce basins · computation limits",
        S_KW,
    ))

    s.append(P(
        "<b>Honest framing.</b> Treat everything labeled “sample” as a dress rehearsal. "
        "The intended Merced River (USGS&nbsp;11264500) study and the seven-basin "
        "extension are designed and configured, but the full CAMELS run is incomplete "
        "simply because of computation and data footprint — not because the method "
        "is unfinished in software.",
        S_NOTE,
    ))

    # 1
    s.append(H1("1. Objective"))
    s.append(P(
        "The goal is an early-warning-ready streamflow workflow for basins that "
        "look like they only have ~2 years of usable local data. Concretely:"
    ))
    s.append(P(
        "1.&nbsp;<b>Learn a regional rainfall–runoff representation</b> by pre-training "
        "an EA-LSTM on many CAMELS-US donors (holding the target out, plus a "
        "spatial buffer)."
    ))
    s.append(P(
        "2.&nbsp;<b>Adapt with little local data</b> via conservative head-only fine-tuning "
        "(Approach&nbsp;A) or progressive unfreezing (Approach&nbsp;B), and compare "
        "against zero-shot transfer and a from-scratch local baseline."
    ))
    s.append(P(
        "3.&nbsp;<b>Evaluate like operations</b>: a rolling-origin walk-forward loop with "
        "periodic refits, online bias correction, and at-site flood/drought "
        "thresholds (Q5 / Q95 / Q99) turned into lead-time warning probabilities."
    ))
    s.append(P(
        "4.&nbsp;<b>Keep the story inspectable</b>: continuous skill (NSE, KGE, PBIAS) plus "
        "warning scores (AUC, F1, Brier), with optional SHAP attributions."
    ))
    s.append(P(
        "The default single-target story is the Merced River at Happy Isles "
        "(USGS&nbsp;11264500), a Sierra Nevada snowmelt basin. A multi-target "
        "extension stretches the same recipe across seven gauges spanning maritime "
        "rain, mixed snow/rain, Rockies snowmelt, humid Southeast, continental "
        "plains, and a semi-arid ephemeral stream."
    ))

    # 2
    s.append(H1("2. Data"))
    s.append(H2("2.1 Intended dataset (full study)"))
    s.append(P(
        "The scientific runs are designed for <b>CAMELS-US</b> (Newman et al.; Addor "
        "et al.): daily Daymet basin forcings, USGS streamflow, and static "
        "catchment attributes. Unpacked size is on the order of ~14&nbsp;GB under "
        "<font face='Courier'>data/</font>. Dynamic inputs are precipitation, "
        "Tmax/Tmin, shortwave radiation, vapor pressure, and day length. Static "
        "inputs cover topography, climate indices, land cover, soils, and geology "
        "(~27 attributes in the project’s canonical list)."
    ))
    s.append(P(
        "Pretrain periods are long (roughly water years 1980–2010 for training, "
        "with a later donor validation window). Target evaluation is intended for "
        "2011–2014 under a 2-year warmup ending 2010-12-31. Extreme thresholds "
        "are fit on a long pre-evaluation climatology so Q5/Q95/Q99 are not "
        "estimated from the scarce warmup alone."
    ))
    s.append(figure(
        "fig_camels_map.png",
        "Figure 1. CAMELS context map used in the project materials. The full "
        "continental donor set is required for the real pretrain; it is not "
        "bundled in the repository.",
        width=5.8 * inch,
    ))

    s.append(H2("2.2 What we actually ran"))
    s.append(P(
        "Without committing days of machine time to full CAMELS, the repo ships "
        "two lighter paths:"
    ))
    s.append(P(
        "• <b>Smoke test</b> — fully synthetic basins; checks that training, "
        "transfer, walk-forward, and warning math wire together."
    ))
    s.append(P(
        "• <b>Sample CAMELS pipeline</b> — a tiny layout-compatible archive "
        "(~8 synthetic basins) generated by "
        "<font face='Courier'>scripts/make_sample_camels.py</font>, then run "
        "through the real <font face='Courier'>CamelsDataset</font> loader "
        "(pretrain → fine-tune → walk-forward in minutes)."
    ))
    s.append(figure(
        "fig_compute_status.png",
        "Figure 2. Status of the experimental ladder. Smoke and sample stages "
        "are complete; subset-200 / full CAMELS / multi-target remain blocked on "
        "compute and the full data download.",
        width=6.0 * inch,
    ))

    # 3
    s.append(H1("3. Methodology"))
    s.append(H2("3.1 Model"))
    s.append(P(
        "We use an <b>Entity-Aware LSTM (EA-LSTM)</b> after Kratzert et al. (2019). "
        "Catchment attributes feed a time-invariant input gate; daily forcings "
        "drive the forget, candidate, and output gates. That split is the reason "
        "one network can behave like a regional model while still “knowing” which "
        "basin it is looking at."
    ))
    s.append(figure(
        "fig1_architecture.png",
        "Figure 3. Framework schematic: regional pretrain → target adaptation → "
        "walk-forward evaluation → early-warning post-process (and optional SHAP).",
        width=6.2 * inch,
    ))

    s.append(H2("3.2 Transfer recipes"))
    s.append(P(
        "<b>Approach A (conservative)</b> freezes the LSTM cell and trains only the "
        "dense head on the target warmup — the safer choice when local data are "
        "scarce. <b>Approach B (progressive)</b> does a short head-only phase, then "
        "unfreezes the last fraction of LSTM parameters at a much smaller learning "
        "rate. Both are compared to zero-shot (no fine-tune) and a local baseline "
        "trained from scratch on the same warmup."
    ))
    s.append(figure(
        "fig3_unfreezing.png",
        "Figure 4. Fine-tuning strategies: head-only vs. progressive unfreezing.",
        width=5.6 * inch,
    ))

    s.append(H2("3.3 Walk-forward evaluation and early warning"))
    s.append(P(
        "Instead of shuffling days into train/test, the walk-forward stage expands "
        "origin-by-origin: predict a block, score it, optionally refit (every 90 "
        "days in the shipped configs), apply online bias correction, and continue. "
        "Configs pin <font face='Courier'>refit_train_start</font> so refits cannot "
        "quietly train on decades of local flow that the “data-scarce” story was "
        "supposed to withhold."
    ))
    s.append(P(
        "Early warning is a post-process on the hindcast series: map residual "
        "uncertainty into probabilities that flow crosses at-site Q5/Q95/Q99 within "
        "lead windows (1, 3, 7 days). Scores are AUC, F1@0.5, and Brier. This is "
        "not a true issued multi-day forecast without future forcings — an important "
        "limitation called out in the project docs."
    ))
    s.append(figure(
        "fig2_walk_forward.png",
        "Figure 5. Rolling-origin walk-forward protocol used for operational-style "
        "evaluation.",
        width=5.8 * inch,
    ))
    s.append(figure(
        "fig4_rfa_thresholds.png",
        "Figure 6. Long-record extreme thresholds (Q5 / Q95 / Q99) used to define "
        "drought and flood warning events.",
        width=5.4 * inch,
    ))

    # 4
    s.append(H1("4. Expected inputs and outputs"))
    s.append(H2("4.1 Inputs"))
    s.append(styled_table(
        ["Input", "Needed for", "Notes"],
        [
            ["Python ≥ 3.10 + deps", "Always", "venv / conda; optional editable install"],
            ["YAML under configs/", "Every stage", "paths, periods, checkpoints, leads"],
            ["CAMELS-US under data/", "Real stages", "~14 GB; see data/README.md"],
            ["Upstream .pt checkpoint", "Transfer / WF / multi-target", "e.g. pretrain_subset200.pt"],
            ["Sample/smoke only", "Quick checks", "No full CAMELS required"],
        ],
        col_widths=[2.0 * inch, 1.7 * inch, 2.6 * inch],
    ))
    s.append(Spacer(1, 8))
    s.append(H2("4.2 Outputs (under results/)"))
    s.append(styled_table(
        ["Artifact", "Typical path", "Contents"],
        [
            ["Checkpoint", "checkpoints/*.pt", "EA-LSTM weights + config"],
            ["Train history", "history/*.json", "epoch train/val loss"],
            ["Walk-forward series", "walk_forward.parquet", "obs / pred / bias"],
            ["Metrics", "walk_forward_metrics.json", "NSE, KGE, PBIAS + EWS"],
            ["Warnings", "walk_forward_warnings.csv", "labels + probs by lead"],
            ["SHAP (optional)", "shap_global_importance.csv", "mean |SHAP| by feature"],
            ["Multi-target", "multi_target/&lt;id&gt;/", "per-basin copies + summary"],
        ],
        col_widths=[1.6 * inch, 2.1 * inch, 2.6 * inch],
    ))
    s.append(Spacer(1, 6))
    s.append(P(
        "Everyday commands (from the repo root): "
        "<font face='Courier'>python scripts/run_experiment.py --config configs/&lt;stage&gt;.yaml</font>; "
        "full single-target ladder via "
        "<font face='Courier'>bash scripts/run_full_pipeline.sh</font>; "
        "seven-basin study via "
        "<font face='Courier'>python scripts/run_multi_target.py</font> after a full "
        "<font face='Courier'>pretrain.pt</font> exists."
    ))

    # 5
    s.append(H1("5. What we can show today"))
    s.append(P(
        "The sample pipeline finished with "
        f"{sm['n_predictions']} daily predictions and {sm['n_refits']} refits on "
        f"synthetic target basin {sm['target_basin']}. Continuous skill on that "
        "toy series is not meaningful as hydrology — NSE can go deeply negative "
        "when a tiny model meets synthetic seasonality — but the early-warning "
        "plumbing does emit finite AUC/F1/Brier values, which is useful as a "
        "systems check."
    ))
    s.append(styled_table(
        ["Quantity", "Sample pipeline value", "How to read it"],
        [
            ["NSE", _fmt(cont["NSE"]), "Pipeline ran; not a scientific claim"],
            ["KGE", _fmt(cont["KGE"]), "Same caveat"],
            ["PBIAS (%)", _fmt(cont["PBIAS"]), "Same caveat"],
            ["Flood Q95 AUC (lead 1 d)", _fmt(ew["flood_q95_lead1d"]["AUC"]), "Warning path exercised"],
            ["Flood Q95 AUC (lead 3 d)", _fmt(ew["flood_q95_lead3d"]["AUC"]), "Warning path exercised"],
            ["Flood Q95 F1@0.5 (lead 3 d)", _fmt(ew["flood_q95_lead3d"]["F1@0.5"]), "Thresholded skill on toy data"],
            ["n_predictions / n_refits", f"{sm['n_predictions']} / {sm['n_refits']}", "Protocol dimensions"],
        ],
        col_widths=[2.2 * inch, 1.6 * inch, 2.5 * inch],
    ))
    s.append(Spacer(1, 6))
    s.append(P(
        "Table 1. Metrics from <font face='Courier'>results/sample/metrics.json</font>. "
        "These validate that the real CAMELS loader, trainer, walk-forward loop, and "
        "warning mapper execute together. They do <b>not</b> estimate skill on Merced "
        "or any other real basin.",
        S_CAP,
    ))

    if smoke is not None:
        smk = smoke["metrics"]
        s.append(P(
            "For completeness, the synthetic smoke summary also completed "
            f"({smoke.get('n_predictions', '—')} predictions). Walk-forward NSE there "
            f"was {_fmt(smk['walk_forward']['NSE'])} with flood Q95 lead-3 AUC "
            f"{_fmt(smk['early_warning']['flood_q95_lead3d']['AUC'])}. Smoke numbers "
            "are even further from real hydrology; they only prove the install."
        ))

    s.append(figure(
        "fig_sample_hydrograph.png",
        "Figure 7. Excerpt of the sample walk-forward hydrograph "
        "(synthetic mini-CAMELS). Observed and predicted series are layout-compatible "
        "stand-ins, not CAMELS observations.",
        width=6.0 * inch,
    ))

    s.append(H2("5.1 Why the full run is not here"))
    s.append(P(
        "Rough compute expectations from the project running guide: smoke/sample "
        "finish in minutes; <font face='Courier'>pretrain_subset200</font> is "
        "hours to a day+ on Apple Silicon or a mid GPU; full CAMELS pretrain is "
        "multi-day on a laptop; one-basin walk-forward is hours; the seven-basin "
        "multi-target study is days after pretrain. Sequence length 365 and batch "
        "size 256 dominate memory and time. This note stops before those stages "
        "on purpose: the software path is ready, the machines (and the 14&nbsp;GB "
        "extract) were the bottleneck for a complete results section."
    ))
    s.append(styled_table(
        ["Stage", "Approx. cost", "Status here"],
        [
            ["Smoke / unit tests", "Minutes (CPU)", "Done"],
            ["Sample CAMELS pipeline", "Minutes", "Done (Table 1 / Fig. 7)"],
            ["pretrain_subset200", "Hours–1+ day", "Not run"],
            ["Full CAMELS pretrain", "Multi-day (prefer CUDA)", "Not run"],
            ["Single-target walk-forward", "Hours", "Not run (needs checkpoint)"],
            ["7-basin multi-target", "Days after pretrain", "Not run"],
        ],
        col_widths=[2.2 * inch, 2.0 * inch, 2.1 * inch],
    ))
    s.append(Spacer(1, 6))
    s.append(P("Table 2. Experimental ladder and compute status.", S_CAP))

    # 6
    s.append(H1("6. Limitations (beyond compute)"))
    s.append(P(
        "Even after a full run, a few scientific caveats remain. Early warning "
        "scores hindcast crossings inside a lead window; it is not a pure forecast "
        "issued at time <i>t</i> without future forcings. Compound warning "
        "probabilities assume independence and can overstate risk under "
        "autocorrelated flow. SHAP in the shipped path is a simplified "
        "DeepExplainer view and should stay qualitative. Static normalizers in "
        "transfer stages are currently fit on a broader attribute table than a "
        "strict donor-only recipe would prefer."
    ))

    # 7
    s.append(H1("7. Closing"))
    s.append(P(
        "If you only remember three things: (1) the objective is transfer learning "
        "for early warning when local records are short; (2) the method is EA-LSTM "
        "pretrain → scarce-data fine-tune → walk-forward evaluation with quantile "
        "warnings; (3) this document is intentionally a methods-and-status note. "
        "The code and configs for the Merced single-target ladder and the "
        "seven-basin study are in the repository. Filling Tables of real NSE/KGE "
        "and flood AUC is mostly a matter of machine time and the CAMELS download — "
        "not a redesign of the approach."
    ))

    s.append(H1("References (selected)"))
    refs = [
        "Addor, N., et al. (2017). The CAMELS data set: catchment attributes and meteorology for large-sample studies. HESS.",
        "Kratzert, F., et al. (2019). Towards learning universal, regional, and local hydrological behaviours via machine learning applied to large-sample datasets. HESS, 23, 5089–5110.",
        "Newman, A. J., et al. (2015). Development of a large-sample watershed-scale hydrometeorological data set for the contiguous USA: CAMELS. NCAR.",
        "Project documentation: README.md; docs/RUNNING.md; docs/OUTPUTS.md; docs/KNOWN_LIMITATIONS.md.",
    ]
    for r in refs:
        s.append(P(r, S_REF))

    return s


def main() -> None:
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=LETTER,
        leftMargin=0.9 * inch,
        rightMargin=0.9 * inch,
        topMargin=0.85 * inch,
        bottomMargin=0.7 * inch,
        title="Transfer Learning for Hydrological Early Warning — Short Paper",
        author="Krish Sharma",
    )
    doc.build(build_story(), onFirstPage=first_page, onLaterPages=header_footer)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
