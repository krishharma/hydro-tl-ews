#!/usr/bin/env python3
"""Build a readable PDF from README.md for distribution."""
from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
OUT = ROOT / "docs" / "README.pdf"

INK = colors.HexColor("#1c2430")
MUTED = colors.HexColor("#5b6570")
ACCENT = colors.HexColor("#0b3d5c")
RULE = colors.HexColor("#c9d2da")
CODE_BG = colors.HexColor("#f3f5f7")
CODE_BORDER = colors.HexColor("#dde3ea")
TABLE_HEAD = colors.HexColor("#e7eef3")
TABLE_ALT = colors.HexColor("#f8fafb")
LINK = "#0b57a4"


def esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def inline_md(text: str) -> str:
    text = esc(text)
    text = text.replace("→", "&rarr;")
    text = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda m: f'<font color="{LINK}"><u>{m.group(1)}</u></font>',
        text,
    )
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(
        r"`([^`]+)`",
        r'<font face="Courier" size="8.5" color="#243447">\1</font>',
        text,
    )
    return text


def build_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleCustom",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=28,
            textColor=ACCENT,
            spaceAfter=8,
            alignment=TA_CENTER,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleCustom",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=11,
            leading=15,
            textColor=MUTED,
            spaceAfter=6,
            alignment=TA_CENTER,
        ),
        "meta": ParagraphStyle(
            "MetaCustom",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "h1": ParagraphStyle(
            "H1Custom",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=13.5,
            leading=17,
            textColor=ACCENT,
            spaceBefore=16,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "H2Custom",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=INK,
            spaceBefore=12,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "BodyCustom",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=15,
            textColor=INK,
            spaceBefore=1,
            spaceAfter=7,
            alignment=TA_JUSTIFY,
        ),
        "bullet": ParagraphStyle(
            "BulletCustom",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=INK,
            leftIndent=14,
            spaceBefore=1,
            spaceAfter=4,
        ),
        "code": ParagraphStyle(
            "CodeCustom",
            parent=base["Code"],
            fontName="Courier",
            fontSize=8,
            leading=11.5,
            textColor=colors.HexColor("#1f2933"),
            leftIndent=0,
            rightIndent=0,
            spaceBefore=0,
            spaceAfter=0,
        ),
        "th": ParagraphStyle(
            "THCustom",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=11.5,
            textColor=ACCENT,
        ),
        "td": ParagraphStyle(
            "TDCustom",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11.5,
            textColor=INK,
        ),
    }


def parse_table(lines: list[str]) -> list[list[str]]:
    """Parse markdown tables; treat \\| as a literal pipe inside cells."""
    rows = []
    for line in lines:
        if re.match(r"^\s*\|?\s*:?-{3,}", line):
            continue
        # Protect escaped pipes, then split.
        protected = line.replace("\\|", "\x00")
        cells = [c.strip().replace("\x00", "|") for c in protected.strip().strip("|").split("|")]
        rows.append(cells)
    # Normalize column count to the header width.
    if not rows:
        return rows
    ncols = len(rows[0])
    normalized = []
    for row in rows:
        if len(row) > ncols:
            row = row[: ncols - 1] + [" | ".join(row[ncols - 1 :])]
        while len(row) < ncols:
            row.append("")
        normalized.append(row)
    return normalized


def make_table(rows: list[list[str]], styles) -> Table:
    data = []
    for i, row in enumerate(rows):
        style = styles["th"] if i == 0 else styles["td"]
        data.append([Paragraph(inline_md(c), style) for c in row])

    ncols = len(data[0])
    width = 6.4 * inch
    if ncols == 2:
        col_widths = [2.3 * inch, 4.1 * inch]
    elif ncols == 3:
        col_widths = [2.15 * inch, 2.05 * inch, 2.2 * inch]
    else:
        col_widths = [width / ncols] * ncols

    table = Table(data, colWidths=col_widths, hAlign="LEFT", repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEAD),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("BOX", (0, 0), (-1, -1), 0.9, RULE),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), TABLE_ALT))
    table.setStyle(TableStyle(style_cmds))
    return table


def make_code_block(code: str, styles) -> Table:
    # Normalize box-drawing / fancy chars that Helvetica/Courier mishandle.
    code = (
        code.replace("├", "+")
        .replace("└", "+")
        .replace("─", "-")
        .replace("│", "|")
        .replace("→", "->")
    )
    lines = []
    for line in code.rstrip().splitlines() or [""]:
        while len(line) > 88:
            lines.append(line[:88])
            line = "    " + line[88:]
        lines.append(line)
    para = Paragraph(
        "<font face='Courier' size='8' color='#1f2933'>"
        + "<br/>".join(esc(l) if l else "&nbsp;" for l in lines)
        + "</font>",
        styles["code"],
    )
    box = Table([[para]], colWidths=[6.4 * inch])
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), CODE_BG),
                ("BOX", (0, 0), (-1, -1), 0.8, CODE_BORDER),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return box


def add_page_decor(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.6)
    y = 0.55 * inch
    canvas.line(0.8 * inch, y, letter[0] - 0.8 * inch, y)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(0.8 * inch, 0.35 * inch, "hydro-tl-ews  ·  README")
    canvas.drawRightString(letter[0] - 0.8 * inch, 0.35 * inch, f"{doc.page}")
    canvas.restoreState()


def build_pdf() -> Path:
    text = README.read_text(encoding="utf-8")
    styles = build_styles()
    story = []

    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("hydro-tl-ews", styles["title"]))
    story.append(
        Paragraph(
            "Transfer Learning for Hydrological Early Warning in Data-Scarce Regions",
            styles["subtitle"],
        )
    )
    story.append(
        Paragraph(
            "Project README &mdash; inputs, how to run, and expected outputs",
            styles["meta"],
        )
    )
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1.2, color=ACCENT, spaceAfter=16))

    lines = text.splitlines()
    i = 0
    while i < len(lines) and not lines[i].startswith("# "):
        i += 1
    if i < len(lines) and lines[i].startswith("# "):
        i += 1

    bullets: list[str] = []
    code_lines: list[str] = []
    in_code = False
    table_lines: list[str] = []

    def flush_bullets():
        nonlocal bullets
        if not bullets:
            return
        for b in bullets:
            story.append(
                Paragraph(f"&bull;&nbsp;&nbsp;{inline_md(b)}", styles["bullet"])
            )
        story.append(Spacer(1, 6))
        bullets = []

    def flush_code():
        nonlocal code_lines
        if not code_lines:
            return
        story.append(Spacer(1, 3))
        story.append(make_code_block("\n".join(code_lines), styles))
        story.append(Spacer(1, 10))
        code_lines = []

    def flush_table():
        nonlocal table_lines
        if not table_lines:
            return
        rows = parse_table(table_lines)
        if rows:
            story.append(Spacer(1, 4))
            story.append(make_table(rows, styles))
            story.append(Spacer(1, 12))
        table_lines = []

    while i < len(lines):
        raw = lines[i].rstrip()

        if raw.startswith("```"):
            flush_bullets()
            flush_table()
            if in_code:
                flush_code()
                in_code = False
            else:
                in_code = True
            i += 1
            continue

        if in_code:
            code_lines.append(raw)
            i += 1
            continue

        if raw.startswith("|"):
            flush_bullets()
            table_lines.append(raw)
            i += 1
            continue
        else:
            flush_table()

        if raw.strip() == "---":
            flush_bullets()
            story.append(Spacer(1, 4))
            story.append(
                HRFlowable(width="100%", thickness=0.7, color=RULE, spaceBefore=4, spaceAfter=12)
            )
            i += 1
            continue

        if raw.startswith("## "):
            flush_bullets()
            story.append(Spacer(1, 6))
            story.append(Paragraph(inline_md(raw[3:].strip()), styles["h1"]))
            i += 1
            continue

        if raw.startswith("### "):
            flush_bullets()
            story.append(Paragraph(inline_md(raw[4:].strip()), styles["h2"]))
            i += 1
            continue

        if re.match(r"^[-*] ", raw):
            bullets.append(raw[2:].strip())
            i += 1
            continue
        else:
            flush_bullets()

        if not raw.strip() or raw.startswith("# "):
            i += 1
            continue

        # Cover already shows the project subtitle — skip the duplicate lead line.
        if raw.strip().startswith("**Transfer Learning for Hydrological"):
            i += 1
            continue

        story.append(Paragraph(inline_md(raw), styles["body"]))
        i += 1

    flush_bullets()
    flush_table()
    flush_code()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=letter,
        leftMargin=0.8 * inch,
        rightMargin=0.8 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.9 * inch,
        title="hydro-tl-ews README",
        author="hydro-tl-ews",
    )
    doc.build(story, onFirstPage=add_page_decor, onLaterPages=add_page_decor)
    return OUT


if __name__ == "__main__":
    path = build_pdf()
    print(f"Wrote {path} ({path.stat().st_size} bytes)")
