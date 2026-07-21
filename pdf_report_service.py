import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


# ── Brand colour palette ──────────────────────────────────────────────────────
PRIMARY       = colors.HexColor("#667eea")
SECONDARY     = colors.HexColor("#764ba2")
ACCENT        = colors.HexColor("#1e293b")
ROW_ALT       = colors.HexColor("#f8f9ff")
ROW_NORMAL    = colors.HexColor("#ffffff")
HEADER_BG     = colors.HexColor("#667eea")
HEADER_FG     = colors.white
BORDER_COLOR  = colors.HexColor("#e2e8f0")
TEXT_MUTED    = colors.HexColor("#64748b")
DANGER        = colors.HexColor("#ef4444")


class PdfReportService:
    """Generates styled PDF fleet management reports from raw query data."""

    def __init__(self, reports_dir: str = None):
        # Default to a 'reports' folder next to this file
        if reports_dir is None:
            reports_dir = os.path.join(os.path.dirname(__file__), "reports")
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        data: List[Dict[str, Any]],
        title: str = "Fleet Management Report",
        user_question: str = "",
    ) -> Dict[str, Any]:
        """
        Build a styled PDF from raw row data.

        Returns a dict with:
            filename   – base name of the PDF file
            filepath   – absolute path on disk
            url_path   – relative URL path for download (/reports/<filename>)
            record_count – number of data rows written
        """
        if not data:
            return {"error": "No data available to generate a report."}

        filename = f"fleet_report_{uuid.uuid4().hex[:10]}.pdf"
        filepath = os.path.join(self.reports_dir, filename)

        try:
            self._build_pdf(filepath, data, title, user_question)
            print(f"[PDF REPORT] Generated: {filepath} ({len(data)} records)")
            return {
                "filename":     filename,
                "filepath":     filepath,
                "url_path":     f"/reports/{filename}",
                "record_count": len(data),
            }
        except Exception as e:
            print(f"[PDF REPORT ERROR] {e}")
            return {"error": str(e)}

    # ── Internal builders ─────────────────────────────────────────────────────

    def _build_pdf(
        self,
        filepath: str,
        data: List[Dict[str, Any]],
        title: str,
        user_question: str,
    ) -> None:
        """Assemble and save the full PDF document."""

        doc = SimpleDocTemplate(
            filepath,
            pagesize=A4,
            leftMargin=1.5 * cm,
            rightMargin=1.5 * cm,
            topMargin=2 * cm,
            bottomMargin=2.5 * cm,
            title=title,
            author="Fleet Management Assistant",
        )

        styles = self._make_styles()
        story  = []

        # ── Cover / header band ───────────────────────────────────────────
        story += self._build_header(title, user_question, len(data), styles)
        story.append(Spacer(1, 0.4 * cm))

        # ── Summary statistics strip ──────────────────────────────────────
        summary_items = self._extract_summary(data)
        if summary_items:
            story += self._build_summary_band(summary_items, styles)
            story.append(Spacer(1, 0.5 * cm))

        # ── Data table ────────────────────────────────────────────────────
        story.append(Paragraph("Data Records", styles["section_title"]))
        story.append(Spacer(1, 0.2 * cm))
        story += self._build_data_table(data, styles)

        # ── Build with page numbering ─────────────────────────────────────
        doc.build(
            story,
            onFirstPage=self._page_footer,
            onLaterPages=self._page_footer,
        )

    # ── Header ────────────────────────────────────────────────────────────────

    def _build_header(
        self,
        title: str,
        user_question: str,
        record_count: int,
        styles: dict,
    ) -> list:
        elements = []

        # Gradient-look banner implemented as a 1-row table
        banner_data = [[
            Paragraph("🚛  Fleet Management Assistant", styles["brand"]),
            Paragraph(
                datetime.now().strftime("%B %d, %Y  %H:%M"),
                styles["date_right"],
            ),
        ]]
        banner = Table(banner_data, colWidths=["70%", "30%"])
        banner.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, -1), PRIMARY),
            ("TEXTCOLOR",   (0, 0), (-1, -1), HEADER_FG),
            ("LEFTPADDING",  (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING",   (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 10),
            ("ROUNDEDCORNERS", [6]),
        ]))
        elements.append(banner)
        elements.append(Spacer(1, 0.4 * cm))

        # Report title
        elements.append(Paragraph(title, styles["report_title"]))
        elements.append(Spacer(1, 0.15 * cm))

        # Sub-info row
        if user_question:
            elements.append(
                Paragraph(f"<i>Query:</i> {user_question}", styles["query_label"])
            )
            elements.append(Spacer(1, 0.1 * cm))

        elements.append(
            Paragraph(
                f"Total records: <b>{record_count:,}</b>",
                styles["query_label"],
            )
        )
        elements.append(Spacer(1, 0.3 * cm))
        elements.append(HRFlowable(width="100%", thickness=1, color=PRIMARY))

        return elements

    # ── Summary band ──────────────────────────────────────────────────────────

    def _extract_summary(self, data: List[Dict[str, Any]]) -> List[Dict]:
        """Pull key stats for the summary strip."""
        if not data:
            return []

        items = []
        first = data[0]

        # Vehicle identity
        for key in ("vehicle_id", "license_plate_number"):
            if key in first and first[key]:
                items.append({"label": key.replace("_", " ").title(), "value": str(first[key])})

        # Numeric aggregate helpers
        def _col_values(col):
            vals = []
            for row in data:
                v = row.get(col)
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    pass
            return vals

        # Distance
        for dist_col in ("trip_distance_miles", "distance_miles", "distance"):
            vals = _col_values(dist_col)
            if vals:
                items.append({
                    "label": "Total Distance (mi)",
                    "value": f"{sum(vals):,.1f}",
                })
                break

        # Score
        for score_col in ("trip_score", "score"):
            vals = _col_values(score_col)
            if vals:
                items.append({
                    "label": "Avg Trip Score",
                    "value": f"{sum(vals)/len(vals):.2f}",
                })
                break

        # Date range
        date_vals = []
        for date_col in ("start_date", "end_date", "ts_in_str", "date"):
            if date_col in first and first[date_col]:
                date_vals = [str(r[date_col])[:10] for r in data if r.get(date_col)]
                if date_vals:
                    items.append({"label": "Date Range", "value": f"{min(date_vals)} → {max(date_vals)}"})
                    break

        return items

    def _build_summary_band(self, items: List[Dict], styles: dict) -> list:
        """Render summary items as a horizontal pill-card row."""
        elements = []
        elements.append(Paragraph("Summary", styles["section_title"]))
        elements.append(Spacer(1, 0.2 * cm))

        # Chunk into rows of up to 3 cards
        chunk_size = 3
        for i in range(0, len(items), chunk_size):
            chunk  = items[i: i + chunk_size]
            # Pad to chunk_size so the table always has equal columns
            while len(chunk) < chunk_size:
                chunk.append({"label": "", "value": ""})

            cell_data = []
            for item in chunk:
                cell_data.append(
                    Paragraph(
                        f'<font color="#64748b" size="8">{item["label"]}</font>'
                        f'<br/><font color="#1e293b" size="13"><b>{item["value"]}</b></font>',
                        styles["card_cell"],
                    )
                )

            row_table = Table([cell_data], colWidths=["33%", "33%", "34%"])
            row_table.setStyle(TableStyle([
                ("BACKGROUND",   (0, 0), (-1, -1), ROW_ALT),
                ("BOX",          (0, 0), (-1, -1), 0.5, BORDER_COLOR),
                ("INNERGRID",    (0, 0), (-1, -1), 0.5, BORDER_COLOR),
                ("LEFTPADDING",  (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING",   (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING",(0, 0), (-1, -1), 8),
                ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
            ]))
            elements.append(row_table)
            elements.append(Spacer(1, 0.15 * cm))

        return elements

    # ── Data table ────────────────────────────────────────────────────────────

    def _build_data_table(self, data: List[Dict[str, Any]], styles: dict) -> list:
        """Build a paginated, styled data table."""
        if not data:
            return [Paragraph("No records to display.", styles["body"])]

        # Decide which columns to show (cap at 8 to avoid overflow)
        all_cols    = list(data[0].keys())
        skip_prefix = ("id",)
        cols        = [c for c in all_cols if not c.lower().startswith(skip_prefix)][:8]

        # Header row
        header = [Paragraph(c.replace("_", " ").title(), styles["th"]) for c in cols]

        # Compute column widths – equal distribution across usable page width
        page_w      = A4[0] - 3 * cm           # account for margins
        col_w       = page_w / len(cols)
        col_widths  = [col_w] * len(cols)

        # Build rows in chunks to keep table from breaking badly
        CHUNK = 40
        elements = []

        for chunk_start in range(0, len(data), CHUNK):
            chunk_data = data[chunk_start: chunk_start + CHUNK]
            table_rows = [header]

            for idx, row in enumerate(chunk_data):
                cells = []
                for col in cols:
                    val = row.get(col, "")
                    if val is None:
                        val = ""
                    # Truncate long strings
                    text = str(val)
                    if len(text) > 40:
                        text = text[:37] + "…"
                    cells.append(Paragraph(text, styles["td"]))
                table_rows.append(cells)

            tbl = Table(table_rows, colWidths=col_widths, repeatRows=1)

            # Alternating row styling
            style_cmds = [
                ("BACKGROUND",   (0, 0), (-1, 0),  HEADER_BG),
                ("TEXTCOLOR",    (0, 0), (-1, 0),  HEADER_FG),
                ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
                ("FONTSIZE",     (0, 0), (-1, 0),  8),
                ("BOTTOMPADDING",(0, 0), (-1, 0),  6),
                ("TOPPADDING",   (0, 0), (-1, 0),  6),
                ("GRID",         (0, 0), (-1, -1), 0.4, BORDER_COLOR),
                ("FONTSIZE",     (0, 1), (-1, -1), 7),
                ("TOPPADDING",   (0, 1), (-1, -1), 4),
                ("BOTTOMPADDING",(0, 1), (-1, -1), 4),
                ("LEFTPADDING",  (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
            ]
            for row_idx in range(1, len(table_rows)):
                bg = ROW_ALT if row_idx % 2 == 0 else ROW_NORMAL
                style_cmds.append(("BACKGROUND", (0, row_idx), (-1, row_idx), bg))

            tbl.setStyle(TableStyle(style_cmds))
            elements.append(KeepTogether([tbl]))
            if chunk_start + CHUNK < len(data):
                elements.append(Spacer(1, 0.3 * cm))

        return elements

    # ── Page footer callback ──────────────────────────────────────────────────

    def _page_footer(self, canvas, doc):
        """Draw page number footer on every page."""
        canvas.saveState()
        page_w, page_h = A4
        footer_y = 1.2 * cm

        # Left: branding
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(TEXT_MUTED)
        canvas.drawString(1.5 * cm, footer_y, "Fleet Management Assistant  •  Confidential")

        # Right: page number
        page_text = f"Page {doc.page}"
        canvas.drawRightString(page_w - 1.5 * cm, footer_y, page_text)

        # Separator line
        canvas.setStrokeColor(BORDER_COLOR)
        canvas.setLineWidth(0.5)
        canvas.line(1.5 * cm, footer_y + 0.4 * cm, page_w - 1.5 * cm, footer_y + 0.4 * cm)
        canvas.restoreState()

    # ── Style definitions ─────────────────────────────────────────────────────

    def _make_styles(self) -> dict:
        base   = getSampleStyleSheet()
        styles = {}

        styles["brand"] = ParagraphStyle(
            "brand",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=13,
            textColor=HEADER_FG,
        )
        styles["date_right"] = ParagraphStyle(
            "date_right",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8,
            textColor=HEADER_FG,
            alignment=TA_RIGHT,
        )
        styles["report_title"] = ParagraphStyle(
            "report_title",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=18,
            textColor=ACCENT,
            spaceAfter=4,
        )
        styles["query_label"] = ParagraphStyle(
            "query_label",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            textColor=TEXT_MUTED,
        )
        styles["section_title"] = ParagraphStyle(
            "section_title",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=PRIMARY,
            spaceBefore=6,
        )
        styles["card_cell"] = ParagraphStyle(
            "card_cell",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            alignment=TA_LEFT,
        )
        styles["th"] = ParagraphStyle(
            "th",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            textColor=HEADER_FG,
            alignment=TA_CENTER,
        )
        styles["td"] = ParagraphStyle(
            "td",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7,
            textColor=ACCENT,
        )
        styles["body"] = ParagraphStyle(
            "body",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            textColor=ACCENT,
        )

        return styles
