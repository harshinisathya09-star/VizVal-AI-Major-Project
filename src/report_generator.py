from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib import colors

from .utils import REPORTS_DIR, ensure_directories, format_inr


DISCLAIMER = (
    "This report is an AI/ML-based property valuation estimate and should not be "
    "considered a formal appraisal or guaranteed market price. Consult a qualified "
    "real estate professional for official valuation."
)


def generate_pdf_report(
    assessment_id: int | str,
    record: dict,
    result: dict,
    valuation_status: str,
    comparable_summary: dict,
    top_factors: pd.DataFrame,
) -> Path:
    ensure_directories()
    path = REPORTS_DIR / f"valuation_report_{assessment_id}.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Real Estate Property Valuation Report", styles["Title"]),
        Paragraph(f"Assessment ID: {assessment_id}", styles["Normal"]),
        Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]),
        Spacer(1, 14),
    ]

    property_rows = [["Field", "Value"]] + [[k.replace("_", " ").title(), str(v)] for k, v in record.items()]
    prediction_rows = [
        ["Metric", "Value"],
        ["ML Prediction", format_inr(result.get("ml_prediction"))],
        ["DL Prediction", format_inr(result.get("dl_prediction"))],
        ["Final Prediction", format_inr(result.get("final_prediction"))],
        ["Valuation Range", f"{format_inr(result.get('range_low'))} - {format_inr(result.get('range_high'))}"],
        ["Price / sq.ft", f"Rs {result.get('price_per_sqft', 0):,.0f}"],
        ["Market Signal", valuation_status],
        ["Comparable Average", format_inr(comparable_summary.get("average_price"))],
        ["Comparable Median", format_inr(comparable_summary.get("median_price"))],
    ]

    for title, rows in [("Property Details", property_rows), ("Valuation Summary", prediction_rows)]:
        story.append(Paragraph(title, styles["Heading2"]))
        table = Table(rows, colWidths=[170, 330])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.extend([table, Spacer(1, 12)])

    story.append(Paragraph("Top Model Factors", styles["Heading2"]))
    if not top_factors.empty:
        factor_rows = [["Feature", "Contribution"]] + [
            [str(row["feature"]), f"{row['contribution']:,.3f}"] for _, row in top_factors.head(8).iterrows()
        ]
        story.append(Table(factor_rows, colWidths=[300, 170]))
    story.extend([Spacer(1, 12), Paragraph("Disclaimer", styles["Heading2"]), Paragraph(DISCLAIMER, styles["Normal"])])
    doc.build(story)
    return path
