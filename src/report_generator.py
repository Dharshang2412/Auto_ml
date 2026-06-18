"""
Visualization Report Generator
-------------------------------
Builds a single polished PDF that pulls together everything the AutoML
dashboard has already computed in the current session: EDA charts, the
model leaderboard, feature importance, the confusion matrix (classification
only), the Trust Auditor agent's verdict (if it was run), and the AI result
summary -- all rendered as real embedded charts, not just text.

Design:
  - Charts are rendered with matplotlib directly to PNG bytes in-process.
    Deliberately NOT using Plotly's fig.to_image()/kaleido here: kaleido
    (especially the v1.x engine, which manages its own headless Chrome)
    has a long history of hanging indefinitely on Windows with no error
    -- see https://github.com/plotly/Kaleido/issues (300, 110, 126, 134,
    402, among others). matplotlib draws in-process with no subprocess or
    browser dependency, so it can't hang the same way.
  - Each section is wrapped defensively: if a given session_state key is
    missing (e.g. the user never ran the Trust Auditor, or trained a
    regression model with no confusion matrix), that section is skipped
    rather than crashing the whole report.
  - Returns the output path so main.py can offer it as a download.
"""

import io
import os
import tempfile

# Belt-and-suspenders: force the Agg backend via env var BEFORE matplotlib
# is imported at all. matplotlib.use("Agg") below can silently no-op if
# something else (e.g. Streamlit's own internals, or another module
# imported earlier in the process) already touched matplotlib.pyplot and
# initialized a different (GUI) backend first -- which then hangs forever
# in a headless/server context with no error message.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak,
    HRFlowable,
)

_ACCENT = "#2563eb"
_DARK = "#374151"
_GRID = "#d1d5db"


def _fig_to_image_flowable(fig, width_inches=6.5):
    """Renders a matplotlib figure to PNG bytes and wraps it as a reportlab
    Image flowable, scaled to fit the page width while preserving aspect
    ratio. Closes the figure afterward to free memory."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    fig_width_px, fig_height_px = fig.get_size_inches()
    aspect = fig_height_px / fig_width_px if fig_width_px else 0.6

    img = Image(buf, width=width_inches * inch, height=width_inches * inch * aspect)
    return img


def _styles():
    base = getSampleStyleSheet()
    base.add(ParagraphStyle(
        name="SectionHeading",
        parent=base["Heading1"],
        fontSize=16,
        spaceAfter=10,
        textColor=colors.HexColor("#1f2937"),
    ))
    base.add(ParagraphStyle(
        name="SubHeading",
        parent=base["Heading2"],
        fontSize=12,
        spaceAfter=6,
        textColor=colors.HexColor("#374151"),
    ))
    base.add(ParagraphStyle(
        name="BodyTextCustom",
        parent=base["BodyText"],
        fontSize=10,
        leading=14,
    ))
    return base


def _build_dataframe_table(df_preview: pd.DataFrame, max_rows=8, max_cols=8):
    """Renders a small dataframe preview as a reportlab Table."""
    display_df = df_preview.iloc[:max_rows, :max_cols].copy()
    display_df = display_df.astype(str)

    data = [list(display_df.columns)] + display_df.values.tolist()
    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(_DARK)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor(_GRID)),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return table


def _histogram_figure(df: pd.DataFrame, col: str):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(df[col].dropna(), bins=30, color=_ACCENT, edgecolor="white")
    ax.set_title(f"Distribution of {col}", fontsize=12)
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def _correlation_heatmap_figure(df: pd.DataFrame, num_cols: list):
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(num_cols)))
    ax.set_yticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(num_cols, fontsize=8)
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation Heatmap", fontsize=12)
    fig.tight_layout()
    return fig


def _leaderboard_bar_figure(ranked_df: pd.DataFrame, metric_col: str):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(ranked_df["Model"], ranked_df[metric_col], color=_ACCENT)
    for bar, val in zip(bars, ranked_df[metric_col]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.2f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_title("Model Performance Comparison", fontsize=12)
    ax.set_ylabel(metric_col)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def _feature_importance_figure(importance_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    ordered = importance_df.iloc[::-1]
    ax.barh(ordered["Feature"], ordered["Importance"], color=_ACCENT)
    ax.set_title("Top Important Features", fontsize=12)
    ax.set_xlabel("Importance")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def _confusion_matrix_figure(cm: np.ndarray):
    fig, ax = plt.subplots(figsize=(5, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    n = cm.shape[0]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    max_val = cm.max() if cm.size else 1
    for i in range(n):
        for j in range(n):
            color = "white" if cm[i, j] > max_val / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Confusion Matrix", fontsize=12)
    fig.tight_layout()
    return fig


def generate_visual_report(
    df: pd.DataFrame,
    session_state: dict,
    output_path: str = None
) -> str:
    """Builds the full PDF report from the dataframe and current Streamlit
    session_state. Returns the path to the generated PDF.

    Expected (optional) session_state keys, all gracefully skipped if absent:
      - results (DataFrame): model leaderboard
      - best_model, best_score, is_classification
      - X_test, y_test
      - feature_names
      - audit_verdict (dict from the Trust Auditor agent)
    """
    if output_path is None:
        output_path = os.path.join(tempfile.gettempdir(), "AutoML_Visual_Report.pdf")

    styles = _styles()
    story = []

    # ---------------- Cover / Header ----------------
    story.append(Paragraph("AutoML Visual Report", styles["Title"]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        f"Dataset: {df.shape[0]} rows x {df.shape[1]} columns",
        styles["BodyTextCustom"]
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor(_GRID)))
    story.append(Spacer(1, 16))

    # ---------------- Section: Dataset Preview ----------------
    story.append(Paragraph("Dataset Preview", styles["SectionHeading"]))
    story.append(_build_dataframe_table(df.head(8)))
    story.append(Spacer(1, 16))

    # ---------------- Section: EDA ----------------
    story.append(Paragraph("Exploratory Data Analysis", styles["SectionHeading"]))

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        story.append(Paragraph(f"Distribution: {num_cols[0]}", styles["SubHeading"]))
        story.append(_fig_to_image_flowable(_histogram_figure(df, num_cols[0])))
        story.append(Spacer(1, 12))

    if len(num_cols) > 1:
        story.append(Paragraph("Correlation Heatmap", styles["SubHeading"]))
        story.append(_fig_to_image_flowable(_correlation_heatmap_figure(df, num_cols)))

    story.append(PageBreak())

    # ---------------- Section: Trust Auditor Verdict ----------------
    audit_verdict = session_state.get("audit_verdict")
    if audit_verdict:
        story.append(Paragraph("Data Quality & Trust Audit", styles["SectionHeading"]))

        verdict_label = audit_verdict.get("verdict", "unknown")
        verdict_text = {
            "safe_to_train": "Safe to Train",
            "train_with_caution": "Train with Caution",
            "fix_before_training": "Fix Before Training",
        }.get(verdict_label, verdict_label)

        story.append(Paragraph(f"<b>Verdict:</b> {verdict_text}", styles["BodyTextCustom"]))
        story.append(Spacer(1, 6))
        story.append(Paragraph(audit_verdict.get("summary", ""), styles["BodyTextCustom"]))
        story.append(Spacer(1, 10))

        ranked = audit_verdict.get("ranked_findings", [])
        if ranked:
            story.append(Paragraph("Findings", styles["SubHeading"]))
            risk_order = {"high": 0, "medium": 1, "low": 2}
            ranked_sorted = sorted(ranked, key=lambda f: risk_order.get(f.get("risk", "medium"), 1))

            table_data = [["Risk", "Column", "Issue", "Recommendation"]]
            for f in ranked_sorted:
                table_data.append([
                    f.get("risk", "").upper(),
                    f.get("column") or "Dataset-wide",
                    f.get("issue", ""),
                    f.get("recommendation", "")
                ])

            findings_table = Table(table_data, colWidths=[0.6*inch, 1.2*inch, 1.4*inch, 3.0*inch])
            findings_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(_DARK)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor(_GRID)),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
            ]))
            story.append(findings_table)
        else:
            story.append(Paragraph("No risk findings were detected.", styles["BodyTextCustom"]))

        story.append(PageBreak())

    # ---------------- Section: Model Leaderboard ----------------
    result_df = session_state.get("results")
    if result_df is not None and len(result_df) > 0:
        story.append(Paragraph("Model Leaderboard", styles["SectionHeading"]))

        ranked_df = result_df.sort_values(
            by=result_df.columns[-1], ascending=False
        ).reset_index(drop=True)
        ranked_df.insert(0, "Rank", range(1, len(ranked_df) + 1))

        lb_table_data = [list(ranked_df.columns)] + ranked_df.astype(str).values.tolist()
        lb_table = Table(lb_table_data, repeatRows=1)
        lb_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(_DARK)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor(_GRID)),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
        ]))
        story.append(lb_table)
        story.append(Spacer(1, 12))

        metric_col = ranked_df.columns[2] if len(ranked_df.columns) > 2 else ranked_df.columns[-1]
        story.append(_fig_to_image_flowable(_leaderboard_bar_figure(ranked_df, metric_col)))

        best_score = session_state.get("best_score")
        if best_score is not None:
            story.append(Spacer(1, 8))
            story.append(Paragraph(
                f"<b>Best Score:</b> {best_score * 100:.2f}%",
                styles["BodyTextCustom"]
            ))

        story.append(PageBreak())

    # ---------------- Section: Feature Importance ----------------
    best_model = session_state.get("best_model")
    feature_names = session_state.get("feature_names")
    if best_model is not None and feature_names is not None and hasattr(best_model, "feature_importances_"):
        story.append(Paragraph("Feature Importance", styles["SectionHeading"]))

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": best_model.feature_importances_
        }).sort_values(by="Importance", ascending=False).head(15)

        story.append(_fig_to_image_flowable(_feature_importance_figure(importance_df)))

        top_feature = importance_df.iloc[0]
        story.append(Spacer(1, 8))
        story.append(Paragraph(
            f"<b>Most Important Feature:</b> {top_feature['Feature']}",
            styles["BodyTextCustom"]
        ))
        story.append(PageBreak())

    # ---------------- Section: Confusion Matrix ----------------
    is_classification = session_state.get("is_classification")
    X_test = session_state.get("X_test")
    y_test = session_state.get("y_test")
    if is_classification and best_model is not None and X_test is not None and y_test is not None:
        from sklearn.metrics import confusion_matrix

        story.append(Paragraph("Confusion Matrix", styles["SectionHeading"]))
        y_pred = best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        story.append(_fig_to_image_flowable(_confusion_matrix_figure(cm), width_inches=4.5))
        story.append(PageBreak())

    # ---------------- Section: AI Result Summary ----------------
    if result_df is not None and len(result_df) > 0 and session_state.get("best_score") is not None:
        story.append(Paragraph("Result Summary", styles["SectionHeading"]))

        ranked_df2 = result_df.sort_values(by=result_df.columns[-1], ascending=False).reset_index(drop=True)
        best_model_name = ranked_df2.iloc[0]["Model"]
        best_score = session_state["best_score"]

        summary_text = (
            f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns. "
            f"Across all trained models, <b>{best_model_name}</b> achieved the highest "
            f"performance at <b>{best_score * 100:.2f}%</b>. "
        )

        audit = session_state.get("audit_verdict")
        if audit:
            verdict_label = audit.get("verdict", "")
            if verdict_label == "fix_before_training":
                summary_text += (
                    "Note: the Trust Auditor flagged data quality risks before training -- "
                    "review those findings, as they may affect how much this score can be trusted."
                )
            elif verdict_label == "train_with_caution":
                summary_text += (
                    "The Trust Auditor flagged some minor data quality concerns worth reviewing."
                )
            else:
                summary_text += "The Trust Auditor found no significant data quality concerns."

        story.append(Paragraph(summary_text, styles["BodyTextCustom"]))
        story.append(Spacer(1, 8))
        story.append(Paragraph(
            f"<b>Recommendation:</b> Deploy {best_model_name} for production use, "
            f"subject to validation on a held-out or live data sample.",
            styles["BodyTextCustom"]
        ))

    # ---------------- Build ----------------
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
    )
    doc.build(story)

    return output_path
