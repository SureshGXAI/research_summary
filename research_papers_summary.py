import warnings
warnings.filterwarnings('ignore')

import io
import os
import re
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    HRFlowable, Table, TableStyle, Image as RLImage,
)
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ── LLM setup ──────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0,
)
parser = StrOutputParser()

# Chart colour palette
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
           "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
sns.set_theme(style="whitegrid", palette=PALETTE)


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(path: str) -> pd.DataFrame:
    """
    Load a CSV, TSV, Excel (.xlsx/.xls), or JSON file into a DataFrame.
    Raises ValueError for unsupported formats.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    suffix = p.suffix.lower()
    loaders = {
        ".csv":  lambda: pd.read_csv(path),
        ".tsv":  lambda: pd.read_csv(path, sep="\t"),
        ".xlsx": lambda: pd.read_excel(path),
        ".xls":  lambda: pd.read_excel(path),
        ".json": lambda: pd.read_json(path),
    }
    if suffix not in loaders:
        raise ValueError(f"Unsupported file type '{suffix}'. Supported: CSV, TSV, XLSX, XLS, JSON.")

    df = loaders[suffix]()
    print(f"  ✓ Loaded dataset: {p.name}  ({len(df):,} rows × {len(df.columns)} columns)")
    return df


def dataset_summary(df: pd.DataFrame) -> str:
    """
    Build a compact text summary of a DataFrame for the LLM prompt:
    shape, dtypes, descriptive stats, and the first 5 rows.
    """
    buf = io.StringIO()
    buf.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n\n")
    buf.write("Column dtypes:\n")
    buf.write(df.dtypes.to_string())
    buf.write("\n\nDescriptive statistics:\n")
    buf.write(df.describe(include="all").to_string())
    buf.write("\n\nFirst 5 rows:\n")
    buf.write(df.head().to_string())

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        buf.write("\n\nMissing values:\n")
        buf.write(missing.to_string())

    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
#  CHART GENERATION  (LLM decides which charts to make, then we render them)
# ══════════════════════════════════════════════════════════════════════════════

CHART_PLANNER_SYSTEM = """You are a data visualization expert.
Given a dataset summary and a research topic, decide which charts best communicate the data.
Respond ONLY with a valid JSON array — no preamble, no markdown fences.
Each element has:
  "chart_type"  : one of [bar, line, scatter, histogram, box, heatmap, pie]
  "title"       : concise chart title
  "x_col"       : exact column name for x-axis (null for histogram/heatmap/pie)
  "y_col"       : exact column name for y-axis (null for histogram/heatmap/pie)
  "hue_col"     : exact column name for colour grouping, or null
  "cols"        : list of column names (used for heatmap/pie only), or null
  "description" : 1-sentence insight this chart reveals
Suggest 3–5 charts only. Use exact column names from the summary."""

CHART_PLANNER_HUMAN = """\
Research topic: {research_topic}

Dataset summary:
{dataset_summary}

Return the JSON array of chart specifications now."""


def plan_charts(research_topic: str, df: pd.DataFrame) -> list[dict]:
    """Ask the LLM to propose chart specs; return a list of dicts."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", CHART_PLANNER_SYSTEM),
        ("human",  CHART_PLANNER_HUMAN),
    ])
    chain  = prompt | llm | parser
    raw    = chain.invoke({
        "research_topic":  research_topic,
        "dataset_summary": dataset_summary(df),
    })
    # strip markdown code fences if present
    raw = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        specs = json.loads(raw)
    except json.JSONDecodeError:
        print("  ⚠  Chart planner returned invalid JSON — skipping chart generation.")
        specs = []
    return specs


def render_charts(specs: list[dict], df: pd.DataFrame,
                  out_dir: str = ".") -> list[dict]:
    """
    Render each chart spec to a PNG file.
    Returns a list of dicts: {path, title, description}.
    """
    rendered = []
    os.makedirs(out_dir, exist_ok=True)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    for i, spec in enumerate(specs):
        chart_type  = spec.get("chart_type", "").lower()
        title       = spec.get("title", f"Chart {i+1}")
        x_col       = spec.get("x_col")
        y_col       = spec.get("y_col")
        hue_col     = spec.get("hue_col")
        cols        = spec.get("cols") or []
        description = spec.get("description", "")

        # Validate columns exist
        all_cols = df.columns.tolist()
        def safe(col):
            return col if (col and col in all_cols) else None

        x_col   = safe(x_col)
        y_col   = safe(y_col)
        hue_col = safe(hue_col)
        cols    = [c for c in cols if c in all_cols]

        try:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            fig.patch.set_facecolor("white")

            if chart_type == "bar":
                if x_col and y_col:
                    plot_df = df[[x_col, y_col] + ([hue_col] if hue_col else [])].dropna()
                    if hue_col:
                        sns.barplot(data=plot_df, x=x_col, y=y_col, hue=hue_col, ax=ax)
                    else:
                        agg = plot_df.groupby(x_col)[y_col].mean().reset_index()
                        sns.barplot(data=agg, x=x_col, y=y_col, ax=ax, color=PALETTE[0])
                    ax.tick_params(axis='x', rotation=30)

            elif chart_type == "line":
                if x_col and y_col:
                    plot_df = df[[x_col, y_col] + ([hue_col] if hue_col else [])].dropna()
                    if hue_col:
                        sns.lineplot(data=plot_df, x=x_col, y=y_col, hue=hue_col, ax=ax)
                    else:
                        sns.lineplot(data=plot_df, x=x_col, y=y_col, ax=ax, color=PALETTE[0])

            elif chart_type == "scatter":
                if x_col and y_col:
                    plot_df = df[[x_col, y_col] + ([hue_col] if hue_col else [])].dropna()
                    sns.scatterplot(data=plot_df, x=x_col, y=y_col,
                                   hue=hue_col if hue_col else None, ax=ax, alpha=0.7)

            elif chart_type == "histogram":
                col = x_col or y_col or (numeric_cols[0] if numeric_cols else None)
                if col:
                    sns.histplot(df[col].dropna(), kde=True, ax=ax, color=PALETTE[0])
                    ax.set_xlabel(col)

            elif chart_type == "box":
                if x_col and y_col:
                    plot_df = df[[x_col, y_col]].dropna()
                    sns.boxplot(data=plot_df, x=x_col, y=y_col,
                               palette=PALETTE[:len(df[x_col].unique())], ax=ax)
                    ax.tick_params(axis='x', rotation=30)
                elif y_col:
                    sns.boxplot(y=df[y_col].dropna(), ax=ax, color=PALETTE[0])

            elif chart_type == "heatmap":
                heat_cols = cols if cols else numeric_cols[:10]
                if len(heat_cols) >= 2:
                    corr = df[heat_cols].corr()
                    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                                linewidths=0.5, ax=ax)

            elif chart_type == "pie":
                pie_col = (cols[0] if cols else None) or x_col
                if pie_col:
                    counts = df[pie_col].value_counts().head(8)
                    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                           colors=PALETTE[:len(counts)], startangle=140)
                    ax.set_aspect("equal")

            else:
                plt.close(fig)
                continue

            ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
            plt.tight_layout()

            out_path = os.path.join(out_dir, f"chart_{i+1:02d}_{chart_type}.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            rendered.append({"path": out_path, "title": title, "description": description})
            print(f"  ✓ Chart saved → {out_path}")

        except Exception as e:
            print(f"  ⚠  Skipped chart '{title}': {e}")
            plt.close()

    return rendered


# ══════════════════════════════════════════════════════════════════════════════
#  LLM CHAINS
# ══════════════════════════════════════════════════════════════════════════════

def make_chain(system_prompt: str, human_template: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human",  human_template),
    ])
    return prompt | llm | parser


LITERATURE_REVIEW_SYSTEM = (
    "You are an expert researcher skilled at navigating academic databases. "
    "Your role is to find and summarize high-quality, peer-reviewed papers "
    "relevant to the research topic, ensuring a comprehensive literature review."
)

DATA_ANALYST_SYSTEM = (
    "You are a data scientist proficient in statistical analysis and data visualization. "
    "You analyze provided datasets or simulated data to uncover insights and trends "
    "relevant to the research topic, ensuring robust and reproducible results."
)

WRITER_SYSTEM = (
    "You are an academic writer experienced in crafting research papers. "
    "You use inputs from the Literature Review and Data Analyst agents to write "
    "clear, concise, and well-structured paper sections adhering to academic standards."
)

CITATION_SYSTEM = (
    "You are a meticulous librarian specializing in citation management. "
    "You format references from the literature review and ensure all citations "
    "comply with the specified citation style (e.g., APA, MLA)."
)

PEER_REVIEW_SYSTEM = (
    "You are an academic reviewer with a critical eye for detail. "
    "You evaluate the draft research paper for logical consistency, methodological rigor, "
    "and completeness, providing constructive feedback to improve the manuscript."
)

LITERATURE_REVIEW_TEMPLATE = """\
Perform a literature review on the topic: {research_topic}

1. Search academic databases (e.g., Google Scholar, PubMed) for peer-reviewed papers on {research_topic}.
2. Select 5 high-quality, relevant papers published within the last 5 years.
3. Summarize each paper, highlighting key findings, methodologies, and gaps.
4. Provide a synthesis of the literature to guide the research paper.

Return a detailed literature review in **markdown format**, including summaries of 5 papers and a synthesis section.
"""

DATA_ANALYSIS_TEMPLATE = """\
Perform a data analysis on the topic: {research_topic}

{dataset_context}

1. Analyze the dataset (or use simulated data if none is provided).
2. Perform statistical analysis (descriptive statistics, correlations, trends).
3. Describe the charts that were generated (listed below) and interpret their insights.
4. Summarize results in a clear academic format.

Charts generated:
{chart_descriptions}

Return a data analysis report in **markdown format**, including statistical insights and chart interpretations.
"""

WRITING_TEMPLATE = """\
Draft a research paper on: {research_topic}

--- LITERATURE REVIEW ---
{literature_review}

--- DATA ANALYSIS ---
{data_analysis}
--- END INPUTS ---

1. Write Introduction, Literature Review Summary, Methodology, Results, and Discussion.
2. Reference the charts by their titles where appropriate.
3. Ensure clarity, academic tone, and scholarly standards.

Return the draft in **markdown format**.
"""

CITATION_TEMPLATE = """\
Format references for a research paper on {research_topic} using {citation_style} style.

--- DRAFT PAPER ---
{draft_paper}
--- END DRAFT ---

1. Collect all references cited in the literature review and draft.
2. Format according to {citation_style} style.
3. Ensure in-text citations and the reference list are complete.

Return a formatted reference list in **markdown format**.
"""

PEER_REVIEW_TEMPLATE = """\
Peer review the following research paper on {research_topic}.

--- DRAFT PAPER ---
{draft_paper}

--- REFERENCES ---
{references}
--- END INPUTS ---

1. Check logical consistency, methodological rigor, and completeness.
2. Identify gaps, unclear arguments, or areas needing evidence.
3. Provide constructive feedback and improvement suggestions.

Return a peer review report in **markdown format**.
"""

literature_review_chain = make_chain(LITERATURE_REVIEW_SYSTEM, LITERATURE_REVIEW_TEMPLATE)
data_analysis_chain     = make_chain(DATA_ANALYST_SYSTEM,      DATA_ANALYSIS_TEMPLATE)
writing_chain           = make_chain(WRITER_SYSTEM,            WRITING_TEMPLATE)
citation_chain          = make_chain(CITATION_SYSTEM,          CITATION_TEMPLATE)
peer_review_chain       = make_chain(PEER_REVIEW_SYSTEM,       PEER_REVIEW_TEMPLATE)


# ══════════════════════════════════════════════════════════════════════════════
#  PDF EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def save_results_to_pdf(results: dict, research_topic: str,
                        citation_style: str, charts: list[dict],
                        output_path: str = "research_output.pdf"):
    doc = SimpleDocTemplate(
        output_path, pagesize=letter,
        leftMargin=1*inch, rightMargin=1*inch,
        topMargin=1*inch, bottomMargin=1*inch,
    )

    S = getSampleStyleSheet()

    cover_title = ParagraphStyle("CoverTitle", parent=S["Title"],
        fontSize=26, leading=32, textColor=colors.HexColor("#1a1a2e"), spaceAfter=10)
    cover_sub = ParagraphStyle("CoverSub", parent=S["Normal"],
        fontSize=13, textColor=colors.HexColor("#444466"), spaceAfter=6)
    sec_head = ParagraphStyle("SecHead", parent=S["Heading1"],
        fontSize=16, leading=20, textColor=colors.HexColor("#1a1a2e"),
        spaceBefore=18, spaceAfter=6)
    sub_head = ParagraphStyle("SubHead", parent=S["Heading2"],
        fontSize=13, textColor=colors.HexColor("#2e4057"), spaceBefore=12, spaceAfter=4)
    body = ParagraphStyle("Body", parent=S["Normal"],
        fontSize=10, leading=15, textColor=colors.HexColor("#222222"), spaceAfter=6)
    bullet = ParagraphStyle("Bullet", parent=body, leftIndent=18, bulletIndent=6, spaceAfter=4)
    caption_style = ParagraphStyle("Caption", parent=body,
        fontSize=9, textColor=colors.HexColor("#555577"),
        alignment=1, spaceAfter=10, spaceBefore=2)

    def inline(text):
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'\*(.+?)\*',     r'<i>\1</i>', text)
        text = re.sub(r'`(.+?)`',       r'<font name="Courier">\1</font>', text)
        return text

    def md_to_flowables(text):
        flowables = []
        for line in text.splitlines():
            s = line.strip()
            if not s:
                flowables.append(Spacer(1, 6))
            elif s.startswith("### "):
                flowables.append(Paragraph(inline(s[4:]), sub_head))
            elif s.startswith("## "):
                flowables.append(Paragraph(inline(s[3:]), sec_head))
            elif s.startswith("# "):
                flowables.append(Paragraph(inline(s[2:]), sec_head))
            elif s.startswith(("- ", "* ")):
                flowables.append(Paragraph(f"• {inline(s[2:])}", bullet))
            elif re.match(r'^\d+\.\s', s):
                flowables.append(Paragraph(f"&nbsp;&nbsp;{inline(re.sub(r'^\d+\.\s','',s))}", bullet))
            elif s.startswith("---"):
                flowables.append(HRFlowable(width="100%", thickness=0.5,
                                            color=colors.HexColor("#ccccdd")))
                flowables.append(Spacer(1, 4))
            else:
                flowables.append(Paragraph(inline(s), body))
        return flowables

    story = []

    # ── Cover ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1.2*inch))
    story.append(Paragraph("AI Research Pipeline", cover_sub))
    story.append(Paragraph(research_topic, cover_title))
    story.append(Spacer(1, 0.15*inch))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1a1a2e")))
    story.append(Spacer(1, 0.15*inch))

    meta = [
        ["Citation Style", citation_style],
        ["Generated",      datetime.now().strftime("%B %d, %Y  %H:%M")],
        ["Model",          "llama-3.3-70b-versatile (Groq)"],
        ["Charts",         str(len(charts))],
    ]
    mt = Table(meta, colWidths=[1.5*inch, 4*inch])
    mt.setStyle(TableStyle([
        ("FONTSIZE",      (0,0),(-1,-1), 10),
        ("TEXTCOLOR",     (0,0),(0,-1),  colors.HexColor("#444466")),
        ("FONTNAME",      (0,0),(0,-1),  "Helvetica-Bold"),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
    ]))
    story.append(mt)
    story.append(PageBreak())

    # ── Charts section ────────────────────────────────────────────────────────
    if charts:
        story.append(Paragraph("0. Generated Charts", sec_head))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#aaaacc")))
        story.append(Spacer(1, 8))
        for ch in charts:
            if os.path.exists(ch["path"]):
                img = RLImage(ch["path"], width=5.5*inch, height=3.1*inch)
                story.append(img)
                story.append(Paragraph(f"<b>{ch['title']}</b> — {ch['description']}", caption_style))
                story.append(Spacer(1, 12))
        story.append(PageBreak())

    # ── Text sections ─────────────────────────────────────────────────────────
    sections = [
        ("1. Literature Review",  results["literature_review"]),
        ("2. Data Analysis",      results["data_analysis"]),
        ("3. Draft Paper",        results["draft_paper"]),
        ("4. References",         results["references"]),
        ("5. Peer Review Report", results["peer_review"]),
    ]
    for i, (title, content) in enumerate(sections):
        story.append(Paragraph(title, sec_head))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#aaaacc")))
        story.append(Spacer(1, 8))
        story.extend(md_to_flowables(content))
        if i < len(sections) - 1:
            story.append(PageBreak())

    doc.build(story)
    print(f"\n  ✓ PDF saved → {output_path}")
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_research_pipeline(
    research_topic: str,
    citation_style: str = "APA",
    dataset_path:   Optional[str] = None,
    charts_dir:     str = "charts",
    output_pdf:     str = "research_output.pdf",
) -> dict:
    """
    Run the full five-agent research pipeline.

    Args:
        research_topic : Topic to research.
        citation_style : "APA", "MLA", "IEEE", etc.
        dataset_path   : Optional path to a CSV / TSV / XLSX / JSON file.
                         If provided, real data is analyzed and charts are generated.
        charts_dir     : Directory to save chart PNG files.
        output_pdf     : Output PDF filename.
    """
    print("\n" + "="*60)
    print(f"  Research Topic : {research_topic}")
    print(f"  Citation Style : {citation_style}")
    print(f"  Dataset        : {dataset_path or 'None (simulated)'}")
    print("="*60)

    # ── Optional: load dataset & generate charts ───────────────────────────
    df = None
    charts: list[dict] = []
    dataset_context    = "No dataset was provided. Use simulated or hypothetical data relevant to the topic."
    chart_descriptions = "No charts were generated."

    if dataset_path:
        print("\n[Dataset] Loading and profiling dataset...")
        df = load_dataset(dataset_path)

        print("[Dataset] Planning charts with LLM...")
        specs = plan_charts(research_topic, df)

        if specs:
            print(f"[Dataset] Rendering {len(specs)} chart(s)...")
            charts = render_charts(specs, df, out_dir=charts_dir)

        dataset_context = (
            "A real dataset has been provided. Here is its summary:\n\n"
            + dataset_summary(df)
        )
        if charts:
            chart_descriptions = "\n".join(
                f"- **{c['title']}**: {c['description']}" for c in charts
            )

    # ── Step 1: Literature Review ──────────────────────────────────────────
    print("\n[1/5] Running Literature Review Agent...")
    literature_review = literature_review_chain.invoke({
        "research_topic": research_topic,
    })
    print("  ✓ Literature review complete.")

    # ── Step 2: Data Analysis ──────────────────────────────────────────────
    print("\n[2/5] Running Data Analyst Agent...")
    data_analysis = data_analysis_chain.invoke({
        "research_topic":   research_topic,
        "dataset_context":  dataset_context,
        "chart_descriptions": chart_descriptions,
    })
    print("  ✓ Data analysis complete.")

    # ── Step 3: Writing ────────────────────────────────────────────────────
    print("\n[3/5] Running Writer Agent...")
    draft_paper = writing_chain.invoke({
        "research_topic":   research_topic,
        "literature_review": literature_review,
        "data_analysis":    data_analysis,
    })
    print("  ✓ Draft paper complete.")

    # ── Step 4: Citations ──────────────────────────────────────────────────
    print("\n[4/5] Running Citation Agent...")
    references = citation_chain.invoke({
        "research_topic": research_topic,
        "citation_style": citation_style,
        "draft_paper":    draft_paper,
    })
    print("  ✓ References formatted.")

    # ── Step 5: Peer Review ────────────────────────────────────────────────
    print("\n[5/5] Running Peer Review Agent...")
    peer_review = peer_review_chain.invoke({
        "research_topic": research_topic,
        "draft_paper":    draft_paper,
        "references":     references,
    })
    print("  ✓ Peer review complete.")

    results = {
        "literature_review": literature_review,
        "data_analysis":     data_analysis,
        "draft_paper":       draft_paper,
        "references":        references,
        "peer_review":       peer_review,
    }

    # ── Export PDF ─────────────────────────────────────────────────────────
    print("\n[PDF] Building output document...")
    save_results_to_pdf(
        results=results,
        research_topic=research_topic,
        citation_style=citation_style,
        charts=charts,
        output_path=output_pdf,
    )

    print("\n" + "="*60)
    print("  Pipeline finished successfully.")
    print("="*60 + "\n")
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── Example A: no dataset (LLM uses simulated data) ───────────────────
    run_research_pipeline(
        research_topic="Machine Learning in Healthcare",
        citation_style="APA",
        output_pdf="research_output_no_dataset.pdf",
    )

    # ── Example B: with a custom dataset ──────────────────────────────────
    # Uncomment and set the path to your file (CSV / TSV / XLSX / JSON):
    #
    # run_research_pipeline(
    #     research_topic="Machine Learning in Healthcare",
    #     citation_style="APA",
    #     dataset_path="data/healthcare_ml.csv",
    #     charts_dir="charts",
    #     output_pdf="research_output_with_data.pdf",
    # )
