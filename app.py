import os
import uuid
import logging
import tempfile
from html import escape
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("datadetective")

APP_TITLE = "DataDetective AI"
MAX_FILE_SIZE_MB = 100
MAX_ROWS_SHOWN = 500

SOURCE_SAMPLE = "Sample dataset"
SOURCE_UPLOAD = "Upload a CSV"

STATUS = "Record status"
COLOR_MAP = {"Normal": "#2A7F9E", "Flagged": "#D6453D"}

# Download files live in one private folder that Gradio is allowed to serve.
OUTPUT_DIR = Path(tempfile.mkdtemp(prefix="datadetective_"))


# --------------------------------------------------
# Sample dataset
# --------------------------------------------------

def make_sample_data():
    rng = np.random.default_rng(42)
    rows = 250

    df = pd.DataFrame({
        "transaction_amount": rng.normal(120, 25, rows).round(2),
        "processing_time_ms": rng.normal(180, 35, rows).round(1),
        "items_count": rng.integers(1, 12, rows),
        "customer_rating": rng.uniform(1, 5, rows).round(1),
        "region": rng.choice(["Abu Dhabi", "Dubai", "Sharjah", "Al Ain"], rows),
        "status": rng.choice(
            ["Completed", "Pending", "Cancelled"], rows, p=[0.75, 0.15, 0.10]
        ),
    })

    # Deliberately unusual records for demonstration.
    df.loc[[12, 67, 143], "transaction_amount"] = [850, 920, 1100]
    df.loc[[12, 67, 143], "processing_time_ms"] = [900, 1100, 1500]

    # Missing values and duplicate rows.
    df.loc[[5, 19, 72], "customer_rating"] = np.nan
    df.loc[[31, 108], "region"] = None
    df = pd.concat([df, df.iloc[[8, 22]]], ignore_index=True)

    return df


# --------------------------------------------------
# Small HTML helpers
# --------------------------------------------------

HERO_HTML = """
<div class="dd-hero">
  <div>
    <h1>DataDetective AI</h1>
    <p>Load a CSV, check its quality, and find the records that don't
    fit the pattern.</p>
    <small>Files are processed by the server hosting this app. Don't
    upload confidential data to a public demo.</small>
  </div>
  <svg viewBox="0 0 100 100" role="img" aria-label="Magnifying glass over a cluster of data points, one highlighted">
    <circle cx="40" cy="40" r="26" fill="none" stroke="#8FD3E8" stroke-width="5"/>
    <line x1="59" y1="59" x2="88" y2="88" stroke="#8FD3E8" stroke-width="8" stroke-linecap="round"/>
    <circle cx="30" cy="38" r="3" fill="#8FD3E8"/>
    <circle cx="37" cy="29" r="3" fill="#8FD3E8"/>
    <circle cx="45" cy="41" r="3" fill="#8FD3E8"/>
    <circle cx="33" cy="49" r="3" fill="#8FD3E8"/>
    <circle cx="44" cy="51" r="3" fill="#8FD3E8"/>
    <circle cx="52" cy="31" r="4.5" fill="#FF6B5E"/>
  </svg>
</div>
"""

FOOTER_HTML = """
<div class="dd-footer">
  Statistical anomaly detection can flag normal records and miss real
  problems. Review flagged rows with domain knowledge, and keep your
  original data unchanged.
</div>
"""


def notice(kind, text):
    """kind: info, warn, error, ok. `text` may contain trusted markup."""
    return f'<div class="dd-notice dd-notice-{kind}">{text}</div>'


def kpi_cards(items):
    """items: list of (label, value, tone). Tones: neutral, warn, flag, good."""
    cards = "".join(
        f'<div class="dd-kpi dd-{tone}">'
        f'<div class="dd-kpi-value">{escape(str(value))}</div>'
        f'<div class="dd-kpi-label">{escape(label)}</div>'
        f"</div>"
        for label, value, tone in items
    )
    return f'<div class="dd-kpis">{cards}</div>'


def unique_name(name, existing):
    candidate, n = name, 2
    while candidate in existing:
        candidate = f"{name} ({n})"
        n += 1
    return candidate


# --------------------------------------------------
# File loading and validation
# --------------------------------------------------

def make_unique_columns(columns):
    used, result = set(), []

    for column in columns:
        base = str(column).strip() or "unnamed_column"
        name, n = base, 1

        while name in used:
            n += 1
            name = f"{base}_{n}"

        used.add(name)
        result.append(name)

    return result


def read_dataset(file_path, use_sample):
    if use_sample:
        return make_sample_data()

    if not file_path:
        raise ValueError("Upload a CSV file to begin.")

    path = str(getattr(file_path, "name", file_path))

    if not os.path.isfile(path):
        raise ValueError("The uploaded file could not be found. Upload it again.")

    if not path.lower().endswith(".csv"):
        raise ValueError("This file isn't a CSV. Upload a file ending in .csv.")

    size_mb = os.path.getsize(path) / (1024 * 1024)

    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"The file is over the {MAX_FILE_SIZE_MB} MB limit.")

    df = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            raise ValueError("The CSV file is empty.")
        except pd.errors.ParserError as exc:
            raise ValueError(f"The file couldn't be parsed as CSV: {exc}")

    if df is None or df.empty:
        raise ValueError("The CSV has no data rows.")

    df.columns = make_unique_columns(df.columns)
    return df


def load_data(source, file_path):
    use_sample = source == SOURCE_SAMPLE

    try:
        df = read_dataset(file_path, use_sample)
        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

        origin = "sample dataset" if use_sample else "your file"
        status = notice(
            "ok",
            f"Loaded the {origin}: <b>{len(df):,} rows</b> and "
            f"<b>{len(df.columns):,} columns</b>. "
            f"Open the <b>Investigate</b> tab to analyze it.",
        )

        return (
            df,
            status,
            df.head(20),
            gr.update(choices=numeric_columns, value=[]),
            gr.update(
                choices=numeric_columns,
                value=numeric_columns[0] if numeric_columns else None,
            ),
        )

    except Exception as exc:
        logger.warning("Dataset loading failed: %s", exc)
        kind = "info" if "Upload a CSV file to begin" in str(exc) else "error"
        return (
            None,
            notice(kind, escape(str(exc))),
            pd.DataFrame(),
            gr.update(choices=[], value=[]),
            gr.update(choices=[], value=None),
        )


def toggle_upload(source):
    return gr.update(visible=source == SOURCE_UPLOAD)


# --------------------------------------------------
# Data profiling
# --------------------------------------------------

def profile_data(df):
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    quality_rows = []

    for column in df.columns:
        series = df[column]
        missing = int(series.isna().sum())

        quality_rows.append({
            "Column": str(column),
            "Data type": str(series.dtype),
            "Missing values": missing,
            "Missing (%)": round(missing / len(df) * 100 if len(df) else 0, 2),
            "Unique values": int(series.nunique(dropna=True)),
        })

    quality = pd.DataFrame(quality_rows)

    numeric_df = df.select_dtypes(include=np.number)

    if not numeric_df.empty:
        statistics = numeric_df.describe().T.round(3)
        statistics.index.name = "Column"
        statistics = statistics.reset_index()
    else:
        statistics = pd.DataFrame({"Information": ["No numeric columns found."]})

    metrics = {
        "rows": len(df),
        "columns": len(df.columns),
        "numeric": len(numeric_df.columns),
        "missing": missing_cells,
        "missing_pct": missing_cells / total_cells * 100 if total_cells else 0,
        "duplicates": duplicate_rows,
    }

    return metrics, quality, statistics


# --------------------------------------------------
# Download files
# --------------------------------------------------

def make_download_file(df, prefix):
    path = OUTPUT_DIR / f"{prefix}_{uuid.uuid4().hex[:8]}.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    return str(path)


# --------------------------------------------------
# Charts
# --------------------------------------------------

def style_figure(fig):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Manrope, system-ui, sans-serif", color="#7C8AA0"),
        margin=dict(l=20, r=20, t=56, b=20),
        legend_title_text="",
        title_font_size=16,
    )
    fig.update_xaxes(gridcolor="rgba(124,138,160,0.25)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(124,138,160,0.25)", zeroline=False)
    return fig


def build_histogram(df, column, flagged):
    values = df[column].replace([np.inf, -np.inf], np.nan)
    plot_df = pd.DataFrame({column: values, STATUS: "Normal"})

    if flagged is not None:
        plot_df.loc[flagged, STATUS] = "Flagged"

    plot_df = plot_df.dropna(subset=[column])

    if plot_df.empty:
        return None

    fig = px.histogram(
        plot_df,
        x=column,
        color=STATUS,
        nbins=30,
        barmode="stack",
        color_discrete_map=COLOR_MAP,
        category_orders={STATUS: ["Normal", "Flagged"]},
        title=f"Distribution of {column}",
    )
    return style_figure(fig)


def build_scatter(model_data, flagged):
    columns = list(model_data.columns)
    x_col = columns[0]

    plot_df = pd.DataFrame({x_col: model_data[x_col].to_numpy()})

    if len(columns) > 1:
        y_col = columns[1]
        plot_df[y_col] = model_data[y_col].to_numpy()
    else:
        y_col = "Row number"
        plot_df[y_col] = np.arange(1, len(model_data) + 1)

    plot_df[STATUS] = np.where(flagged, "Flagged", "Normal")
    plot_df["Original row"] = np.arange(1, len(model_data) + 1)

    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        color=STATUS,
        color_discrete_map=COLOR_MAP,
        category_orders={STATUS: ["Normal", "Flagged"]},
        hover_data=["Original row"],
        opacity=0.8,
        title=f"{y_col} vs {x_col}",
    )
    fig.update_traces(marker=dict(size=8))
    return style_figure(fig)


def build_correlation(all_numeric):
    corr_data = all_numeric.iloc[:, :20]

    if corr_data.shape[1] < 2:
        return None

    corr = corr_data.corr()

    fig = px.imshow(
        corr,
        title="Correlation between numeric features",
        zmin=-1,
        zmax=1,
        color_continuous_scale=[[0, "#D6453D"], [0.5, "#F1F4F7"], [1, "#1E6F8C"]],
        text_auto=".2f" if corr.shape[0] <= 10 else False,
    )
    return style_figure(fig)


# --------------------------------------------------
# Anomaly detection
# --------------------------------------------------

def analyze_data(df, selected_features, contamination, hist_column):
    empty = pd.DataFrame()

    def failure(message, kind="warn"):
        return ("", notice(kind, message), empty, empty, empty, None, None, None, None)

    if df is None:
        return failure("Load a dataset first on the <b>Load data</b> tab.")

    try:
        metrics, quality, statistics = profile_data(df)

        all_numeric = df.select_dtypes(include=np.number).replace(
            [np.inf, -np.inf], np.nan
        )

        features = all_numeric
        if selected_features:
            chosen = [c for c in selected_features if c in all_numeric.columns]
            features = all_numeric[chosen]

        features = features.dropna(axis=1, how="all")
        if features.shape[1]:
            features = features.loc[:, features.nunique(dropna=True) > 1]

        anomalies = empty
        anomaly_path = None
        flagged = None
        model_data = None
        message = ""
        message_kind = "info"

        if len(df) < 10:
            message = "Anomaly detection was skipped: it needs at least 10 rows."
            message_kind = "warn"
        elif features.empty:
            message = (
                "Anomaly detection was skipped: no usable numeric features. "
                "Pick numeric columns that vary across rows."
            )
            message_kind = "warn"
        else:
            # Missing model inputs are filled with column medians.
            model_data = features.fillna(features.median()).fillna(0)

            model = IsolationForest(
                n_estimators=200,
                contamination=float(contamination),
                random_state=42,
                n_jobs=-1,
            )

            predictions = model.fit_predict(model_data)
            scores = model.decision_function(model_data)
            positions = np.flatnonzero(predictions == -1)

            flagged = np.zeros(len(df), dtype=bool)
            flagged[positions] = True

            anomalies = df.iloc[positions].copy()
            row_col = unique_name("Original row", anomalies.columns)
            score_col = unique_name("Anomaly score", anomalies.columns)
            anomalies.insert(0, row_col, positions + 1)
            anomalies[score_col] = np.round(scores[positions], 4)
            anomalies = anomalies.sort_values(score_col, ascending=True)

            anomaly_path = make_download_file(anomalies, "datadetective_anomalies")

            message = (
                f"Checked <b>{len(model_data.columns)}</b> numeric feature(s) and "
                f"flagged <b>{len(anomalies):,}</b> of {len(df):,} records "
                f"(expected proportion {float(contamination):.0%}). "
                "A flagged row is a lead to investigate, not proof of fraud "
                "or an error. Lower scores are more unusual."
            )

        if len(anomalies) > MAX_ROWS_SHOWN:
            message += (
                f" The table shows the {MAX_ROWS_SHOWN} most unusual rows; "
                "download the report for all of them."
            )
            shown = anomalies.head(MAX_ROWS_SHOWN)
        else:
            shown = anomalies

        flagged_count = 0 if flagged is None else int(flagged.sum())

        kpis = kpi_cards([
            ("Rows", f"{metrics['rows']:,}", "neutral"),
            ("Columns", f"{metrics['columns']:,}", "neutral"),
            (
                "Missing cells",
                f"{metrics['missing']:,} ({metrics['missing_pct']:.1f}%)",
                "warn" if metrics["missing"] else "good",
            ),
            (
                "Duplicate rows",
                f"{metrics['duplicates']:,}",
                "warn" if metrics["duplicates"] else "good",
            ),
            (
                "Flagged records",
                f"{flagged_count:,}",
                "flag" if flagged_count else "neutral",
            ),
        ])

        # Charts
        histogram = None
        column = hist_column if hist_column in all_numeric.columns else None
        if column is None and not all_numeric.empty:
            column = all_numeric.columns[0]
        if column is not None:
            histogram = build_histogram(df, column, flagged)

        scatter = (
            build_scatter(model_data, flagged) if model_data is not None else None
        )
        correlation = build_correlation(all_numeric)

        return (
            kpis,
            notice(message_kind, message),
            quality,
            statistics,
            shown,
            histogram,
            scatter,
            correlation,
            anomaly_path,
        )

    except Exception:
        logger.exception("Dataset analysis failed")
        return failure(
            "Analysis failed. Check the CSV and try again. "
            "The server logs have the details.",
            kind="error",
        )


# --------------------------------------------------
# Data cleaning
# --------------------------------------------------

STRATEGY_KEEP = "Keep missing values"
STRATEGY_FILL = "Fill numeric median and categorical mode"
STRATEGY_DROP = "Drop rows containing missing values"


def clean_data(df, missing_strategy, remove_duplicates):
    if df is None:
        return notice("warn", "Load a dataset first on the <b>Load data</b> tab."), None

    try:
        cleaned = df.copy()

        original_rows = len(cleaned)
        original_missing = int(cleaned.isna().sum().sum())
        original_duplicates = int(cleaned.duplicated().sum())

        if remove_duplicates:
            cleaned = cleaned.drop_duplicates()

        if missing_strategy == STRATEGY_FILL:
            for column in cleaned.columns:
                if not cleaned[column].isna().any():
                    continue

                if pd.api.types.is_numeric_dtype(cleaned[column]):
                    median = cleaned[column].median()
                    if pd.notna(median):
                        cleaned[column] = cleaned[column].fillna(median)
                else:
                    modes = cleaned[column].mode(dropna=True)
                    if not modes.empty:
                        cleaned[column] = cleaned[column].fillna(modes.iloc[0])

        elif missing_strategy == STRATEGY_DROP:
            cleaned = cleaned.dropna()

        path = make_download_file(cleaned, "datadetective_cleaned")
        missing_after = int(cleaned.isna().sum().sum())

        summary = kpi_cards([
            ("Rows before", f"{original_rows:,}", "neutral"),
            ("Rows after", f"{len(cleaned):,}", "good"),
            ("Duplicates found", f"{original_duplicates:,}", "warn" if original_duplicates else "neutral"),
            ("Missing cells before", f"{original_missing:,}", "warn" if original_missing else "neutral"),
            ("Missing cells after", f"{missing_after:,}", "good" if missing_after == 0 else "warn"),
        ])

        summary += notice(
            "info",
            "The cleaned file is ready to download. Review it before use: "
            "filling values can change what the data means. Your original "
            "upload is not modified.",
        )

        return summary, path

    except Exception:
        logger.exception("Data cleaning failed")
        return notice("error", "Cleaning failed. Check the data and try again."), None


# --------------------------------------------------
# Theme and styling
# --------------------------------------------------

THEME = gr.themes.Base(
    primary_hue="cyan",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_md,
    font=[gr.themes.GoogleFont("Manrope"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
).set(
    button_primary_background_fill="#1E5F7A",
    button_primary_background_fill_hover="#16495E",
    button_primary_background_fill_dark="#2A7F9E",
    button_primary_background_fill_hover_dark="#3590B1",
    button_primary_text_color="#FFFFFF",
    button_primary_text_color_dark="#FFFFFF",
)

CSS = """
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; }

.dd-hero {
  display: flex; align-items: center; justify-content: space-between; gap: 24px;
  padding: 28px 32px; border-radius: 14px;
  background: #16384D; color: #F2F6F8;
}
.dd-hero h1 {
  margin: 0 0 8px; font-size: 2.1rem; font-weight: 700;
  letter-spacing: -0.02em; color: #FFFFFF !important;
}
.dd-hero p { margin: 0; max-width: 52ch; color: #C3D3DC; line-height: 1.55; font-size: 1.05rem; }
.dd-hero small { display: block; margin-top: 14px; max-width: 60ch; color: #8FAABA; line-height: 1.5; }
.dd-hero svg { flex: none; width: 104px; height: 104px; }

.dd-kpis {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px; margin: 4px 0 12px;
}
.dd-kpi {
  padding: 14px 16px; border-radius: 10px;
  border: 1px solid var(--border-color-primary);
  border-left-width: 4px;
  background: var(--background-fill-secondary);
}
.dd-kpi-value {
  font-size: 1.5rem; font-weight: 700; line-height: 1.2;
  font-variant-numeric: tabular-nums; color: var(--body-text-color);
}
.dd-kpi-label { margin-top: 2px; font-size: 0.88rem; color: var(--body-text-color-subdued); }
.dd-neutral { border-left-color: #2A7F9E; }
.dd-good    { border-left-color: #2E9E6B; }
.dd-warn    { border-left-color: #E0A21B; }
.dd-flag    { border-left-color: #D6453D; }

.dd-notice {
  padding: 12px 16px; margin: 4px 0; border-radius: 10px; line-height: 1.55;
  border: 1px solid var(--border-color-primary);
  border-left-width: 4px;
  background: var(--background-fill-secondary);
  color: var(--body-text-color);
}
.dd-notice-info  { border-left-color: #2A7F9E; }
.dd-notice-ok    { border-left-color: #2E9E6B; }
.dd-notice-warn  { border-left-color: #E0A21B; }
.dd-notice-error { border-left-color: #D6453D; }

.dd-footer {
  margin-top: 8px; padding-top: 14px; font-size: 0.9rem; line-height: 1.5;
  border-top: 1px solid var(--border-color-primary);
  color: var(--body-text-color-subdued);
}

.dd-hint { color: var(--body-text-color-subdued); margin: 0 0 4px; }

@media (max-width: 640px) {
  .dd-hero { flex-direction: column-reverse; align-items: flex-start; padding: 22px; }
  .dd-hero svg { width: 72px; height: 72px; }
}
"""

# Gradio 6 moved theme/css from Blocks() to launch(); support both.
GRADIO_MAJOR = int(gr.__version__.split(".")[0])
blocks_kwargs = {"title": APP_TITLE}
launch_kwargs = {}

if GRADIO_MAJOR >= 6:
    launch_kwargs.update(theme=THEME, css=CSS)
else:
    blocks_kwargs.update(theme=THEME, css=CSS)


# --------------------------------------------------
# User interface
# --------------------------------------------------

with gr.Blocks(**blocks_kwargs) as demo:
    gr.HTML(HERO_HTML)

    dataset_state = gr.State(None)

    with gr.Tabs():

        # ---------------- Load data ----------------
        with gr.Tab("Load data"):
            gr.HTML('<p class="dd-hint">Choose where the data comes from. '
                    'It loads automatically.</p>')

            source = gr.Radio(
                choices=[SOURCE_SAMPLE, SOURCE_UPLOAD],
                value=SOURCE_SAMPLE,
                label="Data source",
            )

            csv_upload = gr.File(
                label="CSV file (up to 100 MB)",
                file_types=[".csv"],
                type="filepath",
                visible=False,
            )

            load_status = gr.HTML()

            preview = gr.Dataframe(
                label="Preview (first 20 rows)",
                interactive=False,
            )

        # ---------------- Investigate ----------------
        with gr.Tab("Investigate"):
            with gr.Row():
                with gr.Column(scale=3):
                    feature_selector = gr.Dropdown(
                        label="Numeric features to check",
                        choices=[],
                        multiselect=True,
                        value=[],
                        info="Leave empty to use every numeric column that varies.",
                    )
                with gr.Column(scale=2):
                    contamination = gr.Slider(
                        minimum=0.01,
                        maximum=0.20,
                        value=0.05,
                        step=0.01,
                        label="Expected share of unusual records",
                        info="A model setting, not a measured error rate.",
                    )
                with gr.Column(scale=2):
                    hist_column = gr.Dropdown(
                        label="Column for the distribution chart",
                        choices=[],
                        value=None,
                    )

            analyze_button = gr.Button("Analyze dataset", variant="primary")

            kpi_output = gr.HTML()
            report_output = gr.HTML()

            with gr.Tabs():
                with gr.Tab("Flagged records"):
                    anomaly_output = gr.Dataframe(
                        label="Most unusual records first",
                        interactive=False,
                    )
                    anomaly_download = gr.File(label="Download full anomaly report")

                with gr.Tab("Charts"):
                    histogram_output = gr.Plot(label="Distribution")
                    with gr.Row():
                        scatter_output = gr.Plot(label="Flagged records in context")
                        correlation_output = gr.Plot(label="Correlation")

                with gr.Tab("Column quality"):
                    quality_output = gr.Dataframe(
                        label="Missing values and unique counts per column",
                        interactive=False,
                    )

                with gr.Tab("Statistics"):
                    stats_output = gr.Dataframe(
                        label="Descriptive statistics for numeric columns",
                        interactive=False,
                    )

        # ---------------- Clean & export ----------------
        with gr.Tab("Clean and export"):
            gr.HTML('<p class="dd-hint">Create a cleaned copy of the data. '
                    'Your original file stays untouched.</p>')

            missing_strategy = gr.Radio(
                label="Missing values",
                choices=[STRATEGY_KEEP, STRATEGY_FILL, STRATEGY_DROP],
                value=STRATEGY_KEEP,
            )

            remove_duplicates = gr.Checkbox(
                label="Remove duplicate rows",
                value=True,
            )

            clean_button = gr.Button("Create cleaned CSV", variant="primary")
            clean_status = gr.HTML()
            clean_download = gr.File(label="Download cleaned dataset")

    gr.HTML(FOOTER_HTML)

    # ---------------- Events ----------------
    load_outputs = [
        dataset_state,
        load_status,
        preview,
        feature_selector,
        hist_column,
    ]

    demo.load(load_data, inputs=[source, csv_upload], outputs=load_outputs)

    source.change(
        toggle_upload, inputs=source, outputs=csv_upload
    ).then(load_data, inputs=[source, csv_upload], outputs=load_outputs)

    csv_upload.change(load_data, inputs=[source, csv_upload], outputs=load_outputs)

    analyze_button.click(
        fn=analyze_data,
        inputs=[dataset_state, feature_selector, contamination, hist_column],
        outputs=[
            kpi_output,
            report_output,
            quality_output,
            stats_output,
            anomaly_output,
            histogram_output,
            scatter_output,
            correlation_output,
            anomaly_download,
        ],
    )

    clean_button.click(
        fn=clean_data,
        inputs=[dataset_state, missing_strategy, remove_duplicates],
        outputs=[clean_status, clean_download],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "10000")),
        share=False,
        allowed_paths=[str(OUTPUT_DIR)],
        **launch_kwargs,
    )