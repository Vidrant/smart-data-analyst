import io
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Smart Data Analyst", page_icon="📊", layout="wide")

st.title("📊 Smart Data Analyst")
st.caption("Adaptive CSV/Excel profiling, data quality checks, exploratory analysis, and anomaly detection.")

@st.cache_data(show_spinner=False)
def load_file(file_bytes, filename, sheet_name=None):
    name = filename.lower()
    if name.endswith(".csv"):
        # Try common encodings; let pandas infer the separator.
        last_error = None
        for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
            try:
                return pd.read_csv(
                    io.BytesIO(file_bytes),
                    sep=None,
                    engine="python",
                    encoding=encoding,
                    on_bad_lines="warn",
                )
            except Exception as exc:
                last_error = exc
        raise ValueError(f"Could not read this CSV. Last error: {last_error}")
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name or 0)
    raise ValueError("Unsupported file type. Upload a .csv, .xlsx, or .xls file.")

def infer_column_kind(series):
    """Classify columns using pandas dtypes plus conservative content heuristics."""
    if pd.api.types.is_bool_dtype(series):
        return "Boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "Date/time"
    if pd.api.types.is_numeric_dtype(series):
        return "Numeric"
    non_missing = series.dropna()
    if non_missing.empty:
        return "Empty"
    # Detect dates only when a meaningful share of values parse as dates.
    if len(non_missing) >= 3:
        parsed = pd.to_datetime(non_missing.astype(str), errors="coerce")
        if parsed.notna().mean() >= 0.85:
            return "Date/time (text)"
    unique_count = non_missing.nunique()
    if unique_count <= min(30, max(2, int(len(non_missing) * 0.05))):
        return "Categorical"
    return "Text / identifier"

def profile_frame(df):
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append({
            "Column": str(col),
            "Detected type": infer_column_kind(s),
            "Non-null": int(s.notna().sum()),
            "Missing": int(s.isna().sum()),
            "Missing %": round(float(s.isna().mean() * 100), 2),
            "Unique values": int(s.nunique(dropna=True)),
            "Duplicate values": int(s.duplicated().sum()),
        })
    return pd.DataFrame(rows)

def safe_numeric_frame(df):
    numeric = df.select_dtypes(include=np.number).copy()
    # Drop constant columns: they are not useful for correlations or anomaly models.
    return numeric.loc[:, numeric.nunique(dropna=True) > 1]

def make_summary(df):
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_pct": round(float(df.isna().sum().sum() / max(df.size, 1) * 100), 2),
    }

uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])

if not uploaded:
    st.info("Upload any tabular dataset to begin. The app adapts its analysis to the columns it finds.")
    st.subheader("Try the app with different dataset shapes")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Sales data**\n\nDate, product, region, revenue")
    c2.markdown("**Student data**\n\nStudent ID, attendance, marks")
    c3.markdown("**Sensor data**\n\nTimestamp, temperature, pressure")
    st.stop()

file_bytes = uploaded.getvalue()
sheet_name = None
if uploaded.name.lower().endswith((".xlsx", ".xls")):
    try:
        xls = pd.ExcelFile(io.BytesIO(file_bytes))
        if not xls.sheet_names:
            st.error("No worksheets found in this workbook.")
            st.stop()
        sheet_name = st.selectbox("Choose worksheet", xls.sheet_names)
    except Exception as exc:
        st.error(f"Could not inspect Excel workbook: {exc}")
        st.stop()

try:
    original_df = load_file(file_bytes, uploaded.name, sheet_name)
except Exception as exc:
    st.error(str(exc))
    st.stop()

if original_df.empty:
    st.warning("The selected sheet/file has no data rows.")
    st.stop()

# Avoid ambiguous duplicate column names in downstream charts.
df = original_df.copy()
df.columns = [str(c).strip() if str(c).strip() else f"unnamed_{i+1}" for i, c in enumerate(df.columns)]
if df.columns.duplicated().any():
    counts = {}
    new_cols = []
    for c in df.columns:
        counts[c] = counts.get(c, 0) + 1
        new_cols.append(c if counts[c] == 1 else f"{c}_{counts[c]}")
    df.columns = new_cols

summary = make_summary(df)
st.success(f"Loaded **{uploaded.name}** — {summary['rows']:,} rows × {summary['columns']:,} columns.")

# Sidebar cleaning controls are explicit; never silently alter uploaded data.
st.sidebar.header("Preprocessing options")
drop_duplicates = st.sidebar.checkbox("Remove duplicate rows", value=False)
missing_action = st.sidebar.selectbox(
    "Handle missing values",
    ["Keep as-is", "Drop rows with any missing value", "Fill numeric with median, text with mode"],
)
convert_dates = st.sidebar.checkbox("Try converting date-like text columns", value=True)

work_df = df.copy()
if convert_dates:
    for col in work_df.select_dtypes(include=["object", "string"]).columns:
        non_missing = work_df[col].dropna()
        if len(non_missing) >= 3:
            parsed = pd.to_datetime(work_df[col], errors="coerce")
            if parsed.notna().mean() >= 0.85:
                work_df[col] = parsed
if drop_duplicates:
    work_df = work_df.drop_duplicates()
if missing_action == "Drop rows with any missing value":
    work_df = work_df.dropna()
elif missing_action == "Fill numeric with median, text with mode":
    for col in work_df.columns:
        if pd.api.types.is_numeric_dtype(work_df[col]):
            median = work_df[col].median()
            if pd.notna(median):
                work_df[col] = work_df[col].fillna(median)
        else:
            mode = work_df[col].mode(dropna=True)
            if not mode.empty:
                work_df[col] = work_df[col].fillna(mode.iloc[0])

tabs = st.tabs(["Overview", "Column profile", "Explore & visualize", "Correlations", "Anomalies", "Clean & export"])

with tabs[0]:
    a, b, c, d = st.columns(4)
    a.metric("Rows", f"{len(work_df):,}", delta=f"{len(work_df)-len(df):,} after preprocessing" if len(work_df) != len(df) else None)
    b.metric("Columns", f"{work_df.shape[1]:,}")
    c.metric("Missing cells", f"{int(work_df.isna().sum().sum()):,}")
    d.metric("Duplicate rows", f"{int(work_df.duplicated().sum()):,}")
    st.subheader("Data preview")
    st.dataframe(work_df.head(100), use_container_width=True)
    st.subheader("Automatic observations")
    missing_cols = work_df.isna().sum().sort_values(ascending=False)
    missing_cols = missing_cols[missing_cols > 0]
    if missing_cols.empty:
        st.write("No missing values detected in the current working dataset.")
    else:
        top = ", ".join(f"`{col}` ({int(n)} missing)" for col, n in missing_cols.head(5).items())
        st.write(f"Columns with the most missing values: {top}.")
    if work_df.duplicated().sum():
        st.write(f"Potential issue: {int(work_df.duplicated().sum())} duplicate rows remain.")
    numeric = work_df.select_dtypes(include=np.number)
    st.write(f"Detected **{len(numeric.columns)} numeric column(s)** and **{len(work_df.select_dtypes(exclude=np.number).columns)} non-numeric column(s)**.")

with tabs[1]:
    st.subheader("Column-level data quality profile")
    prof = profile_frame(work_df)
    st.dataframe(prof, use_container_width=True, hide_index=True)
    numeric = work_df.select_dtypes(include=np.number)
    if not numeric.empty:
        st.subheader("Descriptive statistics (numeric columns)")
        st.dataframe(numeric.describe().T, use_container_width=True)
    categorical = work_df.select_dtypes(include=["object", "string", "category", "bool"])
    if not categorical.empty:
        st.subheader("Categorical summaries")
        cat_col = st.selectbox("Choose a categorical column", categorical.columns, key="cat_profile")
        counts = work_df[cat_col].astype("string").fillna("(missing)").value_counts().head(20).rename_axis("Value").reset_index(name="Count")
        st.dataframe(counts, use_container_width=True, hide_index=True)
        st.plotly_chart(px.bar(counts, x="Value", y="Count", title=f"Top values in {cat_col}"), use_container_width=True)

with tabs[2]:
    st.subheader("Build a chart from detected columns")
    all_cols = list(work_df.columns)
    if not all_cols:
        st.warning("No columns available.")
    else:
        chart_type = st.selectbox("Chart type", ["Histogram", "Box plot", "Scatter plot", "Bar chart", "Line chart"])
        numeric_cols = list(work_df.select_dtypes(include=np.number).columns)
        categorical_cols = list(work_df.select_dtypes(exclude=np.number).columns)
        if chart_type == "Histogram":
            if numeric_cols:
                col = st.selectbox("Numeric column", numeric_cols, key="hist_col")
                fig = px.histogram(work_df, x=col, title=f"Distribution of {col}", marginal="box")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric columns detected. A histogram requires numeric values.")
        elif chart_type == "Box plot":
            if numeric_cols:
                col = st.selectbox("Numeric column", numeric_cols, key="box_col")
                group_options = ["(no grouping)"] + categorical_cols
                group = st.selectbox("Optional group column", group_options, key="box_group")
                fig = px.box(work_df, y=col, x=None if group == "(no grouping)" else group, title=f"Box plot of {col}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric columns detected. A box plot requires numeric values.")
        elif chart_type == "Scatter plot":
            if len(numeric_cols) >= 2:
                x = st.selectbox("X axis", numeric_cols, key="scatter_x")
                y_options = [c for c in numeric_cols if c != x] or numeric_cols
                y = st.selectbox("Y axis", y_options, key="scatter_y")
                color_options = ["(no color)"] + categorical_cols
                color = st.selectbox("Color by (optional)", color_options, key="scatter_color")
                fig = px.scatter(work_df, x=x, y=y, color=None if color == "(no color)" else color, title=f"{y} vs {x}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("A scatter plot needs at least two numeric columns.")
        elif chart_type == "Bar chart":
            if categorical_cols:
                cat = st.selectbox("Category column", categorical_cols, key="bar_cat")
                count_df = work_df[cat].astype("string").fillna("(missing)").value_counts().head(30).rename_axis(cat).reset_index(name="Count")
                fig = px.bar(count_df, x=cat, y="Count", title=f"Top categories in {cat}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No non-numeric columns detected for a category-count bar chart.")
        elif chart_type == "Line chart":
            if numeric_cols:
                y = st.selectbox("Value column", numeric_cols, key="line_y")
                date_cols = [c for c in work_df.columns if pd.api.types.is_datetime64_any_dtype(work_df[c])]
                x_options = date_cols + [c for c in all_cols if c not in date_cols]
                x = st.selectbox("X axis / ordering column", x_options, key="line_x")
                line_df = work_df[[x, y]].dropna().sort_values(x).head(5000)
                fig = px.line(line_df, x=x, y=y, title=f"{y} by {x}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric columns detected for a line chart.")

with tabs[3]:
    st.subheader("Correlation analysis")
    numeric = safe_numeric_frame(work_df)
    if len(numeric.columns) < 2:
        st.info("Correlation analysis needs at least two non-constant numeric columns. If your file contains numeric values stored as text, convert those columns in your source or preprocess them first.")
    else:
        method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"])
        corr = numeric.corr(method=method)
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                        title=f"{method.title()} correlation matrix")
        st.plotly_chart(fig, use_container_width=True)
        pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().rename("Correlation").reset_index()
        pairs.columns = ["Column 1", "Column 2", "Correlation"]
        pairs["Absolute correlation"] = pairs["Correlation"].abs()
        st.write("Strongest absolute relationships (correlation is association, not causation):")
        st.dataframe(pairs.sort_values("Absolute correlation", ascending=False).head(15), use_container_width=True, hide_index=True)

with tabs[4]:
    st.subheader("Unusual numeric records")
    numeric = safe_numeric_frame(work_df)
    if len(numeric.columns) == 0 or len(work_df) < 10:
        st.info("Anomaly detection needs at least one varying numeric column and approximately 10 or more rows.")
    else:
        contamination = st.slider("Expected anomaly fraction", min_value=0.01, max_value=0.20, value=0.05, step=0.01)
        if st.button("Run Isolation Forest"):
            model_data = numeric.copy()
            # Median imputation is used only for the anomaly model, not to silently edit exported data.
            model_data = model_data.replace([np.inf, -np.inf], np.nan)
            model_data = model_data.fillna(model_data.median()).fillna(0)
            model = IsolationForest(contamination=contamination, random_state=42, n_estimators=150)
            labels = model.fit_predict(model_data)
            scores = model.decision_function(model_data)
            result = work_df.copy()
            result["Anomaly"] = np.where(labels == -1, "Potential anomaly", "Typical")
            result["Anomaly score"] = scores
            anomalies = result[result["Anomaly"] == "Potential anomaly"].sort_values("Anomaly score")
            st.metric("Potential anomalies", f"{len(anomalies):,}", f"{len(anomalies)/len(result)*100:.1f}% of rows")
            st.caption("Isolation Forest flags records that differ from the dataset's overall numeric patterns. These are candidates for review, not proof of errors or fraud.")
            st.dataframe(anomalies.head(200), use_container_width=True)
            st.download_button("Download anomaly results", result.to_csv(index=False).encode("utf-8"), "anomaly_results.csv", "text/csv")
        else:
            st.write("Click **Run Isolation Forest** to fit the model to this dataset's numeric columns.")

with tabs[5]:
    st.subheader("Cleaned dataset")
    st.write("The exported data reflects only the preprocessing options selected in the sidebar.")
    st.dataframe(work_df.head(100), use_container_width=True)
    csv_bytes = work_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download cleaned CSV", data=csv_bytes, file_name="cleaned_dataset.csv", mime="text/csv")
    profile_csv = profile_frame(work_df).to_csv(index=False).encode("utf-8")
    st.download_button("Download column profile", data=profile_csv, file_name="column_profile.csv", mime="text/csv")
    report = [
        "# Smart Data Analyst — Dataset Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Source file: {uploaded.name}",
        f"Rows: {len(work_df)}",
        f"Columns: {work_df.shape[1]}",
        f"Missing cells: {int(work_df.isna().sum().sum())}",
        f"Duplicate rows: {int(work_df.duplicated().sum())}",
        "",
        "## Column profile",
        profile_frame(work_df).to_markdown(index=False),
    ]
    st.download_button("Download Markdown report", data="\n".join(report).encode("utf-8"), file_name="dataset_report.md", mime="text/markdown")

st.divider()
st.caption("Prototype note: automatic type inference is heuristic. Validate inferred dates, identifiers, and anomaly flags before using results for consequential decisions.")
