# Smart Data Analyst

A starter portfolio project for adaptive CSV/Excel profiling, preprocessing, EDA, correlation analysis, and anomaly detection.

## Features
- Upload CSV, XLSX, or XLS files; choose an Excel worksheet.
- CSV separator/encoding fallback.
- Column profiling: inferred type, missing values, unique values, duplicates.
- Numeric descriptive statistics and categorical frequency summaries.
- Interactive histograms, box plots, scatter plots, bar charts, and line charts.
- Pearson, Spearman, and Kendall correlation analysis.
- Optional preprocessing: remove duplicate rows, handle missing values, attempt date conversion.
- Isolation Forest anomaly detection for datasets with numeric columns.
- Export cleaned CSV, column profile CSV, anomaly results, and a Markdown report.

## Run locally (Windows)
1. Install Python 3.10 or newer.
2. Extract the ZIP and open a terminal in this folder.
3. Create a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

4. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Start the application:

   ```powershell
   streamlit run app.py
   ```

The app should open in your browser. If it does not, visit the local URL printed in the terminal, usually `http://localhost:8501`.

## Run locally (macOS/Linux)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## How the app adapts to different files
The app does not require a fixed schema. It inspects the columns in the uploaded dataset and creates different analyses based on the detected types:
- Numeric columns → statistics, histograms, box plots, correlations, anomaly model.
- Categorical/text columns → frequency summaries and category-count charts.
- Date-like text columns → optional date conversion and time-based line charts.
- Mixed or sparse datasets → profile shows missingness and inferred types; analyses that need unavailable columns are disabled or replaced with an explanatory message.

This is heuristic type detection, not semantic understanding. For example, a numeric `customer_id` should usually be treated as an identifier rather than a continuous measurement. A future improvement is a column-role editor that lets users override the inferred type.

## Suggested test datasets
Try at least three different shapes:
1. Sales: date, product/category, region, quantity, revenue.
2. Student performance: student_id, attendance, marks, department.
3. Sensor/IoT: timestamp, temperature, pressure, vibration.

Test edge cases too: empty cells, duplicate rows, text stored in numeric columns, constant columns, unusual separators, and Excel files with multiple worksheets.

## Important limitations
- The app currently provides general-purpose EDA; it does not infer business meaning or train predictive models.
- Isolation Forest finds unusual patterns in numeric features; it does not prove a row is wrong.
- Median/mode imputation is a simple baseline and may be inappropriate for some datasets.
- Large files may require sampling or more memory-efficient processing.
- For production use, add file-size limits, explicit schema overrides, tests, logging, and privacy safeguards.
