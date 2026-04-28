import warnings
warnings.filterwarnings("ignore")
import json
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, Response
import pandas as pd
import numpy as np
from scipy.stats import chisquare, entropy
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors as rl_colors
import io
import os
from gemini import explain_all_bias

app = Flask(__name__)
app.secret_key = "bias_secret_key"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def compute_gini(values):
    arr = np.array(sorted(values))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    gini = (2 * np.sum((np.arange(1, n+1)) * arr) - (n + 1) * arr.sum()) / (n * arr.sum())
    return round(float(gini), 3)

def compute_bias(series, df=None, all_columns=None):
    counts = series.value_counts()
    distribution = counts.to_dict()
    values = list(counts.values)

    if len(counts) < 2:
        return {
            "distribution": distribution,
            "bias_score": 0.0,
            "label": "✅ Balanced",
            "chi_square": {"stat": 0, "p_value": 1.0, "significant": False},
            "entropy_score": 1.0,
            "gini": 0.0,
            "correlations": {}
        }

    max_val = counts.max()
    min_val = counts.min()
    bias_score = round((max_val - min_val) / max_val, 2)

    try:
        expected = [sum(values) / len(values)] * len(values)
        chi_stat, p_value = chisquare(values, f_exp=expected)
        chi_significant = p_value < 0.05
    except:
        chi_stat, p_value, chi_significant = 0, 1.0, False

    try:
        ent = entropy(values, base=2)
        max_ent = np.log2(len(values)) if len(values) > 1 else 1
        norm_entropy = round(ent / max_ent, 3) if max_ent > 0 else 1.0
    except:
        norm_entropy = 1.0

    gini = compute_gini(values)

    correlations = {}
    if df is not None and all_columns is not None:
        cat_col = series.name
        for other_col in all_columns:
            if other_col == cat_col:
                continue
            try:
                if pd.api.types.is_numeric_dtype(df[other_col]):
                    group_means = df.groupby(cat_col)[other_col].mean()
                    if len(group_means) >= 2:
                        diff = round(float(group_means.max() - group_means.min()), 2)
                        pct_diff = round(diff / float(group_means.mean()) * 100, 1) if group_means.mean() != 0 else 0
                        correlations[other_col] = {
                            "group_means": {str(k): round(float(v), 2) for k, v in group_means.items()},
                            "difference": diff,
                            "pct_difference": pct_diff,
                            "biased": pct_diff > 10
                        }
            except:
                continue

    signals = 0
    if bias_score > 0.2: signals += 1
    if chi_significant: signals += 1
    if gini > 0.2: signals += 1
    if norm_entropy < 0.8: signals += 1

    if signals >= 3:
        label = "⚠️ High Bias"
    elif signals >= 1:
        label = "⚡ Moderate Bias"
    else:
        label = "✅ Balanced"

    return {
        "distribution": distribution,
        "bias_score": bias_score,
        "label": label,
        "chi_square": {
            "stat": round(float(chi_stat), 3),
            "p_value": round(float(p_value), 4),
            "significant": chi_significant
        },
        "entropy_score": norm_entropy,
        "gini": gini,
        "correlations": correlations
    }

def compute_model_bias(df, actual_col, predicted_col, group_cols):
    results = {}
    for col in group_cols:
        if col in [actual_col, predicted_col]:
            continue
        groups = df[col].unique()
        group_stats = {}
        for group in groups:
            subset = df[df[col] == group]
            total = len(subset)
            if total == 0:
                continue
            actual = subset[actual_col]
            predicted = subset[predicted_col]
            tp = int(((actual == 1) & (predicted == 1)).sum())
            fp = int(((actual == 0) & (predicted == 1)).sum())
            tn = int(((actual == 0) & (predicted == 0)).sum())
            fn = int(((actual == 1) & (predicted == 0)).sum())
            accuracy = round((tp + tn) / total * 100, 1)
            fpr = round(fp / (fp + tn) * 100, 1) if (fp + tn) > 0 else 0
            fnr = round(fn / (fn + tp) * 100, 1) if (fn + tp) > 0 else 0
            precision = round(tp / (tp + fp) * 100, 1) if (tp + fp) > 0 else 0
            group_stats[str(group)] = {
                "total": total,
                "accuracy": accuracy,
                "false_positive_rate": fpr,
                "false_negative_rate": fnr,
                "precision": precision,
                "tp": tp, "fp": fp, "tn": tn, "fn": fn
            }
        if len(group_stats) < 2:
            continue
        accuracies = [v["accuracy"] for v in group_stats.values()]
        fnrs = [v["false_negative_rate"] for v in group_stats.values()]
        acc_gap = round(max(accuracies) - min(accuracies), 1)
        fnr_gap = round(max(fnrs) - min(fnrs), 1)
        if acc_gap > 15 or fnr_gap > 20:
            bias_label = "⚠️ High Model Bias"
        elif acc_gap > 5 or fnr_gap > 10:
            bias_label = "⚡ Moderate Model Bias"
        else:
            bias_label = "✅ Fair Model"
        results[col] = {
            "group_stats": group_stats,
            "accuracy_gap": acc_gap,
            "fnr_gap": fnr_gap,
            "label": bias_label
        }
    return results

# ↓ ADD THESE 4 LINES HERE
@app.route("/")
def home():
    return render_template("landing.html")

# ↓ THIS WAS ALREADY HERE — just change / to /upload
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files["file"]
        ...
    if request.method == "POST":
        file = request.files["file"]
        domain = request.form.get("domain", "hiring")
        if file and file.filename.endswith(".csv"):
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)
            session["filepath"] = path
            session["domain"] = domain
            return redirect(url_for("select_columns"))
    return render_template("upload.html")

@app.route("/select", methods=["GET", "POST"])
def select_columns():
    filepath = session.get("filepath")
    df = pd.read_csv(filepath)
    columns = df.columns.tolist()
    if request.method == "POST":
        selected = request.form.getlist("columns")
        session["selected"] = selected
        return redirect(url_for("results"))
    return render_template("select.html", columns=columns)

@app.route("/results")
def results():
    filepath = session.get("filepath")
    selected = session.get("selected")
    df = pd.read_csv(filepath)

    results = {}
    for col in selected:
        results[col] = compute_bias(df[col], df=df, all_columns=selected)

    total = len(results)
    high = sum(1 for d in results.values() if "⚠️" in d["label"])
    moderate = sum(1 for d in results.values() if "⚡" in d["label"])
    balanced = sum(1 for d in results.values() if "✅" in d["label"])

    avg_score = round(sum(d["bias_score"] for d in results.values()) / total, 2) if total else 0
    health_score = max(0, int(100 - (avg_score * 100)))
    most_biased = max(results.items(), key=lambda x: x[1]["bias_score"])

    if high > 0:
        overall_label = "⚠️ High Bias Detected"
        overall_class = "overall-high"
    elif moderate > 0:
        overall_label = "⚡ Moderate Bias Detected"
        overall_class = "overall-moderate"
    else:
        overall_label = "✅ Dataset Looks Balanced"
        overall_class = "overall-low"

    overall = {
        "label": overall_label,
        "class": overall_class,
        "health_score": health_score,
        "total": total,
        "high": high,
        "moderate": moderate,
        "balanced": balanced,
        "most_biased_col": most_biased[0],
        "most_biased_score": most_biased[1]["bias_score"]
    }

    domain = session.get("domain", "hiring")
    explanation = explain_all_bias(results, domain=domain)
    if isinstance(explanation, str):
        explanation = {"full": explanation, "summary": explanation}

    return render_template("result.html", results=results, explanation=explanation, overall=overall, domain=domain)

@app.route("/model", methods=["GET", "POST"])
def model_bias():
    return render_template("model_upload.html", columns=[])

@app.route("/model/columns", methods=["POST"])
def model_columns():
    file = request.files["file"]
    if file and file.filename.endswith(".csv"):
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)
        session["model_filepath"] = path
        df = pd.read_csv(path)
        columns = df.columns.tolist()
        return render_template("model_upload.html", columns=columns)
    return render_template("model_upload.html", columns=[])

@app.route("/model/results", methods=["POST"])
def model_results():
    filepath = session.get("model_filepath")
    actual_col = request.form.get("actual_col")
    predicted_col = request.form.get("predicted_col")
    group_cols = request.form.getlist("group_cols")
    df = pd.read_csv(filepath)
    model_results = compute_model_bias(df, actual_col, predicted_col, group_cols)
    high = sum(1 for d in model_results.values() if "⚠️" in d["label"])
    moderate = sum(1 for d in model_results.values() if "⚡" in d["label"])
    fair = sum(1 for d in model_results.values() if "✅" in d["label"])
    from gemini import explain_model_bias
    explanation = explain_model_bias(model_results)
    if isinstance(explanation, str):
        explanation = {"full": explanation, "summary": explanation}
    return render_template("model_result.html",
        model_results=model_results,
        actual_col=actual_col,
        predicted_col=predicted_col,
        high=high, moderate=moderate, fair=fair,
        explanation=explanation
    )

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    filepath = session.get("filepath")

    if not filepath:
        return jsonify({"error": "No dataset loaded"})

    df = pd.read_csv(filepath)
    columns = df.columns.tolist()
    sample = df.head(3).to_dict(orient="records")

    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    model = None
    for m in ["gemini-2.5-flash", "gemini-2.0-flash-lite", "gemini-2.0-flash"]:
        try:
            model = genai.GenerativeModel(m)
            break
        except:
            continue

    prompt = f"""You are a data analysis assistant. The user has a pandas DataFrame called 'df'.

DataFrame columns: {columns}
Sample data: {sample}

User request: "{user_message}"

Your job:
1. Generate a pandas filter expression to find matching rows
2. Give a short friendly response explaining what you found

Respond ONLY in this exact JSON format:
{{
  "filter": "df[pandas_filter_here]",
  "response": "friendly explanation here",
  "description": "short label for what was found"
}}

Rules:
- filter must be valid pandas code using 'df'
- If no filter needed, use "filter": "df"
- Use exact column names from: {columns}
- For string columns use .str.contains() for partial matches
- Keep response simple and friendly
- If you cannot create a filter, set filter to "df" and explain in response"""

    try:
        result = model.generate_content(prompt)
        text = result.text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        parsed = json.loads(text)
        filter_expr = parsed.get("filter", "df")
        response_text = parsed.get("response", "Here are the results!")
        description = parsed.get("description", "Filtered results")

        filtered_df = eval(filter_expr, {"df": df, "pd": pd})
        rows = filtered_df.head(50).to_dict(orient="records")
        count = len(filtered_df)

        session["filtered_data"] = filtered_df.head(50).to_json()
        session["filter_description"] = description

        return jsonify({
            "response": response_text,
            "rows": rows,
            "columns": columns,
            "count": count,
            "description": description,
            "filter": filter_expr
        })

    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "quota" in error_str.lower():
            msg = "⏳ Gemini quota exceeded. Please wait a few minutes and try again!"
        elif "404" in error_str:
            msg = "⚠️ AI model unavailable. Please try again shortly."
        else:
            msg = "⚠️ Something went wrong. Try rephrasing your question!"
        return jsonify({
            "response": msg,
            "rows": [],
            "columns": [],
            "count": 0,
            "description": "",
            "filter": ""
        })

@app.route("/export/csv")
def export_csv():
    filtered_json = session.get("filtered_data")
    description = session.get("filter_description", "filtered_results")
    if not filtered_json:
        return "No data to export", 400
    filtered_df = pd.read_json(filtered_json)
    csv_data = filtered_df.to_csv(index=False)
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename={description.replace(' ', '_')}.csv"}
    )

@app.route("/export/pdf")
def export_pdf():
    filtered_json = session.get("filtered_data")
    description = session.get("filter_description", "Filtered Results")
    if not filtered_json:
        return "No data to export", 400

    filtered_df = pd.read_json(filtered_json)
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter),
                           rightMargin=30, leftMargin=30,
                           topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    elements = []
    title = Paragraph(f"<b>AI Bias Detector — {description}</b>", styles['Title'])
    elements.append(title)
    elements.append(Spacer(1, 12))
    subtitle = Paragraph(f"Total rows: {len(filtered_df)}", styles['Normal'])
    elements.append(subtitle)
    elements.append(Spacer(1, 12))
    cols = filtered_df.columns.tolist()
    table_data = [cols]
    for _, row in filtered_df.iterrows():
        table_data.append([str(row[c])[:20] for c in cols])
    col_width = (landscape(letter)[0] - 60) / len(cols)
    table = Table(table_data, colWidths=[col_width] * len(cols))
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor('#4285F4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor('#f0f4ff')]),
        ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.HexColor('#dddddd')),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return Response(
        buffer.getvalue(),
        mimetype="application/pdf",
        headers={"Content-Disposition": f"attachment;filename={description.replace(' ', '_')}.pdf"}
    )

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)