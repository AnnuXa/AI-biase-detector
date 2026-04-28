import warnings
warnings.filterwarnings("ignore")
import google.generativeai as genai
import os

def explain_all_bias(results, domain="hiring"):
    api_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=api_key)

    domain_context = {
        "hiring": "recruitment and hiring decisions",
        "finance": "loan approvals and banking decisions",
        "healthcare": "medical diagnosis and patient care"
    }
    context = domain_context.get(domain, "automated decision making")

    summary = []
    for col, data in results.items():
        summary.append({
            "column": col,
            "distribution": data["distribution"],
            "bias_score": data["bias_score"],
            "label": data["label"]
        })

    prompt_full = f"""You are a bias expert explaining results to a non-technical person.

Dataset is used in: {context}

Bias results:
{summary}

For each column write EXACTLY in this simple format:

📌 [Column Name]
- What's happening: (1 simple sentence about the imbalance)
- Why it's a problem: (1 simple sentence about real-world harm)
- How to fix it: (1 simple actionable step)

Use very simple English. No technical jargon. No asterisks for bold. Maximum 3 bullet points per column."""

    prompt_summary = f"""You are a bias expert explaining to a non-technical person.

Dataset used in: {context}

Bias results:
{summary}

Write 2-3 simple sentences total:
- Which columns have bias
- What's the main risk in {context}
- One simple fix

Use plain English. No bullet points. No technical terms."""

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        full = model.generate_content(prompt_full).text
        short = model.generate_content(prompt_summary).text
        return {"full": full, "summary": short}

    except Exception as e:
        error_str = str(e)
        if "429" in error_str:
            msg = "⏳ Quota exceeded. Bias scores above are still accurate! Try again in a few minutes."
        else:
            msg = f"⚠️ AI explanation unavailable: {error_str}"
        return {"full": msg, "summary": msg}


def explain_model_bias(model_results):
    api_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=api_key)

    summary = []
    for col, data in model_results.items():
        group_info = {}
        for group, stats in data["group_stats"].items():
            group_info[group] = {
                "accuracy": stats["accuracy"],
                "false_negative_rate": stats["false_negative_rate"],
                "total": stats["total"]
            }
        summary.append({
            "column": col,
            "label": data["label"],
            "accuracy_gap": data["accuracy_gap"],
            "fnr_gap": data["fnr_gap"],
            "groups": group_info
        })

    prompt_full = f"""You are an AI fairness expert explaining model bias to a non-technical person.

Model bias results:
{summary}

For each column write EXACTLY in this simple format:

📌 [Column Name]
- What's happening: (1 simple sentence — which group gets wrong decisions more often)
- Why it's unfair: (1 simple sentence — real world impact on people)
- How to fix it: (1 simple actionable step)

Use very simple English. No technical jargon. No asterisks. Maximum 3 bullet points per column."""

    prompt_summary = f"""You are an AI fairness expert explaining to a non-technical person.

Model bias results:
{summary}

Write 2-3 simple sentences total:
- Which group is being treated most unfairly
- What that means for real people
- One simple fix

Use plain English. No bullet points. No technical terms."""

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        full = model.generate_content(prompt_full).text
        short = model.generate_content(prompt_summary).text
        return {"full": full, "summary": short}

    except Exception as e:
        error_str = str(e)
        if "429" in error_str:
            msg = "⏳ Quota exceeded. Try again in a few minutes."
        else:
            msg = f"⚠️ AI explanation unavailable: {error_str}"
        return {"full": msg, "summary": msg}