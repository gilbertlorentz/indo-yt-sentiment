from flask import Flask, render_template, request, redirect, url_for, flash, Response
import re
import os
from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import csv
import io
import pandas as pd

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret-key")

# Global variable to store latest results for export
last_results = []


# ===============================
#   INDONESIAN SENTIMENT MODEL
# ===============================
MODEL_NAME = "w11wo/indonesian-roberta-base-sentiment-classifier"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# dynamic label mapping
id2label = model.config.id2label  # {0:'negative', 1:'neutral', 2:'positive'}

# YouTube API key
YT_API_KEY = os.environ.get("YT_API_KEY", "")


# ===============================
#       UTILITIES
# ===============================

def extract_video_id(url_or_id: str):
    url_or_id = url_or_id.strip()

    if re.match(r'^[A-Za-z0-9_\-]{11}$', url_or_id) and "http" not in url_or_id:
        return url_or_id

    try:
        u = urlparse(url_or_id)
        if 'youtu.be' in u.netloc:
            return u.path.lstrip('/')
        if 'youtube' in u.netloc:
            qs = parse_qs(u.query)
            if 'v' in qs:
                return qs['v'][0]

        parts = u.path.rstrip('/').split('/')
        return parts[-1]
    except:
        return None


def fetch_comments(video_id, max_results=100):
    if not YT_API_KEY:
        raise RuntimeError("YT_API_KEY missing. Provide it or paste comments manually.")

    youtube = build("youtube", "v3", developerKey=YT_API_KEY)
    comments = []

    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,
        textFormat="plainText",
        order="relevance"
    )

    while request and len(comments) < max_results:
        resp = request.execute()
        for item in resp.get("items", []):
            text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(text)
            if len(comments) >= max_results:
                break
        request = youtube.commentThreads().list_next(request, resp)

    return comments


def analyze_comments(comments):
    results = []
    pos = neu = neg = 0

    for c in comments:
        inputs = tokenizer(c, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[0]
        sentiment_id = torch.argmax(probs).item()

        label = id2label[sentiment_id]
        confidence = float(probs[sentiment_id])

        if label.lower() == "positive":
            pos += 1
        elif label.lower() == "neutral":
            neu += 1
        else:
            neg += 1

        results.append({
            "text": c,
            "label": label.lower(),
            "confidence": round(confidence, 4)
        })

    total = len(comments)

    summary = {
        "count": total,
        "positive": pos,
        "neutral": neu,
        "negative": neg,
        "positive_pct": round(pos / total * 100, 1),
        "neutral_pct": round(neu / total * 100, 1),
        "negative_pct": round(neg / total * 100, 1),
    }

    return results, summary


# ===============================
#     EXPORT ROUTES
# ===============================

@app.route("/export/csv")
def export_csv():
    global last_results
    if not last_results:
        return "No data to export", 400

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["Comment", "Sentiment", "Confidence"])
    for r in last_results:
        writer.writerow([r["text"], r["label"], r["confidence"]])

    response = Response(output.getvalue(), mimetype="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=sentiment_results.csv"
    return response


@app.route("/export/excel")
def export_excel():
    global last_results
    if not last_results:
        return "No data to export", 400

    df = pd.DataFrame(last_results)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Sentiments")

    response = Response(
        output.getvalue(),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    response.headers["Content-Disposition"] = "attachment; filename=sentiment_results.xlsx"
    return response

# ===============================
#     MAIN ROUTE
# ===============================

@app.route("/", methods=["GET", "POST"])
def index():
    global last_results

    if request.method == "POST":
        video_input = request.form.get("video_input", "").strip()
        manual_comments = request.form.get("comments_textarea", "").strip()
        max_comments = int(request.form.get("max_comments", 50))

        # PRIORITY 1 — Manual comments
        if manual_comments:
            comments = [x.strip() for x in manual_comments.splitlines() if x.strip()]
            if not comments:
                flash("No comments detected in manual input.", "warning")
                return redirect(url_for("index"))

            results, summary = analyze_comments(comments[:max_comments])
            last_results = results  # store for export
            return render_template("results.html", results=results, summary=summary, source="manual")

        # PRIORITY 2 — YouTube comments
        vid_id = extract_video_id(video_input)
        if not vid_id:
            flash("Invalid YouTube URL or ID.", "danger")
            return redirect(url_for("index"))

        if not YT_API_KEY:
            flash("Missing YT_API_KEY. Set it first or use manual comments.", "danger")
            return redirect(url_for("index"))

        try:
            comments = fetch_comments(vid_id, max_results=max_comments)
            if not comments:
                flash("No comments found (maybe comments disabled?).", "warning")
                return redirect(url_for("index"))

            results, summary = analyze_comments(comments)
            last_results = results  # store for export
            return render_template("results.html", results=results, summary=summary, source=f"video:{vid_id}")

        except Exception as e:
            flash(f"Error fetching comments: {e}", "danger")
            return redirect(url_for("index"))

    return render_template("index.html")


# ===============================
#     START FLASK
# ===============================
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
