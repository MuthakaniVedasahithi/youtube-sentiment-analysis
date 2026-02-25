from flask import Flask, render_template, request, send_file
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from transformers import pipeline
from wordcloud import WordCloud
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from io import BytesIO
from reportlab.platypus import Image as RLImage
from reportlab.lib.utils import ImageReader
import re
import html
import os

app = Flask(__name__)

# ==============================
# API KEY
# ==============================
API_KEY = os.getenv("YOUTUBE_API_KEY")

if not API_KEY:
    print("WARNING: YOUTUBE_API_KEY not found")

# ==============================
# Load Sentiment Pipeline (Optimized)
# ==============================
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

# Store report data globally
report_data = {}


# ==============================
# Extract Video ID
# ==============================
def extract_video_id(link):
    if "v=" in link:
        return link.split("v=")[1].split("&")[0]
    elif "youtu.be/" in link:
        return link.split("youtu.be/")[1].split("?")[0]
    elif "shorts/" in link:
        return link.split("shorts/")[1].split("?")[0]
    return None


# ==============================
# Get Video Statistics
# ==============================
def get_video_stats(video_id):
    youtube = build("youtube", "v3", developerKey=API_KEY)
    try:
        request = youtube.videos().list(
            part="statistics,snippet",
            id=video_id
        )
        response = request.execute()

        if not response["items"]:
            return None

        stats = response["items"][0]["statistics"]
        snippet = response["items"][0]["snippet"]

        return {
            "title": snippet["title"],
            "views": stats.get("viewCount", 0),
            "likes": stats.get("likeCount", 0),
            "comments_count": stats.get("commentCount", 0)
        }
    except HttpError:
        return None


# ==============================
# Get Comments (LIMITED TO 10)
# ==============================
def get_comments(video_id):
    youtube = build("youtube", "v3", developerKey=API_KEY)

    comments = []
    next_page_token = None
    MAX_COMMENTS = 20

    try:
        while True:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token
            )

            response = request.execute()

            for item in response.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                comment_text = snippet["textDisplay"]
                like_count = snippet.get("likeCount", 0)
                comments.append((comment_text, like_count))

                if len(comments) >= MAX_COMMENTS:
                    return comments

            next_page_token = response.get("nextPageToken")

            if not next_page_token:
                break

        return comments

    except HttpError:
        return []


# ==============================
# Analyze Sentiment (Optimized)
# ==============================
def analyze_sentiment(comment):
    comment = re.sub(r"<.*?>", "", comment)

    if not comment.strip():
        return "Neutral"

    result = sentiment_pipeline(comment[:256])[0]
    label = result["label"].upper()

    if "NEGATIVE" in label:
        return "Negative"
    elif "NEUTRAL" in label:
        return "Neutral"
    else:
        return "Positive"


# ==============================
# HOME ROUTE
# ==============================
@app.route("/", methods=["GET", "POST"])
def home():

    global report_data

    positive = negative = neutral = 0
    positive_comments = []
    negative_comments = []
    neutral_comments = []
    top_liked_comments = []
    video_stats = None
    error_message = ""

    if request.method == "POST":

        link = request.form.get("link")
        video_id = extract_video_id(link)

        if not video_id:
            error_message = "Invalid YouTube Link!"
        else:
            video_stats = get_video_stats(video_id)

            if not video_stats:
                error_message = "Unable to fetch video details."
                return render_template("index.html", error_message=error_message)

            comments_data = get_comments(video_id)

            if comments_data:

                all_text = ""

                for comment, like_count in comments_data:
                    sentiment = analyze_sentiment(comment)
                    all_text += " " + comment

                    if sentiment == "Positive":
                        positive += 1
                        positive_comments.append(comment)
                    elif sentiment == "Negative":
                        negative += 1
                        negative_comments.append(comment)
                    else:
                        neutral += 1
                        neutral_comments.append(comment)

                top_liked_comments = sorted(
                    comments_data,
                    key=lambda x: x[1],
                    reverse=True
                )[:5]

                if not all_text.strip():
                    all_text = "No comments available"

                wordcloud = WordCloud(
                    width=800,
                    height=400,
                    background_color="white"
                ).generate(all_text)

                if not os.path.exists("static"):
                    os.makedirs("static")


                img_buffer = BytesIO()
                wordcloud.to_image().save(img_buffer, format="PNG")
                img_buffer.seek(0)

                report_data = {
                    "video_stats": video_stats,
                    "positive": positive,
                    "negative": negative,
                    "neutral": neutral,
                    "wordcloud_image": img_buffer.getvalue()
                }


    return render_template(
        "index.html",
        positive=positive,
        negative=negative,
        neutral=neutral,
        positive_comments=positive_comments,
        negative_comments=negative_comments,
        neutral_comments=neutral_comments,
        video_stats=video_stats,
        top_liked_comments=top_liked_comments,
        error_message=error_message
    )


# ==============================
# DOWNLOAD REPORT
# ==============================
@app.route("/download_report")
def download_report():

    global report_data

    if not report_data:
        return "No report available."

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("<b>YouTube Sentiment Analysis Report</b>", styles["Title"]))
    elements.append(Spacer(1, 12))

    vs = report_data["video_stats"]

    safe_title = html.escape(vs["title"])
    elements.append(Paragraph(f"<b>Video Title:</b> {safe_title}", styles["Normal"]))
    elements.append(Paragraph(f"Views: {vs['views']}", styles["Normal"]))
    elements.append(Paragraph(f"Likes: {vs['likes']}", styles["Normal"]))
    elements.append(Paragraph(f"Total Comments: {vs['comments_count']}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("<b>Sentiment Summary:</b>", styles["Heading2"]))
    elements.append(Paragraph(f"Positive: {report_data['positive']}", styles["Normal"]))
    elements.append(Paragraph(f"Negative: {report_data['negative']}", styles["Normal"]))
    elements.append(Paragraph(f"Neutral: {report_data['neutral']}", styles["Normal"]))
    elements.append(Spacer(1, 20))

    # Add WordCloud if available
    if "wordcloud_image" in report_data:
        elements.append(Paragraph("<b>Word Cloud:</b>", styles["Heading2"]))
        elements.append(Spacer(1, 10))

        img_buffer = BytesIO(report_data["wordcloud_image"])
        image = RLImage(img_buffer, width=5 * inch, height=3 * inch)
        elements.append(image)

    doc.build(elements)

    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="YouTube_Analysis_Report.pdf",
        mimetype="application/pdf"
    )
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))