from flask import Flask, render_template, request
import PyPDF2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)



# Extract text from PDF
def extract_text(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Calculate similarity
def get_similarity(resume, job_desc):
    text = [resume, job_desc]
    cv = CountVectorizer()
    matrix = cv.fit_transform(text)
    similarity = cosine_similarity(matrix)[0][1]
    return round(similarity * 100, 2)

def analyze_resume(resume_text, job_desc):
    resume_words = set(resume_text.lower().split())
    job_words = set(job_desc.lower().split())

    matched = resume_words.intersection(job_words)
    missing = job_words - resume_words

    score = (len(matched) / len(job_words)) * 100 if job_words else 0

    return score, list(matched), list(missing)

@app.route("/", methods=["GET", "POST"])
def index():
    score = None
    if request.method == "POST":
        file = request.files["resume"]
        job_desc = request.form["job_desc"]

        resume_text = extract_text(file)
        score = get_similarity(resume_text, job_desc)

    return render_template("index.html", score=score)

@app.route('/analyze', methods=['POST'])
def analyze():
    resume = request.form['resume']
    job_desc = request.form['job_desc']

    score, matched, missing = analyze_resume(resume, job_desc)

    return render_template(
        'result.html',
        score=round(score, 2),
        matched=matched[:10],   # limit to top 10
        missing=missing[:10]
    )

if __name__ == "__main__":
    app.run(debug=True)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)