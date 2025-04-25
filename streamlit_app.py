import numpy as np
import pandas as pd
import re
import nltk
import string
import spacy
import sklearn
import PyPDF2
import docx
import io
import os
import base64
import sqlite3
import datetime
import json
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_md')
except:
    # Download the model if not available
    os.system('python -m spacy download en_core_web_md')
    nlp = spacy.load('en_core_web_md')

class SkillsDatabase:
    def __init__(self, db_path='skills_database.db'):
        """Initialize the skills database"""
        self.db_path = db_path
        self.connection = None
        self.cursor = None
        self.setup_database()

    def setup_database(self):
        """Set up the database and create tables if they don't exist"""
        self.connection = sqlite3.connect(self.db_path)
        self.cursor = self.connection.cursor()

        # Create skills table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS skills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            skill TEXT UNIQUE,
            category TEXT,
            frequency INTEGER DEFAULT 1,
            last_used TIMESTAMP,
            confidence FLOAT DEFAULT 0.75,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # Create embeddings table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS skill_embeddings (
            skill_id INTEGER,
            embedding BLOB,
            FOREIGN KEY (skill_id) REFERENCES skills(id)
        )
        ''')

        # Create table for potential new skills (candidates)
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS skill_candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            skill TEXT UNIQUE,
            occurrences INTEGER DEFAULT 1,
            contexts TEXT,
            first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        self.connection.commit()

    def _skill_exists(self, skill):
        """Check if a skill already exists in the database"""
        self.cursor.execute("SELECT id FROM skills WHERE skill=?", (skill.lower(),))
        return self.cursor.fetchone() is not None

    def _get_skill_id(self, skill):
        """Get the ID of a skill"""
        self.cursor.execute("SELECT id FROM skills WHERE skill=?", (skill.lower(),))
        result = self.cursor.fetchone()
        return result[0] if result else None

    def add_initial_skills(self, skills_list):
        """Add initial skills to the database"""
        categories = {
            "programming": ["python", "java", "javascript", "c++", "c#", "ruby", "go", "swift",
                            "kotlin", "php", "typescript", "rust", "scala", "perl", "r"],
            "frontend": ["html", "css", "react", "angular", "vue", "bootstrap", "tailwind",
                         "jquery", "sass", "redux", "webpack", "nextjs", "gatsby"],
            "backend": ["node", "django", "flask", "spring", "express", "laravel", "ruby on rails",
                        "aspnet", "fastapi", "graphql"],
            "database": ["sql", "mysql", "postgresql", "mongodb", "oracle", "sqlite", "redis",
                         "elasticsearch", "cassandra", "dynamodb", "firestore"],
            "devops": ["docker", "kubernetes", "jenkins", "aws", "azure", "gcp", "terraform",
                       "ansible", "circleci", "travis", "git", "github actions"],
            "mobile": ["android", "ios", "flutter", "react native", "xamarin", "swift", "kotlin",
                        "objective-c", "mobile app development"],
            "data_science": ["machine learning", "deep learning", "data analysis", "nlp", "computer vision",
                              "tensorflow", "pytorch", "pandas", "numpy", "scikit-learn", "statistics"],
            "soft_skills": ["leadership", "communication", "teamwork", "problem solving", "critical thinking",
                            "time management", "project management", "agile", "scrum"]
        }

        for skill in skills_list:
            category = "other"
            for cat, items in categories.items():
                if skill.lower() in items:
                    category = cat
                    break

            if not self._skill_exists(skill):
                # Get the embedding
                doc = nlp(skill)
                embedding = doc.vector

                # Insert the skill
                self.cursor.execute(
                    "INSERT INTO skills (skill, category, last_used) VALUES (?, ?, ?)",
                    (skill.lower(), category, datetime.datetime.now())
                )
                skill_id = self.cursor.lastrowid

                # Insert the embedding
                self.cursor.execute(
                    "INSERT INTO skill_embeddings (skill_id, embedding) VALUES (?, ?)",
                    (skill_id, embedding.tobytes())
                )

        self.connection.commit()

    def get_all_skills(self):
        """Get all skills from the database"""
        self.cursor.execute("SELECT skill FROM skills ORDER BY frequency DESC")
        return [row[0] for row in self.cursor.fetchall()]

    def update_skill_frequency(self, skill):
        """Update the frequency counter for a skill"""
        skill_id = self._get_skill_id(skill)
        if skill_id:
            self.cursor.execute(
                "UPDATE skills SET frequency = frequency + 1, last_used = ? WHERE id = ?",
                (datetime.datetime.now(), skill_id)
            )
            self.connection.commit()

    def add_skill_candidate(self, skill, context):
        """Add a potential new skill to the candidates table"""
        skill = skill.lower()

        # Check if it's already a confirmed skill
        if self._skill_exists(skill):
            return

        # Check if it's already a candidate
        self.cursor.execute("SELECT id, contexts, occurrences FROM skill_candidates WHERE skill=?", (skill,))
        result = self.cursor.fetchone()

        if result:
            # Update existing candidate
            candidate_id, contexts_json, occurrences = result
            contexts = json.loads(contexts_json) if contexts_json else []
            contexts.append(context)

            # Keep only the most recent 10 contexts
            if len(contexts) > 10:
                contexts = contexts[-10:]

            self.cursor.execute(
                "UPDATE skill_candidates SET occurrences = ?, contexts = ? WHERE id = ?",
                (occurrences + 1, json.dumps(contexts), candidate_id)
            )
        else:
            # Add new candidate
            contexts = [context]
            self.cursor.execute(
                "INSERT INTO skill_candidates (skill, contexts) VALUES (?, ?)",
                (skill, json.dumps(contexts))
            )

        self.connection.commit()

    def promote_candidates(self, min_occurrences=3):
        """Promote skill candidates to confirmed skills if they meet criteria"""
        self.cursor.execute(
            "SELECT id, skill, contexts FROM skill_candidates WHERE occurrences >= ?",
            (min_occurrences,)
        )
        candidates = self.cursor.fetchall()

        promoted = []
        for candidate_id, skill, contexts_json in candidates:
            # Skip if it's already in the skills table
            if self._skill_exists(skill):
                self.cursor.execute("DELETE FROM skill_candidates WHERE id = ?", (candidate_id,))
                continue

            # Calculate confidence based on contexts
            contexts = json.loads(contexts_json)
            confidence = min(0.5 + (len(contexts) * 0.05), 0.9)  # Starts at 0.5, max 0.9

            # Determine category using ML clustering
            category = self._determine_category(skill)

            # Get embedding
            doc = nlp(skill)
            embedding = doc.vector

            # Insert into skills table
            self.cursor.execute(
                "INSERT INTO skills (skill, category, confidence, last_used) VALUES (?, ?, ?, ?)",
                (skill, category, confidence, datetime.datetime.now())
            )
            skill_id = self.cursor.lastrowid

            # Insert embedding
            self.cursor.execute(
                "INSERT INTO skill_embeddings (skill_id, embedding) VALUES (?, ?)",
                (skill_id, embedding.tobytes())
            )

            # Remove from candidates
            self.cursor.execute("DELETE FROM skill_candidates WHERE id = ?", (candidate_id,))

            promoted.append((skill, category, confidence))

        self.connection.commit()
        return promoted

    def _determine_category(self, skill):
        """Use ML to determine the category of a new skill"""
        # Get embeddings of existing skills with categories
        self.cursor.execute('''
            SELECT s.category, se.embedding
            FROM skills s
            JOIN skill_embeddings se ON s.id = se.skill_id
        ''')

        categories = []
        embeddings = []

       # Get embedding for the new skill
        doc = nlp(skill)
        new_embedding = doc.vector
        embedding_size = len(new_embedding)

        for category, embedding_bytes in self.cursor.fetchall():
            # Convert bytes to numpy array with correct shape
            stored_embedding = np.frombuffer(embedding_bytes)
        
            # Only use embeddings with matching dimensions
            if len(stored_embedding) == embedding_size:
                categories.append(category)
                embeddings.append(stored_embedding)

        if not embeddings:
            return "other"  # Default category if no compatible embeddings found

        # Find most similar skills
        similarities = [cosine_similarity([new_embedding], [emb])[0][0] for emb in embeddings]
        most_similar_idx = np.argmax(similarities)

        # Use the category of the most similar skill
        return categories[most_similar_idx]

    def clean_old_candidates(self, days_threshold=30):
        """Remove old candidates that haven't received enough occurrences"""
        threshold_date = datetime.datetime.now() - datetime.timedelta(days=days_threshold)

        self.cursor.execute(
           "DELETE FROM skill_candidates WHERE first_seen < ? AND occurrences < 3",
            (threshold_date,)
        )
    
        self.connection.commit()

    def close(self):
        """Close the database connection"""
        if self.connection:
            self.connection.close()


class JobFitMatcher:
    def __init__(self, db_path='skills_database.db'):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer()
        self.skills_db = SkillsDatabase(db_path)

        # Initialize database with initial skills if empty
        if not self.skills_db.get_all_skills():
            initial_skills = [
                # Programming Languages
                "python", "java", "javascript", "c++", "c#", "ruby", "go", "swift", "kotlin",
                "php", "typescript", "rust", "scala", "perl", "r", "matlab", "dart", "lua",

                # Frontend
                "html", "css", "react", "angular", "vue", "bootstrap", "tailwind", "jquery",
                "sass", "less", "redux", "webpack", "nextjs", "gatsby", "nuxt", "svelte",

                # Backend
                "node.js", "django", "flask", "spring", "express", "laravel", "ruby on rails",
                "asp.net", "fastapi", "graphql", "rest api", "microservices", "serverless",

                # Database
                "sql", "mysql", "postgresql", "mongodb", "oracle", "sqlite", "redis",
                "elasticsearch", "cassandra", "dynamodb", "neo4j", "firestore", "couchdb",

                # DevOps
                "docker", "kubernetes", "jenkins", "aws", "azure", "gcp", "terraform",
                "ansible", "circleci", "travis", "git", "github actions", "gitlab ci",
                "cloud computing", "devops", "ci/cd",

                # Mobile
                "android", "ios", "flutter", "react native", "xamarin", "swift", "kotlin",
                "objective-c", "mobile app development", "pwa", "responsive design",

                # Data Science
                "machine learning", "deep learning", "data analysis", "nlp", "computer vision",
                "tensorflow", "pytorch", "pandas", "numpy", "scikit-learn", "statistics",
                "data mining", "big data", "hadoop", "spark", "data visualization",

                # Soft Skills
                "leadership", "communication", "teamwork", "problem solving", "critical thinking",
                "time management", "project management", "agile", "scrum", "kanban", "lean"
            ]
            self.skills_db.add_initial_skills(initial_skills)

    def _get_skills_keywords(self):
        """Get all skills from the database"""
        return self.skills_db.get_all_skills()

    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        # Convert to lowercase
        text = text.lower()

        # Remove special characters and numbers (but keep spaces)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)

        # Tokenize
        tokens = nltk.word_tokenize(text)

        # Remove stopwords and lemmatize
        filtered_tokens = [self.lemmatizer.lemmatize(w) for w in tokens if w not in self.stop_words]

        return ' '.join(filtered_tokens)

    def extract_skills(self, text):
        """Extract skills from text using NLP and the skills database"""
        preprocessed_text = self.preprocess_text(text)
        skills = []

        # Get all skills from the database
        all_skills = self._get_skills_keywords()

        # Check for multi-word skills first (to avoid partial matches)
        multi_word_skills = [skill for skill in all_skills if ' ' in skill]
        for skill in multi_word_skills:
            if skill in preprocessed_text:
                skills.append(skill)
                # Update frequency in database
                self.skills_db.update_skill_frequency(skill)
                # Remove the skill to avoid double counting
                preprocessed_text = preprocessed_text.replace(skill, '')

        # Check for single-word skills
        words = preprocessed_text.split()
        for word in words:
            if word in all_skills:
                skills.append(word)
                # Update frequency in database
                self.skills_db.update_skill_frequency(word)

        # Extract potential new skills using NER
        self._extract_potential_skills(text)

        return list(set(skills))

    def _extract_potential_skills(self, text):
        """Extract potential new skills and add them to the candidates database"""
        doc = nlp(text)

        # Check for named entities that might be skills
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PRODUCT"]:
                candidate = ent.text.lower()
                context = text[max(0, ent.start_char - 50):min(len(text), ent.end_char + 50)]
                self.skills_db.add_skill_candidate(candidate, context)

        # Look for technical terms patterns
        tech_patterns = [
            r'\b[A-Za-z\-\.]+\b \b(?:framework|language|library|tool|platform|system|database|technology)\b',
            r'\bfamiliarity with \b([A-Za-z\-\.]+)',
            r'\bexperience (?:in|with) \b([A-Za-z\-\.]+)',
            r'\bproficient (?:in|with) \b([A-Za-z\-\.]+)',
            r'\bknowledge of \b([A-Za-z\-\.]+)'
        ]

        for pattern in tech_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                if len(match.groups()) > 0:
                    candidate = match.group(1)
                else:
                    candidate = match.group(0).split()[0]  # First word in the match

                # Add candidate with context
                context = text[max(0, match.start() - 30):min(len(text), match.end() + 30)]
                self.skills_db.add_skill_candidate(candidate, context)

        # Try to promote candidates that have enough occurrences
        self.skills_db.promote_candidates()

        # Clean up old candidates
        self.skills_db.clean_old_candidates()

    def missing_keywords(self, resume_text, jd_text):
        """Identify keywords in JD that are missing from resume"""
        jd_skills = self.extract_skills(jd_text)
        resume_skills = self.extract_skills(resume_text)

        return [skill for skill in jd_skills if skill not in resume_skills]

    def calculate_match_score(self, resume_text, jd_text):
        """Calculate match score between resume and job description"""
        # Preprocess texts
        processed_resume = self.preprocess_text(resume_text)
        processed_jd = self.preprocess_text(jd_text)

        # Vectorize the texts
        documents = [processed_resume, processed_jd]
        tfidf_matrix = self.vectorizer.fit_transform(documents)

        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

        # Convert to percentage
        match_percentage = round(similarity * 100, 2)

        return match_percentage

    def calculate_ats_score(self, resume_text, jd_text):
        """Calculate ATS compatibility score"""
        # Extract metrics that might influence ATS scoring
        match_score = self.calculate_match_score(resume_text, jd_text)
        missing_keys = self.missing_keywords(resume_text, jd_text)

        # Resume format check
        format_score = 20  # Base score

        # Check for key sections
        sections = ["experience", "education", "skills", "projects"]
        for section in sections:
            if section in resume_text.lower():
                format_score += 5

        # Check for contact information
        contact_patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone
            r'\blinkedin\.com/in/[A-Za-z0-9_-]+\b',  # LinkedIn
            r'\bgithub\.com/[A-Za-z0-9_-]+\b'  # GitHub
        ]

        for pattern in contact_patterns:
            if re.search(pattern, resume_text):
                format_score += 2.5

        # Calculate keyword match weight
        jd_skills = self.extract_skills(jd_text)
        if len(jd_skills) > 0:
            keyword_match_rate = (len(jd_skills) - len(missing_keys)) / len(jd_skills)
        else:
            keyword_match_rate = 0.5  # Default if no skills are found

        # Final ATS score calculation (weighted average)
        ats_score = (match_score * 0.5) + (format_score * 0.3) + (keyword_match_rate * 100 * 0.2)

        return round(min(ats_score, 100), 2)

    def suggest_improvements(self, resume_text, jd_text):
        """Suggest resume improvements based on JD analysis"""
        missing_skills = self.missing_keywords(resume_text, jd_text)

        # Basic suggestions
        suggestions = []

        if missing_skills:
            if len(missing_skills) > 5:
                top_missing = missing_skills[:5]
                suggestions.append(f"Add these key missing skills to your resume: {', '.join(top_missing)} and {len(missing_skills) - 5} others")
            else:
                suggestions.append(f"Add these missing skills to your resume: {', '.join(missing_skills)}")

        # Check for quantifiable achievements
        if not re.search(r'\d+%|\d+\s*x|\bincreased\b|\bimproved\b|\breduced\b|\bsaved\b|\bgenerated\b', resume_text.lower()):
            suggestions.append("Add quantifiable achievements with metrics (%, numbers, etc.) to showcase your impact")

        # Check for action verbs
        action_verbs = ["led", "managed", "developed", "created", "implemented", "designed", "analyzed",
                        "launched", "achieved", "coordinated", "negotiated", "represented"]
        found_verbs = [verb for verb in action_verbs if verb in self.preprocess_text(resume_text)]
        if len(found_verbs) < 3:
            suggestions.append("Use more action verbs like 'developed', 'implemented', or 'managed' to describe your experiences")

        # Check resume length
        word_count = len(resume_text.split())
        if word_count < 300:
            suggestions.append("Your resume seems too short. Consider adding more details about your experiences and skills")
        elif word_count > 1000:
            suggestions.append("Your resume may be too long. Try to be more concise while maintaining key information")

        # Check for job title alignment
        jd_title = ""
        title_match = re.search(r'^(.+?)(?:\n|$)', jd_text.strip())
        if title_match:
            jd_title = title_match.group(1).lower()
            if not any(word in resume_text.lower() for word in jd_title.split()):
                suggestions.append(f"Consider aligning your job titles or summary to match the target role: '{jd_title}'")

        # If no improvements needed
        if not suggestions:
            suggestions.append("Your resume is well aligned with the job description")

        return suggestions

    def analyze_data(self, resume_text, jd_text):
        """Perform data analysis on resume and JD match"""
        match_score = self.calculate_match_score(resume_text, jd_text)
        ats_score = self.calculate_ats_score(resume_text, jd_text)
        missing_skills = self.missing_keywords(resume_text, jd_text)

        # Skills analysis
        resume_skills = self.extract_skills(resume_text)
        jd_skills = self.extract_skills(jd_text)

        matching_skills = [skill for skill in resume_skills if skill in jd_skills]
        extra_skills = [skill for skill in resume_skills if skill not in jd_skills]

        skill_match_rate = len(matching_skills) / len(jd_skills) if jd_skills else 0

        # Count word frequency for key terms
        resume_words = self.preprocess_text(resume_text).split()
        jd_words = self.preprocess_text(jd_text).split()

        # Get most common words (excluding stopwords)
        resume_word_freq = {word: resume_words.count(word) for word in set(resume_words) if word not in self.stop_words}
        jd_word_freq = {word: jd_words.count(word) for word in set(jd_words) if word not in self.stop_words}

        # Sort by frequency
        resume_top_words = sorted(resume_word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        jd_top_words = sorted(jd_word_freq.items(), key=lambda x: x[1], reverse=True)[:10]

        analysis_results = {
            "match_score": match_score,
            "ats_score": ats_score,
            "missing_skills": missing_skills,
            "matching_skills": matching_skills,
            "extra_skills": extra_skills,
            "skill_match_rate": round(skill_match_rate * 100, 2),
            "suggestions": self.suggest_improvements(resume_text, jd_text),
            "resume_word_count": len(resume_text.split()),
            "jd_word_count": len(jd_text.split()),
            "resume_top_words": resume_top_words,
            "jd_top_words": jd_top_words
        }

        return analysis_results

    def generate_report(self, resume_text, jd_text):
        """Generate a complete analysis report"""
        analysis = self.analyze_data(resume_text, jd_text)

        report = {
            "summary": {
                "match_score": analysis["match_score"],
                "ats_score": analysis["ats_score"],
                "skill_match_rate": analysis["skill_match_rate"]
            },
            "skills_analysis": {
                "matching_skills": analysis["matching_skills"],
                "missing_skills": analysis["missing_skills"],
                "extra_skills": analysis["extra_skills"],
                "total_jd_skills": len(self.extract_skills(jd_text))
            },
            "improvements": analysis["suggestions"],
            "details": {
                "resume_length": analysis["resume_word_count"],
                "jd_length": analysis["jd_word_count"],
                "resume_top_words": analysis["resume_top_words"],
                "jd_top_words": analysis["jd_top_words"]
            }
        }

        return report

    def close(self):
        """Close database connections"""
        self.skills_db.close()

def extract_text_from_pdf(file_bytes):
    """Extract text from a PDF file"""
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

def extract_text_from_docx(file_bytes):
    """Extract text from a DOCX file"""
    doc = docx.Document(io.BytesIO(file_bytes))
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

def extract_text_from_file(uploaded_file):
    """Extract text based on file type"""
    if uploaded_file is None:
        return ""

    file_bytes = uploaded_file.getvalue()
    file_type = uploaded_file.type

    if file_type == 'application/pdf':
        return extract_text_from_pdf(file_bytes)
    elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        return extract_text_from_docx(file_bytes)
    elif file_type == 'text/plain':
        return uploaded_file.getvalue().decode('utf-8')
    else:
        return "Unsupported file format. Please upload a PDF, DOCX, or TXT file."

def create_skill_match_chart(matching_skills, missing_skills, extra_skills):
    """Create a skill match visualization chart"""
    # Prepare data
    categories = ['Matching Skills', 'Missing Skills', 'Extra Skills']
    values = [len(matching_skills), len(missing_skills), len(extra_skills)]
    colors = ['#4CAF50', '#F44336', '#2196F3']

    # Create the figure
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(categories, values, color=colors)

    # Add labels and values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height}', ha='center', va='bottom')

    ax.set_title('Skills Analysis')
    ax.set_ylabel('Number of Skills')

    # Ensure the y-axis starts at 0
    ax.set_ylim(0, max(values) * 1.2)

    return fig

def create_match_gauge(match_score, ats_score):
    """Create gauge charts for match and ATS scores"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Gauge Colors
    gauge_colors = ['#FF5733', '#FFC300', '#4CAF50'] # Red, Yellow, Green

    def create_gauge(ax, score, title):
        # Determine color based on score
        if score < 50:
            color = gauge_colors[0]
        elif score < 75:
            color = gauge_colors[1]
        else:
            color = gauge_colors[2]

        # Create the gauge
        ax.pie([score, 100-score],
               colors=[color, '#f0f0f0'],
               startangle=90,
               counterclock=False,
               wedgeprops={'width': 0.4})

        ax.text(0, 0, f"{score}%", ha='center', va='center', fontsize=24, fontweight='bold')
        ax.set_title(title)
        ax.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.

    create_gauge(ax1, match_score, "Match Score")
    create_gauge(ax2, ats_score, "ATS Score")

    plt.tight_layout()
    return fig

def create_word_cloud_comparison(jd_words, resume_words):
    """Create a chart comparing top words"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # JD Top Words
    jd_words_dict = dict(jd_words)
    jd_names = list(jd_words_dict.keys())
    jd_values = list(jd_words_dict.values())
    # Reverse for highest frequency at top
    jd_names.reverse()
    jd_values.reverse()

    # Resume Top Words
    resume_words_dict = dict(resume_words)
    resume_names = list(resume_words_dict.keys())
    resume_values = list(resume_words_dict.values())
    # Reverse for highest frequency at top
    resume_names.reverse()
    resume_values.reverse()


    # Plot JD words
    ax1.barh(jd_names, jd_values, color='#2196F3')
    ax1.set_title('Top JD Words')
    ax1.set_xlabel('Frequency')


    # Plot Resume words
    ax2.barh(resume_names, resume_values, color='#4CAF50')
    ax2.set_title('Top Resume Words')
    ax2.set_xlabel('Frequency')

    plt.tight_layout()
    return fig


# Streamlit UI
st.set_page_config(page_title="Resume-JD Matcher", layout="wide")

st.title("Resume-Job Description Matcher and Analyzer")

st.markdown("""
Upload your resume and a job description (PDF, DOCX, or TXT) to get a detailed analysis
of how well they match, including a match score, ATS compatibility score, missing skills,
and suggested improvements.
""")

# File uploaders
st.subheader("Upload Files")
col1, col2 = st.columns(2)

with col1:
    resume_file = st.file_uploader("Upload Your Resume", type=['pdf', 'docx', 'txt'])

with col2:
    jd_file = st.file_uploader("Upload Job Description", type=['pdf', 'docx', 'txt'])

resume_text = ""
jd_text = ""

if resume_file is not None:
    resume_text = extract_text_from_file(resume_file)
    if "Unsupported file format" in resume_text:
        st.error(resume_text)
        resume_text = ""
    else:
        st.success("Resume uploaded and text extracted.")

if jd_file is not None:
    jd_text = extract_text_from_file(jd_file)
    if "Unsupported file format" in jd_text:
        st.error(jd_text)
        jd_text = ""
    else:
        st.success("Job Description uploaded and text extracted.")

if resume_text and jd_text:
    if st.button("Analyze Match"):
        with st.spinner("Analyzing..."):
            matcher = JobFitMatcher()
            report = matcher.generate_report(resume_text, jd_text)
            matcher.close() # Close database connection

        st.subheader("Analysis Report")

        # Summary Section
        st.write("### Summary Scores")
        summary_col1, summary_col2 = st.columns(2)
        with summary_col1:
             st.metric("Match Score", f"{report['summary']['match_score']}%")
        with summary_col2:
             st.metric("ATS Score", f"{report['summary']['ats_score']}%")

        # Display Gauge Charts
        gauge_fig = create_match_gauge(report['summary']['match_score'], report['summary']['ats_score'])
        st.pyplot(gauge_fig)


        st.write("### Skills Analysis")
        st.write(f"**Skill Match Rate:** {report['summary']['skill_match_rate']}% (Based on {report['skills_analysis']['total_jd_skills']} skills found in JD)")

        # Display Skill Match Chart
        skill_match_fig = create_skill_match_chart(
            report['skills_analysis']['matching_skills'],
            report['skills_analysis']['missing_skills'],
            report['skills_analysis']['extra_skills']
        )
        st.pyplot(skill_match_fig)

        st.write(f"**Matching Skills:** {', '.join(report['skills_analysis']['matching_skills']) if report['skills_analysis']['matching_skills'] else 'None Found'}")
        st.write(f"**Missing Skills:** {', '.join(report['skills_analysis']['missing_skills']) if report['skills_analysis']['missing_skills'] else 'None Found'}")
        st.write(f"**Extra Skills (In Resume but not JD):** {', '.join(report['skills_analysis']['extra_skills']) if report['skills_analysis']['extra_skills'] else 'None Found'}")

        st.write("### Improvement Suggestions")
        for suggestion in report['improvements']:
            st.info(suggestion)

        st.write("### Detailed Breakdown")
        st.write(f"**Resume Word Count:** {report['details']['resume_length']}")
        st.write(f"**JD Word Count:** {report['details']['jd_length']}")

        st.write("#### Top Words Comparison")
        # Display Word Cloud Comparison Chart
        word_comp_fig = create_word_cloud_comparison(
            report['details']['jd_top_words'],
            report['details']['resume_top_words']
        )
        st.pyplot(word_comp_fig)

        st.write("---")
        st.write("Analysis Complete.")

elif resume_file is None and jd_file is None:
    st.info("Please upload both your resume and a job description to analyze the match.")
else:
    st.warning("Please upload both files to proceed with the analysis.")