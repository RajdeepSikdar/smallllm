
import os
import re
import tempfile
import torch

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from tokenizer import SimpleTokenizer
from model import GPTLanguageModel
from pdf_processor import extract_text_from_pdf, create_chunks


# ==========================================
# 1. FASTAPI APPLICATION
# ==========================================

app = FastAPI(title="MiniDocLLM - PDF Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# 2. LOAD YOUR CUSTOM GPT MODEL
# ==========================================

device = torch.device("cpu")

checkpoint = torch.load(
    "my_llm.pth",
    map_location=device,
    weights_only=True
)

tokenizer = SimpleTokenizer(
    " ".join(checkpoint["words"])
)

model = GPTLanguageModel(**checkpoint["config"])

model.load_state_dict(checkpoint["model_state"])

model.to(device)
model.eval()


# ==========================================
# 3. DOCUMENT STORAGE
# ==========================================

document_pages = []
document_chunks = []

vectorizer = None
document_vectors = None

uploaded_filename = None


# ==========================================
# 4. REQUEST MODELS
# ==========================================

class ChatRequest(BaseModel):
    prompt: str


class QuestionRequest(BaseModel):
    question: str


# ==========================================
# 5. QUESTION INTENT DETECTION
# ==========================================

def detect_intent(question):

    q = question.lower()

    if any(word in q for word in [
        "education",
        "study",
        "studied",
        "college",
        "university",
        "degree",
        "school",
        "graduate",
        "graduated",
        "academic"
    ]):
        return "education"

    if any(word in q for word in [
        "experience",
        "intern",
        "internship",
        "worked",
        "work",
        "company",
        "employment",
        "job"
    ]):
        return "experience"

    if any(word in q for word in [
        "skill",
        "skills",
        "technology",
        "technologies",
        "programming",
        "language",
        "tools"
    ]):
        return "skills"

    if any(word in q for word in [
        "project",
        "projects",
        "developed",
        "built",
        "created"
    ]):
        return "projects"

    if any(word in q for word in [
        "cgpa",
        "gpa",
        "marks",
        "grade",
        "percentage",
        "score"
    ]):
        return "grades"

    if any(word in q for word in [
        "certification",
        "certificate",
        "certified"
    ]):
        return "certifications"

    return "general"


# ==========================================
# 6. QUESTION EXPANSION
# ==========================================

def expand_question(question):

    intent = detect_intent(question)

    expansions = {

        "education": (
            "education academic qualification "
            "university college institute "
            "bachelor degree BTech engineering "
            "course institution attended"
        ),

        "experience": (
            "professional experience employment "
            "internship company organization "
            "designation position work history"
        ),

        "skills": (
            "technical skills programming languages "
            "technologies tools software frameworks"
        ),

        "projects": (
            "projects development application system "
            "project description built developed"
        ),

        "grades": (
            "academic score CGPA GPA grade percentage marks"
        ),

        "certifications": (
            "certifications courses training credentials"
        ),

        "general": ""
    }

    return question + " " + expansions[intent]


# ==========================================
# 7. SPLIT TEXT INTO SENTENCES AND LINES
# ==========================================

def split_into_sentences(text):

    parts = re.split(
        r'\n+|(?<=[.!?])\s+|[•●▪]',
        text
    )

    sentences = []

    for part in parts:

        cleaned = re.sub(
            r'\s+',
            ' ',
            part
        ).strip()

        if len(cleaned) >= 5:
            sentences.append(cleaned)

    return sentences


# ==========================================
# 8. BUILD CANDIDATES FROM ALL PDF PAGES
# ==========================================

def get_all_candidates():

    candidates = []

    for page in document_pages:

        lines = [
            re.sub(r'\s+', ' ', line).strip()
            for line in page["text"].splitlines()
            if line.strip()
        ]

        # Individual lines and sentences.
        for sentence in split_into_sentences(page["text"]):

            candidates.append({
                "text": sentence,
                "page": page["page"]
            })

        # Combine nearby lines.
        # This helps when a degree and university are
        # split across multiple PDF lines.

        for i in range(len(lines)):

            for window in [2, 3]:

                combined = " ".join(
                    lines[i:i + window]
                ).strip()

                if combined:

                    candidates.append({
                        "text": combined,
                        "page": page["page"]
                    })

    # Remove duplicate candidates.
    unique = {}
    for candidate in candidates:

        key = (
            candidate["page"],
            candidate["text"].lower()
        )

        unique[key] = candidate

    return list(unique.values())


# ==========================================
# 9. SCORE ANSWER CANDIDATES
# ==========================================

def score_candidate(question, candidate, similarity):

    intent = detect_intent(question)

    text = candidate["text"]
    lower = text.lower()

    score = float(similarity)

    # Section heading bonus.
    if intent.upper() in lower:
        score += 0.25

    # EDUCATION
    if intent == "education":

        # Institution indicators.
        if re.search(
            r'\b(university|college|institute)\b',
            lower
        ):
            score += 3.0

        # Degree indicators.
        if re.search(
            r'\b(bachelor|b\.?tech|b\.?e\.?|master|degree|bsc|bca|mca|mba)\b',
            lower
        ):
            score += 3.0

        if "computer science" in lower:
            score += 1.5

        # Study dates.
        if re.search(
            r'\b(19|20)\d{2}\b',
            text
        ):
            score += 0.5

        # Penalize unrelated summary sentences.
        if "hands-on experience" in lower:
            score -= 2.0

    # EXPERIENCE
    elif intent == "experience":

        if re.search(
            r'\b(intern|internship|experience|employed|worked)\b',
            lower
        ):
            score += 2.5

        if re.search(
            r'\b(ltd|limited|technologies|solutions|inc)\b',
            lower
        ):
            score += 0.5

    # SKILLS
    elif intent == "skills":

        if re.search(
            r'\b(java|python|javascript|sql|react|angular|fastapi|c\+\+)\b',
            lower
        ):
            score += 1.0

    # PROJECTS
    elif intent == "projects":

        if re.search(
            r'\b(project|application|dashboard|system)\b',
            lower
        ):
            score += 1.5

        if re.search(
            r'\b(built|developed|created|designed)\b',
            lower
        ):
            score += 1.0

    # GRADES
    elif intent == "grades":

        if re.search(
            r'\b(cgpa|gpa|percentage|marks|grade)\b',
            lower
        ):
            score += 3.0

    return score


# ==========================================
# 10. FIND THE BEST ANSWER
# ==========================================

def extract_best_answer(question):

    candidates = get_all_candidates()

    if not candidates:
        return None, []

    expanded_question = expand_question(question)

    # Create candidate vectors.
    candidate_texts = [
        candidate["text"]
        for candidate in candidates
    ]

    candidate_vectors = vectorizer.transform(
        candidate_texts
    )

    question_vector = vectorizer.transform([
        expanded_question
    ])

    similarities = cosine_similarity(
        question_vector,
        candidate_vectors
    )[0]

    scored_candidates = []

    for i, candidate in enumerate(candidates):

        score = score_candidate(
            question,
            candidate,
            similarities[i]
        )

        scored_candidates.append({
            "text": candidate["text"],
            "page": candidate["page"],
            "score": score
        })

    scored_candidates.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    # Select the strongest relevant candidates.
    selected = []

    seen = set()

    intent = detect_intent(question)

    # Require a meaningful relevance signal.
    minimum_score = {
        "education": 1.0,
        "experience": 0.8,
        "skills": 0.3,
        "projects": 0.5,
        "grades": 1.0,
        "certifications": 0.5,
        "general": 0.05
    }.get(intent, 0.05)

    for item in scored_candidates:

        text = item["text"].strip()

        key = text.lower()

        if key in seen:
            continue

        if item["score"] < minimum_score:
            continue

        seen.add(key)

        selected.append(item)

        if len(selected) >= 3:
            break

    if not selected:
        return None, []

    # Avoid combining unrelated long sentences.
    # Prefer concise answers.

    if intent == "education":

        education_candidates = [
            item for item in selected
            if re.search(
                r'\b(university|college|institute|bachelor|b\.?tech|degree)\b',
                item["text"],
                re.IGNORECASE
            )
        ]

        if education_candidates:
            selected = education_candidates[:2]

    answer = " ".join(
        item["text"]
        for item in selected
    )

    sources = [
        {
            "page": item["page"],
            "text": item["text"],
            "score": round(item["score"], 3)
        }
        for item in selected
    ]

    return answer, sources


# ==========================================
# 11. HOME ROUTE
# ==========================================

@app.get("/")
def home():

    return {
        "message": "MiniDocLLM API is running!",
        "status": "online"
    }


# ==========================================
# 12. UPLOAD PDF
# ==========================================

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    global document_pages
    global document_chunks
    global vectorizer
    global document_vectors
    global uploaded_filename

    if not file.filename or not file.filename.lower().endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail="Please upload a valid PDF file."
        )

    contents = await file.read()

    if not contents:

        raise HTTPException(
            status_code=400,
            detail="The uploaded PDF is empty."
        )

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False
        ) as temp_file:

            temp_file.write(contents)
            temp_path = temp_file.name

        # Extract pages.
        pages = extract_text_from_pdf(temp_path)

        if not pages:

            raise HTTPException(
                status_code=400,
                detail="No readable text found in this PDF."
            )

        # Create chunks.
        chunks = create_chunks(pages)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Could not create document chunks."
            )

        # Build TF-IDF index.
        new_vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True
        )

        new_vectors = new_vectorizer.fit_transform(
            [chunk["text"] for chunk in chunks]
        )

        # Save the new document.
        document_pages = pages
        document_chunks = chunks

        vectorizer = new_vectorizer
        document_vectors = new_vectors

        uploaded_filename = file.filename

        return {
            "message": "PDF uploaded successfully!",
            "filename": uploaded_filename,
            "pages": len(document_pages),
            "chunks": len(document_chunks),
            "status": "ready"
        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"PDF processing failed: {str(error)}"
        )

    finally:

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

        await file.close()


# ==========================================
# 13. ASK QUESTIONS ABOUT PDF
# ==========================================

@app.post("/ask")
def ask_question(request: QuestionRequest):

    if not document_pages or vectorizer is None:

        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF first."
        )

    question = request.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Please enter a question."
        )

    try:

        answer, sources = extract_best_answer(question)

        if not answer:

            return {
                "question": question,
                "answer": "I could not find a reliable answer in the uploaded PDF.",
                "sources": [],
                "filename": uploaded_filename
            }

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "filename": uploaded_filename,
            "method": "TF-IDF + full-document sentence ranking"
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Question processing failed: {str(error)}"
        )


# ==========================================
# 14. YOUR CUSTOM GPT CHAT
# ==========================================

@app.post("/chat")
def chat(request: ChatRequest):

    try:

        prompt = request.prompt.strip()

        if not prompt:

            raise HTTPException(
                status_code=400,
                detail="Please enter a prompt."
            )

        tokens = tokenizer.encode(prompt)

        if not tokens:

            raise HTTPException(
                status_code=400,
                detail="Please enter a valid prompt."
            )

        x = torch.tensor(
            [tokens],
            dtype=torch.long,
            device=device
        )

        with torch.no_grad():

            output = model.generate(
                x,
                max_new_tokens=20,
                temperature=0.8
            )

        generated = output[0, len(tokens):].tolist()

        response = tokenizer.decode(generated)

        return {
            "prompt": prompt,
            "response": response,
            "model": "Custom PyTorch Transformer"
        }

    except KeyError:

        raise HTTPException(
            status_code=400,
            detail="Your prompt contains words outside the vocabulary."
        )

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"LLM generation failed: {str(error)}"
        )
