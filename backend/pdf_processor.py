
import pymupdf
import re


# ==========================================
# 1. KNOWN DOCUMENT SECTION HEADINGS
# ==========================================

SECTION_HEADINGS = {
    "education": "EDUCATION",
    "academic": "EDUCATION",
    "academic background": "EDUCATION",

    "skills": "SKILLS",
    "technical skills": "SKILLS",
    "technical expertise": "SKILLS",

    "experience": "EXPERIENCE",
    "work experience": "EXPERIENCE",
    "professional experience": "EXPERIENCE",
    "internship": "EXPERIENCE",

    "projects": "PROJECTS",
    "personal projects": "PROJECTS",
    "academic projects": "PROJECTS",

    "summary": "SUMMARY",
    "profile": "SUMMARY",
    "professional summary": "SUMMARY",
    "objective": "SUMMARY",

    "certifications": "CERTIFICATIONS",
    "certificates": "CERTIFICATIONS",

    "achievements": "ACHIEVEMENTS",
    "accomplishments": "ACHIEVEMENTS",

    "contact": "CONTACT",
    "contact information": "CONTACT"
}


# ==========================================
# 2. EXTRACT TEXT FROM PDF
# ==========================================

def extract_text_from_pdf(file_path):

    document = pymupdf.open(file_path)

    pages = []

    try:

        for page_number, page in enumerate(document):

            # Extract text while preserving line breaks.
            text = page.get_text("text", sort=True)

            if text.strip():

                pages.append({
                    "page": page_number + 1,
                    "text": text
                })

    finally:

        document.close()

    return pages


# ==========================================
# 3. IDENTIFY SECTION HEADINGS
# ==========================================

def identify_heading(line):

    cleaned = line.strip()

    # Remove bullets and common heading punctuation.
    cleaned = re.sub(
        r'^[•●▪\-*]+\s*',
        '',
        cleaned
    )

    cleaned = cleaned.strip(" :|.-")

    normalized = cleaned.lower()

    if normalized in SECTION_HEADINGS:
        return SECTION_HEADINGS[normalized]

    return None


# ==========================================
# 4. SPLIT PDF INTO DOCUMENT SECTIONS
# ==========================================

def extract_sections(pages):

    sections = []

    current_section = "GENERAL"
    current_text = []
    current_page = None

    def save_current_section():

        nonlocal current_text

        content = "\n".join(current_text).strip()

        if content:

            sections.append({
                "section": current_section,
                "page": current_page,
                "text": content
            })

        current_text = []

    for page in pages:

        lines = page["text"].splitlines()

        for line in lines:

            cleaned = line.strip()

            if not cleaned:
                continue

            heading = identify_heading(cleaned)

            if heading:

                save_current_section()

                current_section = heading
                current_page = page["page"]

            else:

                if current_page is None:
                    current_page = page["page"]

                current_text.append(cleaned)

        # Preserve content when moving to another page.
        save_current_section()

        current_section = "GENERAL"
        current_page = None

    save_current_section()

    return sections


# ==========================================
# 5. SPLIT LARGE TEXT INTO SMALL CHUNKS
# ==========================================

def split_into_chunks(
    text,
    section,
    page,
    chunk_size=120,
    overlap=20
):

    words = text.split()

    chunks = []

    if not words:
        return chunks

    start = 0

    while start < len(words):

        end = min(start + chunk_size, len(words))

        chunk_text = " ".join(words[start:end])

        chunks.append({

            "page": page,

            "section": section,

            "text": (
                f"Section: {section}\n"
                f"{chunk_text}"
            )

        })

        if end == len(words):
            break

        start = end - overlap

    return chunks


# ==========================================
# 6. CREATE SMART DOCUMENT CHUNKS
# ==========================================

def create_chunks(pages, chunk_size=120, overlap=20):

    sections = extract_sections(pages)

    chunks = []

    for section in sections:

        section_chunks = split_into_chunks(

            text=section["text"],

            section=section["section"],

            page=section["page"],

            chunk_size=chunk_size,

            overlap=overlap

        )

        chunks.extend(section_chunks)

    return chunks
