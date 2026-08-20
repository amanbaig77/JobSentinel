"""
JobSentinel - Job Similarity Ranking

Uses TF-IDF + cosine similarity to rank
real job opportunities against the submitted job.
"""

import re

from typing import Dict, List

from sklearn.feature_extraction.text import (
    TfidfVectorizer
)

from sklearn.metrics.pairwise import (
    cosine_similarity
)


# ============================================================
# CONFIG
# ============================================================

DEFAULT_RESULTS = 5


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(
    text
) -> str:

    if text is None:

        return ""

    text = str(text)

    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# SUBMITTED JOB TEXT
# ============================================================

def build_submitted_job_text(
    job_data: Dict
) -> str:

    title = clean_text(
        job_data.get(
            "title"
        )
    )

    industry = clean_text(
        job_data.get(
            "industry"
        )
    )

    function = clean_text(
        job_data.get(
            "function"
        )
    )

    department = clean_text(
        job_data.get(
            "department"
        )
    )

    employment_type = clean_text(
        job_data.get(
            "employment_type"
        )
    )

    experience = clean_text(
        job_data.get(
            "required_experience"
        )
    )

    education = clean_text(
        job_data.get(
            "required_education"
        )
    )

    location = clean_text(
        job_data.get(
            "location"
        )
    )

    description = clean_text(
        job_data.get(
            "description"
        )
    )

    requirements = clean_text(
        job_data.get(
            "requirements"
        )
    )

    benefits = clean_text(
        job_data.get(
            "benefits"
        )
    )

    company_profile = clean_text(
        job_data.get(
            "company_profile"
        )
    )


    parts = [

        # Title gets extra weight
        title,
        title,
        title,

        industry,

        function,

        department,

        employment_type,

        experience,

        education,

        location,

        description,
        description,

        requirements,
        requirements,

        benefits,

        company_profile
    ]


    return " ".join(
        part
        for part in parts
        if part
    )


# ============================================================
# EXTERNAL JOB TEXT
# ============================================================

def build_external_job_text(
    job: Dict
) -> str:

    title = clean_text(
        job.get(
            "title"
        )
    )

    company = clean_text(
        job.get(
            "company"
        )
    )

    location = clean_text(
        job.get(
            "location"
        )
    )

    description = clean_text(
        job.get(
            "description"
        )
    )

    category = clean_text(
        job.get(
            "category"
        )
    )

    contract_type = clean_text(
        job.get(
            "contract_type"
        )
    )

    contract_time = clean_text(
        job.get(
            "contract_time"
        )
    )


    parts = [

        title,
        title,

        category,

        contract_type,

        contract_time,

        location,

        description,
        description,

        company
    ]


    return " ".join(
        part
        for part in parts
        if part
    )


# ============================================================
# RANK
# ============================================================

def get_ranked_recommendations(
    job_data: Dict,
    jobs: List[Dict],
    results: int = DEFAULT_RESULTS
):

    if not jobs:

        return []


    submitted_text = (
        build_submitted_job_text(
            job_data
        )
    )


    external_texts = [

        build_external_job_text(
            job
        )

        for job in jobs
    ]


    # --------------------------------------------------------
    # Empty input protection
    # --------------------------------------------------------

    if not submitted_text.strip():

        return jobs[:results]


    if not any(
        text.strip()
        for text in external_texts
    ):

        return jobs[:results]


    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1
    )


    try:

        matrix = vectorizer.fit_transform(
            [
                submitted_text
            ]
            + external_texts
        )

    except ValueError:

        return jobs[:results]


    submitted_vector = matrix[0]

    external_vectors = matrix[1:]


    similarities = cosine_similarity(
        submitted_vector,
        external_vectors
    )[0]


    # --------------------------------------------------------
    # Attach scores
    # --------------------------------------------------------

    ranked = []


    for job, score in zip(
        jobs,
        similarities
    ):

        item = dict(
            job
        )

        item[
            "similarity"
        ] = float(
            score
        )

        ranked.append(
            item
        )


    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    ranked.sort(
        key=lambda item:
        item.get(
            "similarity",
            0
        ),
        reverse=True
    )


    return ranked[:results]