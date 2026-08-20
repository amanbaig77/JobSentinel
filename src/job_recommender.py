"""
JobSentinel - Job Recommendation Module

Uses the Adzuna Jobs API to find alternative jobs.

The recommendation system is separate from fraud detection.
"""

import os
import requests

from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

ENV_PATH = os.path.join(
    PROJECT_ROOT,
    ".env"
)

load_dotenv(
    ENV_PATH
)


ADZUNA_APP_ID = os.getenv(
    "ADZUNA_APP_ID"
)

ADZUNA_APP_KEY = os.getenv(
    "ADZUNA_APP_KEY"
)


ADZUNA_COUNTRY = "in"

ADZUNA_BASE_URL = (
    "https://api.adzuna.com/v1/api/jobs/"
    f"{ADZUNA_COUNTRY}/search"
)


# ============================================================
# HELPERS
# ============================================================

def clean_text(value):

    if value is None:

        return ""

    return str(value).strip()


def build_keywords(
    job_data
):

    title = clean_text(
        job_data.get(
            "title"
        )
    )

    if title:

        return title


    function = clean_text(
        job_data.get(
            "function"
        )
    )

    if function:

        return function


    department = clean_text(
        job_data.get(
            "department"
        )
    )

    if department:

        return department


    industry = clean_text(
        job_data.get(
            "industry"
        )
    )

    return industry


def extract_location(
    job_data
):

    return clean_text(
        job_data.get(
            "location"
        )
    )


def make_search_terms(
    job_data
):

    title = clean_text(
        job_data.get(
            "title"
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

    industry = clean_text(
        job_data.get(
            "industry"
        )
    )


    terms = []


    if title:

        terms.append(
            title
        )


        # Remove common location/work-mode words
        simplified = title.lower()

        simplified = (
            simplified
            .replace(
                "work from home",
                ""
            )
            .replace(
                "remote",
                ""
            )
        )

        simplified = " ".join(
            simplified.split()
        )

        if (
            simplified
            and simplified != title.lower()
        ):

            terms.append(
                simplified
            )


    if function:

        terms.append(
            function
        )


    if department:

        terms.append(
            department
        )


    if industry:

        terms.append(
            industry
        )


    # Remove duplicates

    final_terms = []

    for term in terms:

        if term and term not in final_terms:

            final_terms.append(
                term
            )


    return final_terms


# ============================================================
# PARSE JOB
# ============================================================

def parse_job(
    job
):

    company = job.get(
        "company",
        {}
    )

    location = job.get(
        "location",
        {}
    )

    category = job.get(
        "category",
        {}
    )


    if not isinstance(
        company,
        dict
    ):

        company = {}


    if not isinstance(
        location,
        dict
    ):

        location = {}


    if not isinstance(
        category,
        dict
    ):

        category = {}


    return {

        "title":
            clean_text(
                job.get(
                    "title",
                    "Job opportunity"
                )
            ),

        "company":
            clean_text(
                company.get(
                    "display_name"
                )
            )
            or "Company not specified",

        "location":
            clean_text(
                location.get(
                    "display_name"
                )
            )
            or "Location not specified",

        "salary_min":
            job.get(
                "salary_min"
            ),

        "salary_max":
            job.get(
                "salary_max"
            ),

        "salary_is_predicted":
            job.get(
                "salary_is_predicted",
                False
            ),

        "contract_type":
            clean_text(
                job.get(
                    "contract_type"
                )
            ),

        "contract_time":
            clean_text(
                job.get(
                    "contract_time"
                )
            ),

        "category":
            clean_text(
                category.get(
                    "label"
                )
            ),

        "description":
            clean_text(
                job.get(
                    "description"
                )
            ),

        "created":
            clean_text(
                job.get(
                    "created"
                )
            ),

        "redirect_url":
            clean_text(
                job.get(
                    "redirect_url"
                )
            )
    }


# ============================================================
# API SEARCH
# ============================================================

def search_adzuna(
    keywords,
    location="",
    results_per_page=12
):

    if not ADZUNA_APP_ID:

        return {
            "success": False,
            "jobs": [],
            "total_count": 0,
            "message":
                "ADZUNA_APP_ID is not configured."
        }


    if not ADZUNA_APP_KEY:

        return {
            "success": False,
            "jobs": [],
            "total_count": 0,
            "message":
                "ADZUNA_APP_KEY is not configured."
        }


    if not keywords:

        return {
            "success": False,
            "jobs": [],
            "total_count": 0,
            "message":
                "No job search keywords were provided."
        }


    params = {

        "app_id":
            ADZUNA_APP_ID,

        "app_key":
            ADZUNA_APP_KEY,

        "results_per_page":
            results_per_page,

        "what":
            keywords,

        "content-type":
            "application/json"
    }


    if location:

        params["where"] = location


    url = (
        f"{ADZUNA_BASE_URL}/1"
    )


    try:

        response = requests.get(
            url,
            params=params,
            timeout=20
        )

        response.raise_for_status()

        data = response.json()


    except requests.exceptions.Timeout:

        return {
            "success": False,
            "jobs": [],
            "total_count": 0,
            "message":
                "Adzuna request timed out."
        }


    except requests.exceptions.HTTPError as error:

        return {
            "success": False,
            "jobs": [],
            "total_count": 0,
            "message":
                f"Adzuna HTTP error: {error}"
        }


    except requests.exceptions.RequestException as error:

        return {
            "success": False,
            "jobs": [],
            "total_count": 0,
            "message":
                f"Adzuna request failed: {error}"
        }


    except ValueError:

        return {
            "success": False,
            "jobs": [],
            "total_count": 0,
            "message":
                "Adzuna returned invalid JSON."
        }


    total_count = data.get(
        "count",
        0
    )

    raw_jobs = data.get(
        "results",
        []
    )


    jobs = []

    for job in raw_jobs:

        jobs.append(
            parse_job(job)
        )


    return {

        "success": True,

        "jobs": jobs,

        "total_count":
            total_count,

        "message":
            (
                f"Found {len(jobs)} "
                f"job opportunities."
            )
            if jobs
            else
            (
                "Adzuna responded successfully, "
                "but no matching jobs were found."
            )
    }


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def find_similar_jobs(
    job_data,
    results_per_page=12
):

    search_terms = make_search_terms(
        job_data
    )

    location = extract_location(
        job_data
    )


    if not search_terms:

        return {
            "success": False,
            "jobs": [],
            "total_count": 0,
            "message":
                "No job title or search keywords were provided."
        }


    all_jobs = []

    total_count = 0


    # --------------------------------------------------------
    # Search using the best available keyword.
    # --------------------------------------------------------

    for keywords in search_terms:

        result = search_adzuna(
            keywords=keywords,
            location=location,
            results_per_page=results_per_page
        )


        if result["jobs"]:

            all_jobs.extend(
                result["jobs"]
            )

            total_count += (
                result["total_count"]
                or 0
            )

            break


    # --------------------------------------------------------
    # If location was too restrictive, search again without it.
    # --------------------------------------------------------

    if not all_jobs and location:

        for keywords in search_terms:

            result = search_adzuna(
                keywords=keywords,
                location="",
                results_per_page=results_per_page
            )


            if result["jobs"]:

                all_jobs.extend(
                    result["jobs"]
                )

                total_count = (
                    result["total_count"]
                    or len(all_jobs)
                )

                break


    # --------------------------------------------------------
    # Remove duplicate jobs
    # --------------------------------------------------------

    unique_jobs = []

    seen = set()


    for job in all_jobs:

        key = (
            job.get("title", ""),
            job.get("company", ""),
            job.get("location", "")
        )


        if key in seen:

            continue


        seen.add(key)

        unique_jobs.append(
            job
        )


    if not unique_jobs:

        return {

            "success": True,

            "jobs": [],

            "total_count": 0,

            "message": (
                "No similar jobs were found. "
                "Try another job title or location."
            )
        }


    return {

        "success": True,

        "jobs":
            unique_jobs,

        "total_count":
            total_count
            or len(unique_jobs),

        "message":
            (
                f"Found {len(unique_jobs)} "
                "similar job opportunities."
            )
    }