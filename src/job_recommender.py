"""
JobSentinel - Adzuna Job Recommendation

Searches Adzuna for real job opportunities and returns
normalized job dictionaries for the recommendation system.
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

load_dotenv(ENV_PATH)


ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY")

ADZUNA_COUNTRY = "in"

ADZUNA_SEARCH_URL = (
    f"https://api.adzuna.com/v1/api/jobs/"
    f"{ADZUNA_COUNTRY}/search/1"
)


# ============================================================
# HELPERS
# ============================================================

def clean_text(value):
    if value is None:
        return ""

    return str(value).strip()


def build_keywords(job_data):
    """
    Build Adzuna search keyword.

    Priority:
        title
        function
        department
        industry
    """

    if not isinstance(job_data, dict):
        return ""

    title = clean_text(
        job_data.get("title")
    )

    if title:
        return title

    function = clean_text(
        job_data.get("function")
    )

    if function:
        return function

    department = clean_text(
        job_data.get("department")
    )

    if department:
        return department

    industry = clean_text(
        job_data.get("industry")
    )

    return industry


def extract_location(job_data):
    if not isinstance(job_data, dict):
        return ""

    return clean_text(
        job_data.get("location")
    )


# ============================================================
# ADZUNA SEARCH
# ============================================================

def search_adzuna(
    job_data,
    results_per_page=10
):

    # --------------------------------------------------------
    # CHECK API CREDENTIALS
    # --------------------------------------------------------

    if not ADZUNA_APP_ID:
        return {
            "success": False,
            "jobs": [],
            "total_count": 0,
            "message": "ADZUNA_APP_ID is not configured."
        }

    if not ADZUNA_APP_KEY:
        return {
            "success": False,
            "jobs": [],
            "total_count": 0,
            "message": "ADZUNA_APP_KEY is not configured."
        }


    # --------------------------------------------------------
    # SEARCH VALUES
    # --------------------------------------------------------

    keywords = build_keywords(job_data)
    location = extract_location(job_data)

    if not keywords:
        return {
            "success": False,
            "jobs": [],
            "total_count": 0,
            "message": "No job title or search keyword was provided."
        }


    # --------------------------------------------------------
    # PARAMETERS
    # --------------------------------------------------------

    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_APP_KEY,
        "results_per_page": int(results_per_page),
        "what": keywords,
        "content-type": "application/json"
    }

    # Only send location if user entered one.
    if location:
        params["where"] = location


    # --------------------------------------------------------
    # DEBUG
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("JOBSENTINEL - ADZUNA JOB SEARCH")
    print("=" * 70)

    print("URL      :", ADZUNA_SEARCH_URL)
    print("Keywords :", keywords)
    print("Location :", location if location else "Any")
    print("Results  :", results_per_page)
    print()


    # --------------------------------------------------------
    # REQUEST
    # --------------------------------------------------------

    try:

        response = requests.get(
            ADZUNA_SEARCH_URL,
            params=params,
            timeout=20
        )

        print(
            "HTTP Status:",
            response.status_code
        )

        response.raise_for_status()

    except requests.exceptions.Timeout:

        print("ERROR: Request timed out.")

        return {
            "success": False,
            "jobs": [],
            "total_count": 0,
            "message": "Adzuna request timed out."
        }

    except requests.exceptions.HTTPError as error:

        print(
            "HTTP ERROR:",
            error
        )

        print(
            "Response:",
            response.text[:1000]
        )

        return {
            "success": False,
            "jobs": [],
            "total_count": 0,
            "message": f"Adzuna HTTP error: {error}"
        }

    except requests.exceptions.RequestException as error:

        print(
            "REQUEST ERROR:",
            error
        )

        return {
            "success": False,
            "jobs": [],
            "total_count": 0,
            "message": f"Adzuna request failed: {error}"
        }


    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    try:

        data = response.json()

    except ValueError:

        print("ERROR: Adzuna returned invalid JSON.")
        print(response.text[:1000])

        return {
            "success": False,
            "jobs": [],
            "total_count": 0,
            "message": "Adzuna returned invalid JSON."
        }


    # --------------------------------------------------------
    # RAW RESPONSE CHECK
    # --------------------------------------------------------

    print(
        "Response type:",
        type(data).__name__
    )

    if not isinstance(data, dict):

        print(
            "Unexpected response:",
            str(data)[:1000]
        )

        return {
            "success": False,
            "jobs": [],
            "total_count": 0,
            "message": "Unexpected Adzuna response format."
        }


    # --------------------------------------------------------
    # EXTRACT RESULTS
    # --------------------------------------------------------

    total_count = data.get(
        "count",
        0
    )

    raw_jobs = data.get(
        "results",
        []
    )


    print(
        "Total jobs found:",
        total_count
    )

    print(
        "Raw jobs returned:",
        len(raw_jobs)
    )


    # --------------------------------------------------------
    # NORMALIZE JOBS
    # --------------------------------------------------------

    jobs = []

    for job in raw_jobs:

        if not isinstance(job, dict):
            continue


        # COMPANY

        company_data = job.get(
            "company",
            {}
        )

        if isinstance(company_data, dict):

            company_name = clean_text(
                company_data.get(
                    "display_name",
                    ""
                )
            )

        else:

            company_name = clean_text(
                company_data
            )


        # LOCATION

        location_data = job.get(
            "location",
            {}
        )

        if isinstance(location_data, dict):

            job_location = clean_text(
                location_data.get(
                    "display_name",
                    ""
                )
            )

        else:

            job_location = clean_text(
                location_data
            )


        # CATEGORY

        category_data = job.get(
            "category",
            {}
        )

        if isinstance(category_data, dict):

            category = clean_text(
                category_data.get(
                    "label",
                    ""
                )
            )

        else:

            category = ""


        # ADD JOB

        normalized_job = {

            "title": clean_text(
                job.get(
                    "title",
                    "Job Opportunity"
                )
            ),

            "company": (
                company_name
                if company_name
                else "Company not specified"
            ),

            "location": (
                job_location
                if job_location
                else "Location not specified"
            ),

            "salary_min": job.get(
                "salary_min"
            ),

            "salary_max": job.get(
                "salary_max"
            ),

            "salary_is_predicted": job.get(
                "salary_is_predicted",
                False
            ),

            "contract_type": clean_text(
                job.get(
                    "contract_type",
                    ""
                )
            ),

            "contract_time": clean_text(
                job.get(
                    "contract_time",
                    ""
                )
            ),

            "category": category,

            "description": clean_text(
                job.get(
                    "description",
                    ""
                )
            ),

            "created": clean_text(
                job.get(
                    "created",
                    ""
                )
            ),

            "redirect_url": clean_text(
                job.get(
                    "redirect_url",
                    ""
                )
            ),

            # Keep these because similarity ranking
            # can use them if available.
            "id": clean_text(
                job.get(
                    "id",
                    ""
                )
            ),

            "adref": clean_text(
                job.get(
                    "adref",
                    ""
                )
            )
        }

        jobs.append(
            normalized_job
        )


    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print(
        "Normalized jobs:",
        len(jobs)
    )

    if jobs:

        print()
        print(
            "Successfully retrieved",
            len(jobs),
            "jobs."
        )

        for index, job in enumerate(
            jobs[:3],
            start=1
        ):

            print(
                f"{index}. "
                f"{job['title']} | "
                f"{job['company']} | "
                f"{job['location']}"
            )

        print("=" * 70)

        return {
            "success": True,
            "jobs": jobs,
            "total_count": total_count,
            "message": (
                f"Found {len(jobs)} job opportunities."
            )
        }


    print()
    print(
        "Adzuna returned no usable jobs."
    )

    print("=" * 70)

    return {
        "success": True,
        "jobs": [],
        "total_count": total_count,
        "message": (
            "Adzuna responded successfully, "
            "but no usable jobs were returned."
        )
    }


# ============================================================
# PUBLIC FUNCTION USED BY APP
# ============================================================

def find_similar_jobs(
    job_data,
    results_per_page=10
):

    return search_adzuna(
        job_data,
        results_per_page
    )


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("JobSentinel - Adzuna API Test")
    print("=" * 70)


    test_job = {
        "title": "developer",
        "location": ""
    }


    result = search_adzuna(
        test_job,
        5
    )


    print()
    print("Success:", result["success"])
    print("Total:", result["total_count"])
    print("Jobs:", len(result["jobs"]))
    print("Message:", result["message"])


    print()
    print("=" * 70)