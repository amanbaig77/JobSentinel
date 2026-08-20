```python
"""
JobSentinel - Adzuna Job Recommendation Module

Supports:
- Local development using .env
- Streamlit Cloud using st.secrets
- Adzuna India API
- Job title + location search
- Normalized job results
- Similar-job recommendation support
"""

import os
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Optional Streamlit import
# ---------------------------------------------------------------------------

try:
    import streamlit as st
except ImportError:
    st = None


# ---------------------------------------------------------------------------
# Load local .env
# ---------------------------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ADZUNA_BASE_URL = "https://api.adzuna.com/v1/api/jobs/in/search"
ADZUNA_TIMEOUT = 20


# ---------------------------------------------------------------------------
# Secrets / Environment Variables
# ---------------------------------------------------------------------------

def get_config_value(name: str) -> Optional[str]:
    """
    Get configuration value.

    Priority:
    1. Streamlit Secrets
    2. Environment variables / .env
    """

    # Streamlit Cloud
    if st is not None:
        try:
            value = st.secrets.get(name)

            if value is not None:
                value = str(value).strip()

                if value:
                    return value

        except Exception:
            pass

    # Local .env / environment
    value = os.getenv(name)

    if value is not None:
        value = str(value).strip()

        if value:
            return value

    return None


ADZUNA_APP_ID = get_config_value("ADZUNA_APP_ID")
ADZUNA_APP_KEY = get_config_value("ADZUNA_APP_KEY")


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _safe_text(value: Any, default: str = "") -> str:
    """Safely convert a value to a string."""

    if value is None:
        return default

    try:
        return str(value).strip()
    except Exception:
        return default


def _safe_number(value: Any) -> Optional[float]:
    """Safely convert numeric values."""

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_nested(data: Dict[str, Any], *keys: str, default: Any = "") -> Any:
    """Safely retrieve nested dictionary values."""

    current = data

    for key in keys:

        if not isinstance(current, dict):
            return default

        current = current.get(key)

        if current is None:
            return default

    return current


# ---------------------------------------------------------------------------
# Normalize Adzuna Job
# ---------------------------------------------------------------------------

def normalize_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert an Adzuna job object into the format expected by JobSentinel.
    """

    location = _get_nested(
        job,
        "location",
        "display_name",
        default="India"
    )

    company = _get_nested(
        job,
        "company",
        "display_name",
        default="Unknown Company"
    )

    category = _get_nested(
        job,
        "category",
        "label",
        default=""
    )

    salary_min = _safe_number(job.get("salary_min"))
    salary_max = _safe_number(job.get("salary_max"))

    return {
        "title": _safe_text(
            job.get("title"),
            "Unknown Job"
        ),

        "company": _safe_text(
            company,
            "Unknown Company"
        ),

        "location": _safe_text(
            location,
            "India"
        ),

        "salary_min": salary_min,

        "salary_max": salary_max,

        "salary_is_predicted": _safe_text(
            job.get("salary_is_predicted")
        ),

        "contract_type": _safe_text(
            job.get("contract_type")
        ),

        "contract_time": _safe_text(
            job.get("contract_time")
        ),

        "category": _safe_text(
            category
        ),

        "description": _safe_text(
            job.get("description")
        ),

        "created": _safe_text(
            job.get("created")
        ),

        "redirect_url": _safe_text(
            job.get("redirect_url")
        ),

        "id": _safe_text(
            job.get("id")
        ),

        "adref": _safe_text(
            job.get("adref")
        ),
    }


# ---------------------------------------------------------------------------
# Search Adzuna
# ---------------------------------------------------------------------------

def search_adzuna(
    job_data: Optional[Dict[str, Any]] = None,
    results_per_page: int = 5
) -> Dict[str, Any]:
    """
    Search Adzuna India jobs.

    Example:

        search_adzuna(
            {
                "title": "Python Developer",
                "location": "Bangalore"
            },
            5
        )
    """

    # -----------------------------------------------------------------------
    # Credentials
    # -----------------------------------------------------------------------

    app_id = get_config_value("ADZUNA_APP_ID")
    app_key = get_config_value("ADZUNA_APP_KEY")

    if not app_id:
        return {
            "success": False,
            "jobs": [],
            "total_count": 0,
            "message": "ADZUNA_APP_ID is not configured."
        }

    if not app_key:
        return {
            "success": False,
            "jobs": [],
            "total_count": 0,
            "message": "ADZUNA_APP_KEY is not configured."
        }

    # -----------------------------------------------------------------------
    # Input
    # -----------------------------------------------------------------------

    if job_data is None:
        job_data = {}

    if not isinstance(job_data, dict):
        job_data = {}

    title = _safe_text(
        job_data.get("title")
        or job_data.get("job_title")
        or job_data.get("query")
    )

    location = _safe_text(
        job_data.get("location")
    )

    # -----------------------------------------------------------------------
    # Results limit
    # -----------------------------------------------------------------------

    try:
        results_per_page = int(results_per_page)
    except (TypeError, ValueError):
        results_per_page = 5

    results_per_page = max(
        1,
        min(results_per_page, 50)
    )

    # -----------------------------------------------------------------------
    # Adzuna endpoint
    # -----------------------------------------------------------------------

    url = f"{ADZUNA_BASE_URL}/1"

    params = {
        "app_id": app_id,
        "app_key": app_key,
        "results_per_page": results_per_page,
        "content-type": "application/json",
    }

    # Adzuna "what" parameter
    if title:
        params["what"] = title

    # Adzuna "where" parameter
    if location:
        params["where"] = location

    # -----------------------------------------------------------------------
    # Console debugging
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("JOBSENTINEL - ADZUNA JOB SEARCH")
    print("=" * 70)

    print(f"URL      : {url}")
    print(f"Keywords : {title if title else 'Any'}")
    print(f"Location : {location if location else 'Any'}")
    print(f"Results  : {results_per_page}")

    # -----------------------------------------------------------------------
    # API request
    # -----------------------------------------------------------------------

    try:

        response = requests.get(
            url,
            params=params,
            timeout=ADZUNA_TIMEOUT
        )

    except requests.exceptions.Timeout:

        print("Adzuna request timed out.")

        return {
            "success": False,
            "jobs": [],
            "total_count": 0,
            "message": "Adzuna API request timed out."
        }

    except requests.exceptions.RequestException as exc:

        print(f"Adzuna request failed: {exc}")

        return {
            "success": False,
            "jobs": [],
            "total_count": 0,
            "message": f"Adzuna API request failed: {exc}"
        }

    # -----------------------------------------------------------------------
    # HTTP status
    # -----------------------------------------------------------------------

    print(f"HTTP Status: {response.status_code}")

    if response.status_code != 200:

        error_text = response.text[:1000]

        print("Adzuna API error:")
        print(error_text)

        return {
            "success": False,
            "jobs": [],
            "total_count": 0,
            "message": (
                f"Adzuna API returned HTTP "
                f"{response.status_code}."
            )
        }

    # -----------------------------------------------------------------------
    # JSON response
    # -----------------------------------------------------------------------

    try:

        data = response.json()

    except ValueError:

        return {
            "success": False,
            "jobs": [],
            "total_count": 0,
            "message": "Adzuna returned an invalid JSON response."
        }

    print(f"Response type: {type(data).__name__}")

    # -----------------------------------------------------------------------
    # Validate response
    # -----------------------------------------------------------------------

    if not isinstance(data, dict):

        return {
            "success": False,
            "jobs": [],
            "total_count": 0,
            "message": "Unexpected Adzuna response format."
        }

    # -----------------------------------------------------------------------
    # Extract results
    # -----------------------------------------------------------------------

    total_count = data.get("count", 0)

    try:
        total_count = int(total_count)
    except (TypeError, ValueError):
        total_count = 0

    raw_jobs = data.get("results", [])

    if not isinstance(raw_jobs, list):
        raw_jobs = []

    print(f"Total jobs found: {total_count}")
    print(f"Raw jobs returned: {len(raw_jobs)}")

    # -----------------------------------------------------------------------
    # Normalize
    # -----------------------------------------------------------------------

    normalized_jobs: List[Dict[str, Any]] = []

    for raw_job in raw_jobs:

        if not isinstance(raw_job, dict):
            continue

        try:

            normalized = normalize_job(raw_job)

            # Require at least a title
            if normalized["title"]:
                normalized_jobs.append(normalized)

        except Exception as exc:

            print(f"Could not normalize job: {exc}")

    print(
        f"Normalized jobs: {len(normalized_jobs)}"
    )

    # -----------------------------------------------------------------------
    # No results
    # -----------------------------------------------------------------------

    if not normalized_jobs:

        print(
            "Adzuna responded successfully, "
            "but no matching jobs were found."
        )

        return {
            "success": True,
            "jobs": [],
            "total_count": total_count,
            "message": (
                "Adzuna responded successfully, "
                "but no matching jobs were found."
            )
        }

    # -----------------------------------------------------------------------
    # Print jobs
    # -----------------------------------------------------------------------

    print(
        f"\nSuccessfully retrieved "
        f"{len(normalized_jobs)} jobs."
    )

    for index, job in enumerate(
        normalized_jobs[:5],
        start=1
    ):

        print(
            f"{index}. "
            f"{job['title']} | "
            f"{job['company']} | "
            f"{job['location']}"
        )

    print("=" * 70)

    # -----------------------------------------------------------------------
    # Return
    # -----------------------------------------------------------------------

    return {
        "success": True,
        "jobs": normalized_jobs,
        "total_count": total_count,
        "message": (
            f"Found {len(normalized_jobs)} "
            f"job opportunities."
        )
    }


# ---------------------------------------------------------------------------
# Compatibility Function
# ---------------------------------------------------------------------------

def find_similar_jobs(
    job_data: Optional[Dict[str, Any]] = None,
    results_per_page: int = 5
) -> Dict[str, Any]:
    """
    Compatibility wrapper used by the existing JobSentinel app.

    This allows existing app.py code that calls find_similar_jobs()
    to continue working.
    """

    return search_adzuna(
        job_data=job_data,
        results_per_page=results_per_page
    )


# ---------------------------------------------------------------------------
# Simple Direct Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("JobSentinel - Adzuna API Test")
    print("=" * 70)

    result = search_adzuna(
        {
            "title": "developer",
            "location": ""
        },
        5
    )

    print("\n" + "=" * 70)
    print("TEST RESULT")
    print("=" * 70)

    print(
        f"Success: {result.get('success')}"
    )

    print(
        f"Total: {result.get('total_count')}"
    )

    print(
        f"Jobs: {len(result.get('jobs', []))}"
    )

    print(
        f"Message: {result.get('message')}"
    )

    print("=" * 70)
```
