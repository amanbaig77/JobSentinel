import streamlit as st
import sys
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from predict import predict_job
from job_recommender import find_similar_jobs
from job_similarity import get_ranked_recommendations


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="JobSentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #f5f7fa;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    .app-header {
        background: #172033;
        padding: 25px 30px;
        border-radius: 14px;
        margin-bottom: 25px;
    }

    .app-header h1 {
        color: white;
        margin: 0;
        font-size: 34px;
    }

    .app-header p {
        color: #cbd5e1;
        margin-top: 7px;
        margin-bottom: 0;
        font-size: 14px;
    }

    .section-box {
        background: white;
        padding: 22px;
        border: 1px solid #e1e6ed;
        border-radius: 14px;
        margin-bottom: 18px;
    }

    .risk-high {
        background: #fff1f2;
        border-left: 6px solid #dc2626;
        padding: 18px;
        border-radius: 10px;
        margin: 15px 0;
    }

    .risk-medium {
        background: #fffbeb;
        border-left: 6px solid #d97706;
        padding: 18px;
        border-radius: 10px;
        margin: 15px 0;
    }

    .risk-low {
        background: #f0fdf4;
        border-left: 6px solid #16a34a;
        padding: 18px;
        border-radius: 10px;
        margin: 15px 0;
    }

    .risk-high h3,
    .risk-medium h3,
    .risk-low h3 {
        margin-top: 0;
    }

    .signal {
        background: #fff1f2;
        border: 1px solid #fecdd3;
        padding: 8px 12px;
        border-radius: 8px;
        margin-bottom: 7px;
        color: #991b1b;
        font-size: 14px;
    }

    .job-card {
        background: white;
        border: 1px solid #e1e6ed;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 12px;
    }

    .metric-box {
        background: white;
        border: 1px solid #e1e6ed;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }

    .metric-label {
        color: #6b7280;
        font-size: 12px;
    }

    .metric-value {
        color: #172033;
        font-size: 25px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="app-header">
        <h1>🛡️ JobSentinel</h1>
        <p>
            Machine-learning job fraud detection and
            similar job discovery.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🛡️ JobSentinel")

    st.caption(
        "Job fraud detection system"
    )

    st.divider()

    st.subheader("Model")

    st.write("**Algorithm:** Linear SVM")
    st.write("**Features:** 51,423")
    st.write("**F1 Score:** 0.8563")
    st.write("**ROC-AUC:** 0.9894")
    st.write("**PR-AUC:** 0.9171")

    st.divider()

    st.caption(
        "The ML result is an assessment and does not "
        "independently prove fraud."
    )


# ============================================================
# INTRO
# ============================================================

st.subheader("Analyze a Job Posting")

st.caption(
    "Enter the information available from the job posting. "
    "You do not need to complete every field."
)


# ============================================================
# INPUTS
# ============================================================

left, right = st.columns(
    [1.35, 1],
    gap="large"
)


# ============================================================
# LEFT
# ============================================================

with left:

    st.markdown("### Posting Content")

    title = st.text_input(
        "Job Title",
        placeholder="Example: Python Developer"
    )

    company_profile = st.text_area(
        "Company Profile",
        height=100,
        placeholder="Information about the company..."
    )

    description = st.text_area(
        "Job Description",
        height=180,
        placeholder="Paste the complete job description..."
    )

    requirements = st.text_area(
        "Requirements",
        height=110,
        placeholder="Skills, qualifications and requirements..."
    )

    benefits = st.text_area(
        "Benefits",
        height=90,
        placeholder="Salary, benefits, perks..."
    )


# ============================================================
# RIGHT
# ============================================================

with right:

    st.markdown("### Job Details")

    employment_type = st.selectbox(
        "Employment Type",
        [
            "",
            "Full-time",
            "Part-time",
            "Contract",
            "Temporary",
            "Other"
        ]
    )

    required_experience = st.selectbox(
        "Required Experience",
        [
            "",
            "Internship",
            "Entry level",
            "Associate",
            "Mid-Senior level",
            "Director",
            "Executive",
            "Not Applicable"
        ]
    )

    required_education = st.selectbox(
        "Required Education",
        [
            "",
            "High School",
            "Associate Degree",
            "Bachelor's Degree",
            "Master's Degree",
            "Doctorate",
            "Professional",
            "Unspecified"
        ]
    )

    industry = st.text_input(
        "Industry",
        placeholder="Example: Information Technology"
    )

    function = st.text_input(
        "Function",
        placeholder="Example: Engineering"
    )

    department = st.text_input(
        "Department",
        placeholder="Example: Software Development"
    )

    salary_range = st.text_input(
        "Salary Range",
        placeholder="Example: ₹6,00,000 - ₹12,00,000"
    )

    location = st.text_input(
        "Location",
        placeholder="Example: Bangalore"
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.divider()

analyze = st.button(
    "🔎 Analyze Job Posting",
    type="primary",
    use_container_width=True
)


# ============================================================
# ANALYSIS
# ============================================================

if analyze:

    if not title.strip() and not description.strip():

        st.warning(
            "Please provide at least a Job Title or Job Description."
        )

        st.stop()


    # --------------------------------------------------------
    # NORMALIZE VALUES
    # --------------------------------------------------------

    employment_value = (
        ""
        if employment_type == ""
        else employment_type
    )

    experience_value = (
        ""
        if required_experience == ""
        else required_experience
    )

    education_value = (
        ""
        if required_education == ""
        else required_education
    )


    # --------------------------------------------------------
    # JOB DATA
    # --------------------------------------------------------

    job_data = {

        "title":
            title.strip(),

        "company_profile":
            company_profile.strip(),

        "description":
            description.strip(),

        "requirements":
            requirements.strip(),

        "benefits":
            benefits.strip(),

        "employment_type":
            employment_value,

        "required_experience":
            experience_value,

        "required_education":
            education_value,

        "industry":
            industry.strip(),

        "function":
            function.strip(),

        "department":
            department.strip(),

        "salary_range":
            salary_range.strip(),

        "location":
            location.strip()
    }


    # ========================================================
    # ML PREDICTION
    # ========================================================

    st.divider()

    st.header("Analysis Result")

    st.caption(
        "Machine-learning assessment of the submitted job posting."
    )


    with st.spinner("Analyzing job posting..."):

        try:

            result = predict_job(
                job_data
            )

        except Exception as error:

            st.error(
                "Prediction failed."
            )

            st.exception(error)

            st.stop()


    # --------------------------------------------------------
    # RESULT VALUES
    # --------------------------------------------------------

    prediction = result.get(
        "prediction",
        "UNKNOWN"
    )

    decision_score = result.get(
        "decision_score"
    )

    threshold = result.get(
        "threshold"
    )

    risk_level = result.get(
        "risk_level",
        "MEDIUM"
    )

    model_name = result.get(
        "model",
        "Linear SVM"
    )

    explanation = result.get(
        "interpretation",
        ""
    )

    risk_signals = result.get(
        "risk_signals",
        []
    )


    # ========================================================
    # RISK DISPLAY
    # ========================================================

    if risk_level == "HIGH":

        st.markdown(
            f"""
            <div class="risk-high">
                <h3>🔴 HIGH RISK</h3>
                <div>{explanation}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    elif risk_level == "MEDIUM":

        st.markdown(
            f"""
            <div class="risk-medium">
                <h3>🟡 MEDIUM RISK</h3>
                <div>{explanation}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="risk-low">
                <h3>🟢 LOW RISK</h3>
                <div>{explanation}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # MODEL OUTPUT
    # ========================================================

    st.subheader("Model Output")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Decision Score",
            (
                f"{decision_score:.4f}"
                if decision_score is not None
                else "N/A"
            )
        )

    with col2:

        st.metric(
            "Threshold",
            (
                f"{threshold:.2f}"
                if threshold is not None
                else "N/A"
            )
        )

    with col3:

        st.metric(
            "Risk Level",
            risk_level
        )

    with col4:

        st.metric(
            "Model",
            model_name
        )


    # ========================================================
    # PREDICTION
    # ========================================================

    with st.expander("View Detection Details"):

        st.write(
            f"**Model Classification:** {prediction}"
        )

        if decision_score is not None:

            st.write(
                f"**Decision Score:** "
                f"{decision_score:.4f}"
            )

        if threshold is not None:

            st.write(
                f"**Classification Threshold:** "
                f"{threshold:.4f}"
            )

        if decision_score is not None and threshold is not None:

            if decision_score >= threshold:

                st.write(
                    "The SVM decision score is at or above "
                    "the configured fraud threshold."
                )

            else:

                st.write(
                    "The SVM decision score is below "
                    "the configured fraud threshold."
                )


    # ========================================================
    # RISK SIGNALS
    # ========================================================

    if risk_signals:

        st.subheader("Detected Risk Signals")

        for signal in risk_signals:

            readable = (
                signal
                .replace("_", " ")
                .title()
            )

            st.markdown(
                f"""
                <div class="signal">
                    ⚠️ {readable}
                </div>
                """,
                unsafe_allow_html=True
            )


    # ========================================================
    # SIMILAR JOBS
    # ========================================================

    st.divider()

    st.header("Similar Job Opportunities")

    st.caption(
        "Real job listings are searched using the submitted "
        "job title and location, then ranked using text similarity."
    )


    with st.spinner(
        "Searching for similar jobs..."
    ):

        try:

            jobs_result = find_similar_jobs(
                job_data,
                results_per_page=12
            )

        except Exception as error:

            jobs_result = {
                "success": False,
                "jobs": [],
                "total_count": 0,
                "message": str(error)
            }


    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    jobs = []

    total_jobs = 0

    search_message = ""


    if isinstance(
        jobs_result,
        dict
    ):

        jobs = (
            jobs_result.get("jobs")
            or []
        )

        total_jobs = (
            jobs_result.get("total_count")
            or len(jobs)
        )

        search_message = (
            jobs_result.get("message")
            or ""
        )


    elif isinstance(
        jobs_result,
        list
    ):

        jobs = jobs_result

        total_jobs = len(jobs)


    # ========================================================
    # RANK
    # ========================================================

    ranked_jobs = jobs


    if jobs:

        try:

            ranked_jobs = get_ranked_recommendations(
                job_data,
                jobs,
                5
            )

        except TypeError:

            try:

                ranked_jobs = get_ranked_recommendations(
                    job_data,
                    jobs
                )

            except Exception:

                ranked_jobs = jobs

        except Exception:

            ranked_jobs = jobs


    # ========================================================
    # DISPLAY JOBS
    # ========================================================

    if not ranked_jobs:

        st.info(
            search_message
            if search_message
            else
            "No similar jobs were found. "
            "Try another job title or location."
        )

    else:

        st.success(
            f"Found {total_jobs} possible opportunities."
        )

        for index, job in enumerate(
            ranked_jobs[:5],
            start=1
        ):

            if not isinstance(
                job,
                dict
            ):
                continue


            job_title = (
                job.get("title")
                or "Job Opportunity"
            )

            company = (
                job.get("company")
                or "Company not specified"
            )

            job_location = (
                job.get("location")
                or "Location not specified"
            )

            url = (
                job.get("redirect_url")
                or job.get("url")
                or job.get("link")
            )

            similarity = (
                job.get("similarity")
                if job.get("similarity") is not None
                else job.get("similarity_score")
            )

            salary_min = job.get(
                "salary_min"
            )

            salary_max = job.get(
                "salary_max"
            )


            with st.container(
                border=True
            ):

                left_col, right_col = st.columns(
                    [4, 1]
                )

                with left_col:

                    st.subheader(
                        f"{index}. {job_title}"
                    )

                    st.write(
                        f"**Company:** {company}"
                    )

                    st.write(
                        f"**Location:** {job_location}"
                    )


                    if (
                        salary_min is not None
                        or salary_max is not None
                    ):

                        if (
                            salary_min is not None
                            and salary_max is not None
                        ):

                            salary_text = (
                                f"{salary_min} - {salary_max}"
                            )

                        elif salary_min is not None:

                            salary_text = str(
                                salary_min
                            )

                        else:

                            salary_text = str(
                                salary_max
                            )

                        st.write(
                            f"**Salary:** {salary_text}"
                        )


                with right_col:

                    if similarity is not None:

                        try:

                            similarity_value = float(
                                similarity
                            )

                            if similarity_value <= 1:

                                similarity_value *= 100

                            st.metric(
                                "Match",
                                f"{similarity_value:.1f}%"
                            )

                        except Exception:

                            pass


                if url:

                    st.link_button(
                        "View Job",
                        url,
                        use_container_width=True
                    )


    # ========================================================
    # SAFETY NOTICE
    # ========================================================

    st.divider()

    st.info(
        "JobSentinel provides a machine-learning risk assessment. "
        "A fraudulent classification is not definitive proof of fraud. "
        "Always independently verify the employer and job opportunity."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "JobSentinel • Machine Learning Job Fraud Detection"
)