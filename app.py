import os
import re
import io
from urllib.parse import urljoin, urlparse

import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import pypdf


# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Ask BIONEXT 2.0",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------

st.markdown(
    """
<style>
:root {
    --bionext-teal: #1A6B7C;
    --bionext-green: #8A9A3A;
    --bionext-light: #F0F5F5;
    --bionext-dark: #173F4A;
}

.main-header {
    background: var(--bionext-teal);
    color: white;
    padding: 1.5rem 2rem;
    border-radius: 10px;
    margin-bottom: 1.5rem;
}

.main-header h1 {
    margin: 0;
    font-size: 2.1rem;
    font-weight: 800;
}

.main-header p {
    margin-top: 0.6rem;
    color: #d3dc9a;
    font-size: 1rem;
    font-weight: 500;
}

.source-card {
    background: #ffffff;
    border: 1px solid #d9e1e3;
    border-left: 5px solid var(--bionext-green);
    padding: 0.8rem 1rem;
    border-radius: 8px;
    margin-bottom: 0.7rem;
    font-size: 0.9rem;
}

.agent-card {
    background: #f7fafb;
    border: 1px solid #d9e1e3;
    padding: 0.9rem 1rem;
    border-radius: 8px;
    margin-bottom: 1rem;
}

.answer-box {
    background: white;
    border: 1px solid #d9e1e3;
    padding: 1.1rem 1.2rem;
    border-radius: 10px;
    line-height: 1.6;
}

[data-testid="stSidebar"] {
    background-color: var(--bionext-light);
    border-right: 3px solid var(--bionext-green);
}

.stButton button {
    background-color: var(--bionext-teal) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
}

.stButton button:hover {
    background-color: var(--bionext-green) !important;
    color: var(--bionext-dark) !important;
}

footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Approved source configuration
# -----------------------------------------------------------------------------

APP_NAME = "Ask BIONEXT 2.0"

CORE_BIONEXT_URLS = [
    "[oppla.eu](https://oppla.eu/bionext)",
    "[oppla.eu](https://oppla.eu/bionext/portal)",
    "[oppla.eu](https://oppla.eu/bionext/about-bionext)",
    "[oppla.eu](https://oppla.eu/bionext/decision-analysis-environmental-decision-making)",
    "[oppla.eu](https://oppla.eu/bionext/bionext-news-articles)",
    "[oppla.eu](https://oppla.eu/bionext/bionext-resources)",
]

APPROVED_EXTERNAL_URLS = [
    "[cordis.europa.eu](https://cordis.europa.eu/project/id/101059662)",
    "[ipbes.net](https://www.ipbes.net/)",
    "[ict.ipbes.net](https://ict.ipbes.net/ipbes-ict-guide/data-and-knowledge-management/citations-of-ipbes-assessments/transformative-change-assessment)",
]

DECISION_ANALYSIS_ROOT = (
    "[oppla.eu](https://oppla.eu/bionext/decision-analysis-environmental-decision-making)"
)

RESOURCE_ROOT = "[oppla.eu](https://oppla.eu/bionext/bionext-resources)"


def is_allowed_url(url: str) -> bool:
    """Restrict retrieval to approved BIONEXT and external sources."""
    parsed = urlparse(url)
    clean_url = url.split("#")[0]

    if clean_url.startswith("[oppla.eu](https://oppla.eu/bionext)"):
        return True

    if clean_url.startswith("[oppla.eu](https://oppla.eu/sites/default/files/)"):
        return True

    if clean_url in APPROVED_EXTERNAL_URLS:
        return True

    if clean_url.startswith(
        "[ict.ipbes.net](https://ict.ipbes.net/ipbes-ict-guide/data-and-knowledge-management/citations-of-ipbes-assessments/)"
    ):
        return True

    return False


# -----------------------------------------------------------------------------
# API setup
# -----------------------------------------------------------------------------

def get_api_key():
    try:
        key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        key = ""
    return key or os.environ.get("OPENAI_API_KEY", "")


client = OpenAI(api_key=get_api_key())


# -----------------------------------------------------------------------------
# Retrieval helpers
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_html(url: str) -> dict:
    """Fetch and extract text + links from an allowed HTML page."""
    if not is_allowed_url(url):
        return {
            "url": url,
            "title": "Blocked source",
            "text": "",
            "links": [],
            "error": "URL is outside the approved Ask BIONEXT 2.0 source boundary.",
        }

    headers = {
        "User-Agent": "AskBIONEXT2Prototype/1.0 (+[oppla.eu](https://oppla.eu/bionext)"
    }

    try:
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
    except Exception as e:
        return {
            "url": url,
            "title": url,
            "text": "",
            "links": [],
            "error": f"Could not fetch page: {e}",
        }

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else url
    text = soup.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)

    links = []
    for a in soup.find_all("a", href=True):
        absolute = urljoin(url, a["href"])
        absolute = absolute.split("#")[0]
        if is_allowed_url(absolute):
            label = a.get_text(" ", strip=True) or absolute
            links.append({"url": absolute, "label": label})

    deduped = []
    seen = set()
    for link in links:
        if link["url"] not in seen:
            deduped.append(link)
            seen.add(link["url"])

    return {
        "url": url,
        "title": title,
        "text": text,
        "links": deduped[:80],
        "error": "",
    }


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_pdf(url: str) -> dict:
    """Fetch and extract text from an allowed PDF."""
    if not is_allowed_url(url):
        return {
            "url": url,
            "title": "Blocked PDF",
            "text": "",
            "links": [],
            "error": "PDF is outside the approved Ask BIONEXT 2.0 source boundary.",
        }

    headers = {
        "User-Agent": "AskBIONEXT2Prototype/1.0 (+[oppla.eu](https://oppla.eu/bionext)"
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        reader = pypdf.PdfReader(io.BytesIO(response.content))
        pages = []
        for page in reader.pages[:30]:
            pages.append(page.extract_text() or "")

        text = "\n\n".join(pages)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return {
            "url": url,
            "title": url.split("/")[-1],
            "text": text,
            "links": [],
            "error": "",
        }
    except Exception as e:
        return {
            "url": url,
            "title": url,
            "text": "",
            "links": [],
            "error": f"Could not read PDF: {e}",
        }


def fetch_source(url: str) -> dict:
    if url.lower().endswith(".pdf"):
        return fetch_pdf(url)
    return fetch_html(url)


# -----------------------------------------------------------------------------
# Agent routing
# -----------------------------------------------------------------------------

def route_question(question: str) -> dict:
    """Simple rule-based agent router for the first prototype."""
    q = question.lower()

    route = {
        "label": "General BIONEXT route",
        "description": "Searching core BIONEXT project pages and selected approved reference pages.",
        "seed_urls": list(CORE_BIONEXT_URLS),
    }

    decision_terms = [
        "decision analysis",
        "multi-criteria",
        "multicriteria",
        "mcda",
        "decision-making",
        "decision making",
        "environmental decision",
        "criteria",
        "trade-off",
        "tradeoff",
        "stakeholder preference",
    ]

    resource_terms = [
        "resource",
        "resources",
        "policy brief",
        "report",
        "deliverable",
        "publication",
        "pdf",
        "output",
        "outputs",
        "library",
    ]

    news_terms = [
        "news",
        "article",
        "blog",
        "event",
        "webinar",
        "workshop",
    ]

    ipbes_terms = [
        "ipbes",
        "transformative change",
        "assessment",
        "biodiversity assessment",
    ]

    if any(term in q for term in decision_terms):
        route = {
            "label": "Decision-analysis specialist route",
            "description": "Prioritising BIONEXT decision-analysis pages and their approved linked material.",
            "seed_urls": [
                DECISION_ANALYSIS_ROOT,
                "[oppla.eu](https://oppla.eu/bionext)",
                "[oppla.eu](https://oppla.eu/bionext/about-bionext)",
            ],
        }

    elif any(term in q for term in resource_terms):
        route = {
            "label": "BIONEXT resources route",
            "description": "Prioritising the BIONEXT resources section and linked readable resources.",
            "seed_urls": [
                RESOURCE_ROOT,
                "[oppla.eu](https://oppla.eu/bionext)",
                "[oppla.eu](https://oppla.eu/bionext/about-bionext)",
            ],
        }

    elif any(term in q for term in news_terms):
        route = {
            "label": "BIONEXT news route",
            "description": "Prioritising BIONEXT news and article pages.",
            "seed_urls": [
                "[oppla.eu](https://oppla.eu/bionext/bionext-news-articles)",
                "[oppla.eu](https://oppla.eu/bionext)",
            ],
        }

    elif any(term in q for term in ipbes_terms):
        route = {
            "label": "Transformative-change and IPBES route",
            "description": "Using approved BIONEXT and IPBES sources only.",
            "seed_urls": [
                "[oppla.eu](https://oppla.eu/bionext)",
                "[ipbes.net](https://www.ipbes.net/)",
                "[ict.ipbes.net](https://ict.ipbes.net/ipbes-ict-guide/data-and-knowledge-management/citations-of-ipbes-assessments/transformative-change-assessment)",
            ],
        }

    if "cordis" in q or "eu project" in q or "grant" in q or "funding" in q:
        route["seed_urls"].append("[cordis.europa.eu](https://cordis.europa.eu/project/id/101059662)")

    return route


def score_text(question: str, text: str, title: str = "", url: str = "") -> int:
    stop_words = {
        "what", "how", "why", "when", "where", "who", "which", "is", "are",
        "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or",
        "do", "does", "did", "can", "could", "should", "would", "tell", "me",
        "about", "please", "with", "from", "that", "this", "it", "as", "by",
    }

    words = set(re.sub(r"[^\w\s-]", " ", question.lower()).split())
    words = {w for w in words if len(w) > 2 and w not in stop_words}

    haystack = f"{title}\n{url}\n{text}".lower()
    return sum(haystack.count(w) for w in words)


def collect_evidence(question: str, route: dict) -> tuple[list[dict], list[str]]:
    """Controlled multi-step retrieval over allowed sources."""
    notes = []
    fetched = []

    seed_urls = []
    for url in route["seed_urls"]:
        if url not in seed_urls and is_allowed_url(url):
            seed_urls.append(url)

    notes.append(f"Route selected: {route['label']}")
    notes.append(f"Seed sources checked: {len(seed_urls)}")

    for url in seed_urls:
        item = fetch_source(url)
        fetched.append(item)

    candidate_links = []
    for item in fetched:
        for link in item.get("links", []):
            url = link["url"]
            if url not in seed_urls and is_allowed_url(url):
                candidate_links.append(
                    {
                        "url": url,
                        "label": link.get("label", url),
                        "score": score_text(question, link.get("label", ""), url=url),
                    }
                )

    candidate_links = sorted(candidate_links, key=lambda x: x["score"], reverse=True)

    follow_limit = 5
    followed = 0
    seen_urls = {item["url"] for item in fetched}

    for link in candidate_links:
        if followed >= follow_limit:
            break
        if link["url"] in seen_urls:
            continue
        item = fetch_source(link["url"])
        fetched.append(item)
        seen_urls.add(link["url"])
        followed += 1

    notes.append(f"Follow-up sources checked: {followed}")

    valid = [item for item in fetched if item.get("text") and not item.get("error")]
    valid = sorted(
        valid,
        key=lambda x: score_text(question, x.get("text", ""), x.get("title", ""), x.get("url", "")),
        reverse=True,
    )

    return valid[:6], notes


# -----------------------------------------------------------------------------
# Answer generation
# -----------------------------------------------------------------------------

def build_context(evidence: list[dict]) -> str:
    parts = []
    for i, item in enumerate(evidence, start=1):
        text = item.get("text", "")
