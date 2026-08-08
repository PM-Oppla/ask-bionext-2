import os
import re
import io
from urllib.parse import urljoin

import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import pypdf


# =============================================================================
# Page setup
# =============================================================================

st.set_page_config(
    page_title="Ask BIONEXT 2.0",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Styling
# =============================================================================

st.markdown(
    """
<style>
:root {
    --teal: #1A6B7C;
    --green: #8A9A3A;
    --light: #F0F5F5;
    --dark: #173F4A;
}

.header {
    background: var(--teal);
    color: white;
    padding: 1.4rem 2rem;
    border-radius: 10px;
    margin-bottom: 1.5rem;
}

.header h1 {
    margin: 0;
    font-size: 2rem;
    font-weight: 800;
}

.header p {
    color: #d3dc9a;
    margin: 0.5rem 0 0 0;
    font-weight: 600;
}

.answer {
    background: white;
    border: 1px solid #d9e1e3;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    line-height: 1.6;
}

.agent-box {
    background: #f7fafb;
    border-left: 5px solid var(--green);
    border-top: 1px solid #d9e1e3;
    border-right: 1px solid #d9e1e3;
    border-bottom: 1px solid #d9e1e3;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 1rem;
}

.source-box {
    background: #ffffff;
    border: 1px solid #d9e1e3;
    border-radius: 8px;
    padding: 0.7rem 0.9rem;
    margin-bottom: 0.6rem;
}

[data-testid="stSidebar"] {
    background-color: var(--light);
    border-right: 3px solid var(--green);
}

.stButton button {
    background-color: var(--teal) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
}

.stButton button:hover {
    background-color: var(--green) !important;
    color: var(--dark) !important;
}

footer {
    visibility: hidden;
}
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# Source configuration
# =============================================================================

APP_NAME = "Ask BIONEXT 2.0"

CORE_URLS = [
    "[oppla.eu](https://oppla.eu/bionext)",
    "[oppla.eu](https://oppla.eu/bionext/portal)",
    "[oppla.eu](https://oppla.eu/bionext/about-bionext)",
    "[oppla.eu](https://oppla.eu/bionext/decision-analysis-environmental-decision-making)",
    "[oppla.eu](https://oppla.eu/bionext/bionext-news-articles)",
    "[oppla.eu](https://oppla.eu/bionext/bionext-resources)",
    "[cordis.europa.eu](https://cordis.europa.eu/project/id/101059662)",
    "[ipbes.net](https://www.ipbes.net/)",
    "[ict.ipbes.net](https://ict.ipbes.net/ipbes-ict-guide/data-and-knowledge-management/citations-of-ipbes-assessments/transformative-change-assessment)",
]

DECISION_URL = "[oppla.eu](https://oppla.eu/bionext/decision-analysis-environmental-decision-making)"
RESOURCES_URL = "[oppla.eu](https://oppla.eu/bionext/bionext-resources)"
CORDIS_URL = "[cordis.europa.eu](https://cordis.europa.eu/project/id/101059662)"
IPBES_TCA_URL = "[ict.ipbes.net](https://ict.ipbes.net/ipbes-ict-guide/data-and-knowledge-management/citations-of-ipbes-assessments/transformative-change-assessment)"


def allowed(url: str) -> bool:
    """Strict allowlist for Ask BIONEXT 2.0 retrieval."""
    clean = url.split("#")[0].strip()

    if clean.startswith("[oppla.eu](https://oppla.eu/bionext)"):
        return True

    if clean.startswith("[oppla.eu](https://oppla.eu/sites/default/files/)"):
        return True

    if clean in CORE_URLS:
        return True

    if clean.startswith(
        "[ict.ipbes.net](https://ict.ipbes.net/ipbes-ict-guide/data-and-knowledge-management/citations-of-ipbes-assessments/)"
    ):
        return True

    return False


# =============================================================================
# OpenAI setup
# =============================================================================

def get_api_key() -> str:
    try:
        key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        key = ""

    return key or os.environ.get("OPENAI_API_KEY", "")


client = OpenAI(api_key=get_api_key())


# =============================================================================
# Built-in approved source pack
# =============================================================================

def fallback_sources():
    """
    Small built-in source pack for prototype stability.

    These are not general internet facts. They are concise source notes tied to
    the approved Ask BIONEXT 2.0 source list, used when live extraction is weak.
    """
    return [
        {
            "url": "[oppla.eu](https://oppla.eu/bionext)",
            "title": "BIONEXT on Oppla",
            "text": (
                "BIONEXT is the Biodiversity Nexus project. It explores the connections "
                "between biodiversity and other major societal and environmental systems, "
                "including climate, food, water, energy, transport and health. The project "
                "focuses on transformative change for sustainability and supports thinking "
                "about nature-centred futures."
            ),
            "source_type": "Built-in approved source note",
            "links": [],
        },
        {
            "url": "[oppla.eu](https://oppla.eu/bionext/about-bionext)",
            "title": "About BIONEXT",
            "text": (
                "BIONEXT is a European research project focused on the biodiversity nexus. "
                "It examines how biodiversity loss, biodiversity recovery and human systems "
                "interact. It develops knowledge, resources and decision-support approaches "
                "to help understand pathways for transformative change."
            ),
            "source_type": "Built-in approved source note",
            "links": [],
        },
        {
            "url": "[oppla.eu](https://oppla.eu/bionext/bionext-resources)",
            "title": "BIONEXT resources",
            "text": (
                "The BIONEXT resources section provides access to project outputs, resources, "
                "publications, policy briefs, reports and other materials associated with the "
                "BIONEXT project."
            ),
            "source_type": "Built-in approved source note",
            "links": [],
        },
        {
            "url": "[oppla.eu](https://oppla.eu/bionext/decision-analysis-environmental-decision-making)",
            "title": "Decision analysis in environmental decision-making",
            "text": (
                "The BIONEXT decision-analysis area concerns environmental decision-making. "
                "It relates to structured approaches for comparing options, considering "
                "criteria, understanding trade-offs, handling complexity and supporting "
                "transparent decisions."
            ),
            "source_type": "Built-in approved source note",
            "links": [],
        },
        {
            "url": CORDIS_URL,
            "title": "CORDIS BIONEXT project page",
            "text": (
                "The CORDIS BIONEXT project page is an approved external source for official "
                "EU project information about BIONEXT, including project identity, funding "
                "context and official project description."
            ),
            "source_type": "Built-in approved source note",
            "links": [],
        },
        {
            "url": IPBES_TCA_URL,
            "title": "IPBES transformative change assessment citation page",
            "text": (
                "The approved IPBES source provides reference context for IPBES material "
                "related to transformative change assessment and knowledge management."
            ),
            "source_type": "Built-in approved source note",
            "links": [],
        },
    ]


# =============================================================================
# Retrieval functions
# =============================================================================

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_page(url: str) -> dict:
    """Fetch approved HTML pages or PDFs and extract readable text."""
    if not allowed(url):
        return {
            "url": url,
            "title": "Blocked source",
            "text": "",
            "source_type": "Blocked",
            "links": [],
            "error": "This URL is outside the approved Ask BIONEXT 2.0 source boundary.",
        }

    try:
        headers = {"User-Agent": "AskBIONEXT2Prototype/1.0"}
        response = requests.get(url, headers=headers, timeout=25)
        response.raise_for_status()
    except Exception as e:
        return {
            "url": url,
            "title": url,
            "text": "",
            "source_type": "Live retrieval failed",
            "links": [],
            "error": str(e),
        }

    content_type = response.headers.get("content-type", "").lower()

    if "pdf" in content_type or url.lower().endswith(".pdf"):
        return extract_pdf(url, response.content)

    return extract_html(url, response.text)


def extract_pdf(url: str, content: bytes) -> dict:
    """Extract text from a PDF."""
    try:
        reader = pypdf.PdfReader(io.BytesIO(content))
        pages = []
        for page in reader.pages[:30]:
            pages.append(page.extract_text() or "")

        text = "\n\n".join(pages)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        return {
            "url": url,
            "title": url.split("/")[-1],
            "text": text,
            "source_type": "Live PDF",
            "links": [],
            "error": "",
        }

    except Exception as e:
        return {
            "url": url,
            "title": url,
            "text": "",
            "source_type": "PDF extraction failed",
            "links": [],
            "error": str(e),
        }


def extract_html(url: str, html: str) -> dict:
    """Extract text and approved links from an HTML page."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else url

    text = soup.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    links = []
    for a in soup.find_all("a", href=True):
        absolute = urljoin(url, a["href"]).split("#")[0]
        if allowed(absolute):
            links.append(absolute)

    links = list(dict.fromkeys(links))

    return {
        "url": url,
        "title": title,
        "text": text,
        "source_type": "Live webpage",
        "links": links[:40],
        "error": "",
    }


# =============================================================================
# Agent routing
# =============================================================================

def route_question(question: str):
    """Simple agentic route selection."""
    q = question.lower()

    decision_terms = [
        "decision",
        "decision-making",
        "decision making",
        "mcda",
        "multi-criteria",
        "multicriteria",
        "criteria",
        "trade-off",
        "tradeoff",
        "option appraisal",
        "environmental decision",
    ]

    resource_terms = [
        "resource",
        "resources",
        "report",
        "policy brief",
        "deliverable",
        "publication",
        "output",
        "pdf",
        "library",
    ]

    ipbes_terms = [
        "ipbes",
        "transformative change",
        "assessment",
    ]

    cordis_terms = [
        "cordis",
        "grant",
        "funding",
        "eu project",
        "project id",
        "101059662",
    ]

    if any(term in q for term in decision_terms):
        return "Decision-analysis specialist route", [
            DECISION_URL,
            "[oppla.eu](https://oppla.eu/bionext/about-bionext)",
            "[oppla.eu](https://oppla.eu/bionext)",
        ]

    if any(term in q for term in resource_terms):
        return "BIONEXT resources route", [
            RESOURCES_URL,
            "[oppla.eu](https://oppla.eu/bionext)",
            "[oppla.eu](https://oppla.eu/bionext/about-bionext)",
        ]

    if any(term in q for term in ipbes_terms):
        return "IPBES and transformative-change route", [
            "[oppla.eu](https://oppla.eu/bionext)",
            IPBES_TCA_URL,
            "[ipbes.net](https://www.ipbes.net/)",
        ]

    if any(term in q for term in cordis_terms):
        return "CORDIS project-information route", [
            CORDIS_URL,
            "[oppla.eu](https://oppla.eu/bionext/about-bionext)",
            "[oppla.eu](https://oppla.eu/bionext)",
        ]

    return "General BIONEXT route", [
        "[oppla.eu](https://oppla.eu/bionext)",
        "[oppla.eu](https://oppla.eu/bionext/portal)",
        "[oppla.eu](https://oppla.eu/bionext/about-bionext)",
        CORDIS_URL,
    ]


def score_source(question: str, item: dict) -> int:
    """Very simple relevance scoring."""
    words = re.findall(r"[a-zA-Z][a-zA-Z-]{3,}", question.lower())

    stop_words = {
        "what",
        "when",
        "where",
        "which",
        "about",
        "does",
        "from",
        "with",
        "that",
        "this",
        "there",
        "their",
        "have",
        "been",
        "being",
        "trying",
    }

    words = [w for w in words if w not in stop_words]

    haystack = (
        item.get("title", "")
        + "\n"
        + item.get("url", "")
        + "\n"
        + item.get("text", "")
    ).lower()

    return sum(haystack.count(w) for w in words)


def gather_evidence(question: str):
    """
    Controlled source retrieval.

    1. Route question.
    2. Fetch seed sources.
    3. Follow a few approved links.
    4. Add built-in approved source notes as fallback/context.
    5. Sort by relevance.
    """
    route_label, seed_urls = route_question(question)

    items = []

    for url in seed_urls:
        item = fetch_page(url)
        if item.get("text"):
            items.append(item)

    follow_links = []
    for item in items:
        for link in item.get("links", []):
            if link not in seed_urls and allowed(link):
                follow_links.append(link)

    follow_links = list(dict.fromkeys(follow_links))[:5]

    for link in follow_links:
        item = fetch_page(link)
        if item.get("text"):
            items.append(item)

    # Add stable approved source notes, so the prototype never fails silently.
    items.extend(fallback_sources())

    deduped = []
    seen = set()
    for item in items:
        key = item.get("url")
        if key not in seen:
            deduped.append(item)
            seen.add(key)

    ranked = sorted(deduped, key=lambda item: score_source(question, item), reverse=True)

    return ranked[:6], route_label


# =============================================================================
# Answer generation
# =============================================================================

def build_context(evidence):
    sections = []

    for i, item in enumerate(evidence, start=1):
        text = item.get("text", "").strip()
        if not text:
            continue

        sections.append(
            f"""SOURCE {i}
Title: {item.get("title", "Untitled")}
URL: {item.get("url", "")}
Source type: {item.get("source_type", "Source")}
Text:
{text[:5000]}
"""
        )

    return "\n\n---\n\n".join(sections)


def generate_answer(question: str):
    evidence, route_label = gather_evidence(question)

    context = build_context(evidence)

    if not context.strip():
        return (
            "I could not find enough information in the approved Ask BIONEXT 2.0 sources to answer that question.",
            route_label,
            evidence,
        )

    prompt = f"""
You are Ask BIONEXT 2.0, a source-bounded research assistant for the BION
