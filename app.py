import os
import re
import io
import requests
import streamlit as st
from bs4 import BeautifulSoup
from openai import OpenAI
import pypdf

st.set_page_config(
    page_title="Ask BIONEXT 2.0",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root {
  --teal: #1A6B7C;
  --green: #8A9A3A;
  --light: #F0F5F5;
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
footer {
  visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

BIONEXT_RESOURCES_URL = "[oppla.eu](https://oppla.eu/bionext/bionext-resources)"
KNOWN_BIONEXT_PDF_URL = "[oppla.eu](https://oppla.eu/sites/default/files/2026-06/Bionext_PB%2B_4_2026.pdf)"

APPROVED_SOURCES = [
    {
        "title": "BIONEXT on Oppla",
        "url": "[oppla.eu](https://oppla.eu/bionext)",
        "text": "BIONEXT is the Biodiversity Nexus project. It explores connections between biodiversity and climate, food, water, energy, transport and health. It focuses on transformative change for sustainability and nature-centred futures.",
    },
    {
        "title": "About BIONEXT",
        "url": "[oppla.eu](https://oppla.eu/bionext/about-bionext)",
        "text": "BIONEXT is a European research project focused on the biodiversity nexus. It examines how biodiversity loss, biodiversity recovery and human systems interact. It develops knowledge, resources and decision-support approaches for transformative change.",
    },
    {
        "title": "BIONEXT resources",
        "url": BIONEXT_RESOURCES_URL,
        "text": "The BIONEXT resources section provides access to project outputs, publications, policy briefs, reports, deliverables and other materials associated with the BIONEXT project.",
    },
    {
        "title": "Decision analysis in environmental decision-making",
        "url": "[oppla.eu](https://oppla.eu/bionext/decision-analysis-environmental-decision-making)",
        "text": "The BIONEXT decision-analysis area concerns environmental decision-making. It relates to structured approaches for comparing options, considering criteria, understanding trade-offs, handling complexity and supporting transparent decisions.",
    },
    {
        "title": "BIONEXT news articles",
        "url": "[oppla.eu](https://oppla.eu/bionext/bionext-news-articles)",
        "text": "The BIONEXT news section contains updates, articles, workshops, webinars and other project communications related to BIONEXT activities.",
    },
    {
        "title": "CORDIS BIONEXT project page",
        "url": "[cordis.europa.eu](https://cordis.europa.eu/project/id/101059662)",
        "text": "The CORDIS BIONEXT page is an approved external source for official EU project information about BIONEXT, including project identity, funding context and official project description.",
    },
    {
        "title": "IPBES transformative change assessment citation page",
        "url": "[ict.ipbes.net](https://ict.ipbes.net/ipbes-ict-guide/data-and-knowledge-management/citations-of-ipbes-assessments/transformative-change-assessment)",
        "text": "The approved IPBES page provides reference context for IPBES material related to transformative change assessment and knowledge management.",
    },
]

LIVE_URLS = [
    "[oppla.eu](https://oppla.eu/bionext)",
    "[oppla.eu](https://oppla.eu/bionext/about-bionext)",
    BIONEXT_RESOURCES_URL,
    "[oppla.eu](https://oppla.eu/bionext/decision-analysis-environmental-decision-making)",
    "[cordis.europa.eu](https://cordis.europa.eu/project/id/101059662)",
]

def get_api_key():
    try:
        return st.secrets.get("OPENAI_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
    except Exception:
        return os.environ.get("OPENAI_API_KEY", "")

client = OpenAI(api_key=get_api_key())

def is_approved_url(url):
    return (
        url.startswith("[oppla.eu](https://oppla.eu/bionext)")
        or url.startswith("[oppla.eu](https://oppla.eu/sites/default/files/)")
        or url.startswith("[cordis.europa.eu](https://cordis.europa.eu/project/id/101059662)")
        or url.startswith("[ipbes.net](https://www.ipbes.net/)")
        or url.startswith("[ict.ipbes.net](https://ict.ipbes.net/ipbes-ict-guide/data-and-knowledge-management/citations-of-ipbes-assessments/)")
    )

def make_headers():
    return {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf,*/*;q=0.8",
        "Accept-Language": "en-GB,en;q=0.9",
    }

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_live_text(url):
    if not is_approved_url(url):
        return {
            "ok": False,
            "url": url,
            "error": "URL is outside the approved source boundary.",
        }

    result = {
        "ok": False,
        "requested_url": url,
        "final_url": "",
        "status_code": "",
        "content_type": "",
        "response_bytes": 0,
        "title": "",
        "text": "",
        "chars": 0,
        "links": [],
        "approved_links": [],
        "resource_links": [],
        "pdf_links": [],
        "raw_preview": "",
        "text_preview": "",
        "error": "",
    }

    try:
        r = requests.get(url, headers=make_headers(), timeout=25, allow_redirects=True)
        result["final_url"] = r.url
        result["status_code"] = r.status_code
        result["content_type"] = r.headers.get("content-type", "")
        result["response_bytes"] = len(r.content)
        result["raw_preview"] = r.text[:1000] if r.text else ""

        r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        title = soup.title.get_text(" ", strip=True) if soup.title else url
        text = soup.get_text("\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)

        all_links = []
        for a in soup.find_all("a", href=True):
            link = requests.compat.urljoin(url, a["href"]).split("#")[0]
            all_links.append(link)

        approved_links = [x for x in all_links if is_approved_url(x)]
        approved_links = list(dict.fromkeys(approved_links))

        resource_links = [x for x in approved_links if "/bionext/resource/" in x]
        pdf_links = [x for x in approved_links if ".pdf" in x.lower()]

        result["ok"] = True
        result["title"] = title
        result["text"] = text
        result["chars"] = len(text)
        result["links"] = list(dict.fromkeys(all_links))
        result["approved_links"] = approved_links
        result["resource_links"] = resource_links
        result["pdf_links"] = pdf_links
        result["text_preview"] = text[:1000]

    except Exception as e:
        result["error"] = str(e)

    return result

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_pdf_text(url):
    if not is_approved_url(url):
        return {
            "ok": False,
            "url": url,
            "error": "URL is outside the approved source boundary.",
        }

    result = {
        "ok": False,
        "requested_url": url,
        "final_url": "",
        "status_code": "",
        "content_type": "",
        "downloaded_bytes": 0,
        "title": url.split("/")[-1],
        "text": "",
        "chars": 0,
        "pages_checked": 0,
        "raw_start": "",
        "text_preview": "",
        "error": "",
    }

    try:
        r = requests.get(url, headers=make_headers(), timeout=40, allow_redirects=True)
        result["final_url"] = r.url
        result["status_code"] = r.status_code
        result["content_type"] = r.headers.get("content-type", "")
        result["downloaded_bytes"] = len(r.content)
        result["raw_start"] = r.content[:20].hex()

        r.raise_for_status()

        reader = pypdf.PdfReader(io.BytesIO(r.content))
        pages = []
        for page in reader.pages[:30]:
            pages.append(page.extract_text() or "")

        text = "\n\n".join(pages)
        text = re.sub(r"\n{3,}", "\n\n", text)

        result["ok"] = True
        result["pages_checked"] = min(len(reader.pages), 30)
        result["text"] = text
        result["chars"] = len(text)
        result["text_preview"] = text[:1000]

    except Exception as e:
        result["error"] = str(e)

    return result

def route_question(question):
    q = question.lower()

    if any(x in q for x in ["decision", "mcda", "multi-criteria", "criteria", "trade-off", "tradeoff"]):
        return "Decision-analysis route"

    if any(x in q for x in ["resource", "policy brief", "report", "deliverable", "publication", "output"]):
        return "BIONEXT resources route"

    if any(x in q for x in ["ipbes", "transformative change", "assessment"]):
        return "IPBES and transformative change route"

    if any(x in q for x in ["cordis", "grant", "funding", "eu project", "101059662"]):
        return "CORDIS project-information route"

    return "General BIONEXT route"

def score_source(question, source):
    words = re.findall(r"[a-zA-Z][a-zA-Z-]{3,}", question.lower())
    stop_words = {"what", "when", "where", "which", "about", "does", "from", "with", "that", "this", "there", "their"}
    words = [w for w in words if w not in stop_words]
    haystack = (source["title"] + " " + source["url"] + " " + source["text"]).lower()
    return sum(haystack.count(w) for w in words)

def gather_sources(question):
    route = route_question(question)
    sources = list(APPROVED_SOURCES)

    for url in LIVE_URLS:
        live = fetch_live_text(url)
        if live and live.get("ok") and live.get("chars", 0) > 500:
            sources.append({
                "title": live["title"],
                "url": live["requested_url"],
                "text": live["text"][:5000],
            })

    ranked = sorted(sources, key=lambda s: score_source(question, s), reverse=True)
    return ranked[:5], route

def make_context(sources):
    chunks = []
    for i, source in enumerate(sources, start=1):
        chunk = (
            "SOURCE " + str(i) + "\n"
            + "Title: " + source["title"] + "\n"
            + "URL: " + source["url"] + "\n"
            + "Text: " + source["text"][:3500]
        )
        chunks.append(chunk)
    return "\n\n---\n\n".join(chunks)

def generate_answer(question):
    sources, route = gather_sources(question)
    context = make_context(sources)

    prompt = (
        "You are Ask BIONEXT 2.0, a source-bounded research assistant.\n\n"
        "Answer the user's question using only the approved source material below. "
        "Do not use general knowledge. If the sources do not contain enough information, say so clearly. "
        "When citing sources, show the full source title and full URL, not just the domain. "
        "End with a section called 'Sources used'.\n\n"
        "User question:\n" + question + "\n\n"
        "Approved source material:\n" + context
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=900,
    )

    return response.choices[0].message.content, route, sources

def show_page_diagnostics(result):
    st.write("**Requested URL:**")
    st.write(result.get("requested_url", ""))
    st.write("**Final URL:**")
    st.write(result.get("final_url", ""))
    st.write("**Status code:**")
    st.write(result.get("status_code", ""))
    st.write("**Content type:**")
    st.write(result.get("content_type", ""))
    st.write("**Response bytes:**")
    st.write(result.get("response_bytes", 0))
    st.write("**Extracted characters:**")
    st.write(result.get("chars", 0))
    st.write("**Total links found:**")
    st.write(len(result.get("links", [])))
    st.write("**Approved links found:**")
    st.write(len(result.get("approved_links", [])))
    st.write("**Resource page links found:**")
    st.write(len(result.get("resource_links", [])))
    st.write("**PDF links found:**")
    st.write(len(result.get("pdf_links", [])))

    if result.get("error"):
        st.write("**Error:**")
        st.code(result.get("error", ""))

    if result.get("resource_links"):
        st.write("**Example resource links:**")
        for link in result.get("resource_links", [])[:5]:
            st.write(link)

    if result.get("pdf_links"):
        st.write("**Example PDF links:**")
        for link in result.get("pdf_links", [])[:5]:
            st.write(link)

    st.write("**Extracted text preview:**")
    st.text(result.get("text_preview", "")[:1000])

    st.write("**Raw response preview:**")
    st.text(result.get("raw_preview", "")[:1000])

def show_pdf_diagnostics(result):
    st.write("**Requested URL:**")
    st.write(result.get("requested_url", ""))
    st.write("**Final URL:**")
    st.write(result.get("final_url", ""))
    st.write("**Status code:**")
    st.write(result.get("status_code", ""))
    st.write("**Content type:**")
    st.write(result.get("content_type", ""))
    st.write("**Downloaded bytes:**")
    st.write(result.get("downloaded_bytes", 0))
    st.write("**First 20 bytes, hex:**")
    st.code(result.get("raw_start", ""))
    st.write("**Pages checked:**")
    st.write(result.get("pages_checked", 0))
    st.write
