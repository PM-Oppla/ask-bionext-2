import os
import re
import json
import requests
import streamlit as st
from bs4 import BeautifulSoup
from openai import OpenAI

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
        "url": "[oppla.eu](https://oppla.eu/bionext/bionext-resources)",
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
    "[oppla.eu](https://oppla.eu/bionext/bionext-resources)",
    "[oppla.eu](https://oppla.eu/bionext/decision-analysis-environmental-decision-making)",
    "[cordis.europa.eu](https://cordis.europa.eu/project/id/101059662)",
]

def load_sources_from_json():
    try:
        with open("sources.json", "r", encoding="utf-8") as f:
            sources = json.load(f)

        clean_sources = []
        for source in sources:
            title = source.get("title", "").strip()
            url = source.get("url", "").strip()
            text = source.get("text", "").strip()

            if title and url and text:
                clean_sources.append({
                    "title": title,
                    "url": url,
                    "text": text,
                    "source_type": source.get("source_type", "Approved source"),
                })

        return clean_sources

    except Exception:
        return []

def get_api_key():
    try:
        return st.secrets.get("OPENAI_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
    except Exception:
        return os.environ.get("OPENAI_API_KEY", "")

client = OpenAI(api_key=get_api_key())

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
    stop_words = {
        "what", "when", "where", "which", "about", "does",
        "from", "with", "that", "this", "there", "their",
        "your", "allowed", "sources", "source", "use", "using"
    }
    words = [w for w in words if w not in stop_words]

    haystack = (
        source.get("title", "") + " "
        + source.get("url", "") + " "
        + source.get("text", "")
    ).lower()

    return sum(haystack.count(w) for w in words)

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_live_text(url):
    try:
        r = requests.get(
            url,
            headers={"User-Agent": "AskBIONEXT2"},
            timeout=15,
        )
        r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        title = soup.title.get_text(" ", strip=True) if soup.title else url
        text = soup.get_text("\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return {
            "title": title,
            "url": url,
            "text": text[:5000],
            "source_type": "Live web extraction",
        }

    except Exception:
        return None

def gather_sources(question):
    route = route_question(question)

    sources = load_sources_from_json()
    if not sources:
        sources = list(APPROVED_SOURCES)

    for url in LIVE_URLS:
        live = fetch_live_text(url)
        if live and len(live["text"]) > 500:
            sources.append(live)

    ranked = sorted(sources, key=lambda s: score_source(question, s), reverse=True)
    return ranked[:5], route

def make_context(sources):
    chunks = []

    for i, source in enumerate(sources, start=1):
        source_type = source.get("source_type", "Approved source")

        chunk = (
            "SOURCE " + str(i) + "\n"
            + "Title: " + source["title"] + "\n"
            + "URL: " + source["url"] + "\n"
            + "Source type: " + source_type + "\n"
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
        "When listing citations, use the full source title and full URL. "
        "End every answer with a section called 'Sources used'.\n\n"
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

st.markdown("""
<div class="header">
  <h1>Ask BIONEXT 2.0</h1>
  <p>Agentic prototype for source-bounded BIONEXT research exploration</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("Ask BIONEXT 2.0")
    st.write("This prototype answers from the approved Ask BIONEXT 2.0 source database.")

    st.markdown("### Approved source areas")
    st.write("- BIONEXT on Oppla")
    st.write("- BIONEXT resources")
    st.write("- BIONEXT decision-analysis pages")
    st.write("- CORDIS BIONEXT project page")
    st.write("- selected IPBES page")

    st.markdown("---")
    st.markdown("### Source database")

    loaded_sources = load_sources_from_json()
    if loaded_sources:
        st.success(f"{len(loaded_sources)} sources loaded from sources.json")
    else:
        st.warning("Using built-in fallback sources")

st.write("Ask a question about BIONEXT. The app will choose a source route and answer with citations.")

question = st.text_input(
    "Question",
    placeholder="e.g. What is BIONEXT and what problem is it trying to address?",
)

if st.button("Ask BIONEXT") and question.strip():
    if not get_api_key():
        st.error("No OpenAI API key found. Add OPENAI_API_KEY in Streamlit Secrets.")
    else:
        with st.spinner("Searching approved BIONEXT sources..."):
            try:
                answer, route, sources = generate_answer(question)

                st.markdown(f"**Agent route:** {route}")
                st.markdown(f'<div class="answer">{answer}</div>', unsafe_allow_html=True)

                with st.expander("Sources checked"):
                    for source in sources:
                        st.write(f"**{source['title']}**")
                        st.write(source["url"])
                        st.caption(source.get("source_type", "Approved source"))

            except Exception as e:
                st.error(f"Something went wrong: {e}")
