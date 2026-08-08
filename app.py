import os
import re
import io
from urllib.parse import urljoin

import streamlit as st
import requests
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
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

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

def allowed(url):
    return (
        url.startswith("[oppla.eu](https://oppla.eu/bionext)")
        or url.startswith("[oppla.eu](https://oppla.eu/sites/default/files/)")
        or url in CORE_URLS
        or url.startswith("[ict.ipbes.net](https://ict.ipbes.net/ipbes-ict-guide/data-and-knowledge-management/citations-of-ipbes-assessments/)")
    )

def api_key():
    try:
        return st.secrets.get("OPENAI_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
    except Exception:
        return os.environ.get("OPENAI_API_KEY", "")

client = OpenAI(api_key=api_key())

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_page(url):
    if not allowed(url):
        return {"url": url, "title": "Blocked source", "text": "", "links": []}

    try:
        r = requests.get(url, headers={"User-Agent": "AskBIONEXT2"}, timeout=20)
        r.raise_for_status()
    except Exception:
        return {"url": url, "title": url, "text": "", "links": []}

    content_type = r.headers.get("content-type", "").lower()

    if "pdf" in content_type or url.lower().endswith(".pdf"):
        try:
            reader = pypdf.PdfReader(io.BytesIO(r.content))
            text = "\n\n".join((p.extract_text() or "") for p in reader.pages[:25])
            return {"url": url, "title": url.split("/")[-1], "text": text, "links": []}
        except Exception:
            return {"url": url, "title": url, "text": "", "links": []}

    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else url
    text = soup.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)

    links = []
    for a in soup.find_all("a", href=True):
        u = urljoin(url, a["href"]).split("#")[0]
        if allowed(u):
            links.append(u)

    links = list(dict.fromkeys(links))
    return {"url": url, "title": title, "text": text, "links": links[:30]}

def route(question):
    q = question.lower()

    if any(x in q for x in ["decision", "mcda", "multi-criteria", "multicriteria", "criteria"]):
        return [
            "[oppla.eu](https://oppla.eu/bionext/decision-analysis-environmental-decision-making)",
            "[oppla.eu](https://oppla.eu/bionext/about-bionext)",
        ], "Decision analysis route"

    if any(x in q for x in ["resource", "report", "policy brief", "deliverable", "pdf", "output"]):
        return [
            "[oppla.eu](https://oppla.eu/bionext/bionext-resources)",
            "[oppla.eu](https://oppla.eu/bionext)",
        ], "Resources route"

    if any(x in q for x in ["ipbes", "transformative change", "assessment"]):
        return [
            "[oppla.eu](https://oppla.eu/bionext)",
            "[ipbes.net](https://www.ipbes.net/)",
            "[ict.ipbes.net](https://ict.ipbes.net/ipbes-ict-guide/data-and-knowledge-management/citations-of-ipbes-assessments/transformative-change-assessment)",
        ], "IPBES and transformative change route"

    if any(x in q for x in ["cordis", "grant", "funding", "eu project"]):
        return [
            "[cordis.europa.eu](https://cordis.europa.eu/project/id/101059662)",
            "[oppla.eu](https://oppla.eu/bionext/about-bionext)",
        ], "CORDIS route"

    return [
        "[oppla.eu](https://oppla.eu/bionext)",
        "[oppla.eu](https://oppla.eu/bionext/portal)",
        "[oppla.eu](https://oppla.eu/bionext/about-bionext)",
    ], "General BIONEXT route"

def score(question, item):
    words = re.findall(r"[a-zA-Z]{4,}", question.lower())
    haystack = (item["title"] + "\n" + item["url"] + "\n" + item["text"]).lower()
    return sum(haystack.count(w) for w in words)

def gather(question):
    seeds, label = route(question)
    items = []

    for url in seeds:
        item = fetch_page(url)
        items.append(item)

    follow_links = []
    for item in items:
        for link in item["links"]:
            if link not in seeds:
                follow_links.append(link)

    follow_links = list(dict.fromkeys(follow_links))[:5]

    for link in follow_links:
        items.append(fetch_page(link))

    items = [i for i in items if i["text"]]
    items = sorted(items, key=lambda x: score(question, x), reverse=True)
    return items[:5], label

def answer_question(question):
    evidence, route_label = gather(question)

    if not evidence:
        return "I could not find enough information in the approved Ask BIONEXT 2.0 sources to answer that question.", route_label, []

    context_parts = []
    for i, item in enumerate(evidence, start=1):
        context_parts.append(
            f"SOURCE {i}: {item['title']}\nURL: {item['url']}\nTEXT:\n{item['text'][:5000]}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""
You are Ask BIONEXT 2.0, a source-bounded research assistant.

Answer the user's question using only the source material below.
Do not use general knowledge.
If the sources do not contain enough information, say so clearly.
End with a 'Sources used' list containing the exact source titles and URLs used.

User question:
{question}

Approved source material:
{context}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1000,
    )

    return response.choices[0].message.content, route_label, evidence

st.markdown("""
<div class="header">
  <h1>Ask BIONEXT 2.0</h1>
  <p>Agentic prototype for source-bounded BIONEXT research exploration</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("Ask BIONEXT 2.0")
    st.write("This prototype searches only approved BIONEXT and selected external sources.")
    st.markdown("### Approved source areas")
    st.write("- BIONEXT on Oppla")
    st.write("- BIONEXT resources and linked Oppla PDFs")
    st.write("- BIONEXT decision-analysis pages")
    st.write("- CORDIS BIONEXT project page")
    st.write("- selected IPBES pages")

st.write(
    "Ask a question about BIONEXT. The app will choose a source route, retrieve approved source material, and answer with citations."
)

question = st.text_input(
    "Question",
    placeholder="e.g. What is BIONEXT and what problem is it trying to address?",
)

if st.button("Ask BIONEXT") and question.strip():
    if not api_key():
        st.error("No OpenAI API key found. Add OPENAI_API_KEY in Streamlit Secrets.")
    else:
        with st.spinner("Searching approved BIONEXT sources..."):
            try:
                answer, route_label, evidence = answer_question(question)
                st.markdown(f"**Agent route:** {route_label}")
                st.markdown(f'<div class="answer">{answer}</div>', unsafe_allow_html=True)

                with st.expander("Sources checked"):
                    for item in evidence:
                        st.write(f"**{item['title']}**")
                        st.write(item["url"])
            except Exception as e:
                st.error(f"Something went wrong: {e}")
