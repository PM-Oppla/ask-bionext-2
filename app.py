import os
import re
import io
from urllib.parse import urljoin, urlparse

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

st.markdown(
    """
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
.agent {
    background: #f7fafb;
    border: 1px solid #d9e1e3;
    border-left: 5px solid var(--green);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 1rem;
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
""",
    unsafe_allow_html=True,
)

APPROVED_EXTERNAL_URLS = [
    "[cordis.europa.eu](https://cordis.europa.eu/project/id/101059662)",
    "[ipbes.net](https://www.ipbes.net/)",
    "[ict.ipbes.net](https://ict.ipbes.net/ipbes-ict-guide/data-and-knowledge-management/citations-of-ipbes-assessments/transformative-change-assessment)",
]

CORE_BIONEXT_URLS = [
    "[oppla.eu](https://oppla.eu/bionext)",
    "[oppla.eu](https://oppla.eu/bionext/portal)",
    "[oppla.eu](https://oppla.eu/bionext/about-bionext)",
    "[oppla.eu](https://oppla.eu/bionext/decision-analysis-environmental-decision-making)",
    "[oppla.eu](https://oppla.eu/bionext/bionext-news-articles)",
    "[oppla.eu](https://oppla.eu/bionext/bionext-resources)",
]

DECISION_URL = "[oppla.eu](https://oppla.eu/bionext/decision-analysis-environmental-decision-making)"
RESOURCES_URL = "[oppla.eu](https://oppla.eu/bionext/bionext-resources)"


def is_allowed_url(url):
    clean = url.split("#")[0]
    if clean.startswith("[oppla.eu](https://oppla.eu/bionext)"):
        return True
    if clean.startswith("[oppla.eu](https://oppla.eu/sites/default/files/)"):
        return True
    if clean in APPROVED_EXTERNAL_URLS:
        return True
    if clean.startswith(
        "[ict.ipbes.net](https://ict.ipbes.net/ipbes-ict-guide/data-and-knowledge-management/citations-of-ipbes-assessments/)"
    ):
        return True
    return False


def get_api_key():
    try:
        return st.secrets.get("OPENAI_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
    except Exception:
        return os.environ.get("OPENAI_API_KEY", "")


client = OpenAI(api_key=get_api_key())


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_html(url):
    if not is_allowed_url(url):
        return {"url": url, "title": "Blocked source", "text": "", "links": [], "error": "Blocked source"}

    try:
        headers = {"User-Agent": "AskBIONEXT2Prototype/1.0"}
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
    except Exception as e:
        return {"url": url,
