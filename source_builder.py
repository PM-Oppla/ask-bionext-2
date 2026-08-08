import json
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse


# -------------------------------------------------------------------
# Ask BIONEXT 2.0
# Source Builder v1
#
# This script creates sources.json from a controlled list of
# approved BIONEXT / Oppla pages.
# -------------------------------------------------------------------


START_PAGES = [
    {
        "title": "BIONEXT on Oppla",
        "url": "https://oppla.eu/bionext",
        "source_type": "Oppla BIONEXT page",
    },
    {
        "title": "About BIONEXT",
        "url": "https://oppla.eu/bionext/about-bionext",
        "source_type": "Oppla BIONEXT page",
    },
    {
        "title": "BIONEXT Resources",
        "url": "https://oppla.eu/bionext/bionext-resources",
        "source_type": "Oppla BIONEXT resources page",
    },
    {
        "title": "Decision analysis in environmental decision-making",
        "url": "https://oppla.eu/bionext/decision-analysis-environmental-decision-making",
        "source_type": "Oppla BIONEXT decision-analysis page",
    },
    {
        "title": "BIONEXT News Articles",
        "url": "https://oppla.eu/bionext/bionext-news-articles",
        "source_type": "Oppla BIONEXT news page",
    },
]


EXTERNAL_SOURCES = [
    {
        "title": "CORDIS BIONEXT project page",
        "url": "https://cordis.europa.eu/project/id/101059662",
        "source_type": "Approved external source",
    },
    {
        "title": "IPBES Transformative Change Assessment citation page",
        "url": "https://ict.ipbes.net/ipbes-ict-guide/data-and-knowledge-management/citations-of-ipbes-assessments/transformative-change-assessment",
        "source_type": "Approved external IPBES source",
    },
]


HEADERS = {
    "User-Agent": "AskBIONEXT2-SourceBuilder/1.0"
}


def clean_text(text):
    """Clean extracted webpage text."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_page(url):
    """
    Download an approved webpage and extract readable text.
    Returns None if extraction fails.
    """
    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=20,
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove content that is normally irrelevant to answers
        for tag in soup([
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "form",
            "noscript",
        ]):
            tag.decompose()

        title = (
            soup.title.get_text(" ", strip=True)
            if soup.title
            else url
        )

        text = clean_text(
            soup.get_text(" ", strip=True)
        )

        if len(text) < 100:
            return None

        return {
            "title": title,
            "url": url,
            "text": text,
        }

    except Exception as exc:
        print(f"Could not extract {url}: {exc}")
        return None


def get_bionext_links(url):
    """
    Find links on an approved BIONEXT page that remain within
    the Oppla BIONEXT area.
    """
    links = set()

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=20,
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for link in soup.find_all("a", href=True):
            absolute_url = urljoin(url, link["href"])

            parsed = urlparse(absolute_url)

            if parsed.netloc not in [
                "oppla.eu",
                "www.oppla.eu",
            ]:
                continue

            # Only allow URLs clearly related to BIONEXT
            if "/bionext" not in parsed.path.lower():
                continue

            # Remove fragments
            clean_url = absolute_url.split("#")[0]

            links.add(clean_url)

    except Exception as exc:
        print(f"Could not inspect links on {url}: {exc}")

    return sorted(links)


def build_sources():
    """Create the Ask BIONEXT source database."""

    sources = []
    seen_urls = set()

    print("Building Ask BIONEXT 2.0 source database...")
    print()

    # ---------------------------------------------------------------
    # 1. Extract the approved starting pages
    # ---------------------------------------------------------------

    for source in START_PAGES:

        print(f"Reading: {source['url']}")

        extracted = extract_page(source["url"])

        if extracted:

            sources.append({
                "title": source["title"],
                "url": source["url"],
                "source_type": source["source_type"],
                "text": extracted["text"],
            })

            seen_urls.add(source["url"])

    # ---------------------------------------------------------------
    # 2. Discover additional BIONEXT pages from the main portal
    # ---------------------------------------------------------------

    print()
    print("Discovering linked BIONEXT pages...")

    discovered_links = get_bionext_links(
        "https://oppla.eu/bionext"
    )

    for url in discovered_links:

        if url in seen_urls:
            continue

        print(f"Reading linked page: {url}")

        extracted = extract_page(url)

        if not extracted:
            continue

        sources.append({
            "title": extracted["title"],
            "url": url,
            "source_type": "Linked Oppla BIONEXT page",
            "text": extracted["text"],
        })

        seen_urls.add(url)

    # ---------------------------------------------------------------
    # 3. Add approved external sources
    # ---------------------------------------------------------------

    print()
    print("Reading approved external sources...")

    for source in EXTERNAL_SOURCES:

        print(f"Reading: {source['url']}")

        extracted = extract_page(source["url"])

        if extracted:

            sources.append({
                "title": source["title"],
                "url": source["url"],
                "source_type": source["source_type"],
                "text": extracted["text"],
            })

            seen_urls.add(source["url"])

    # ---------------------------------------------------------------
    # 4. Save sources.json
    # ---------------------------------------------------------------

    with open(
        "sources.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            sources,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("-------------------------------------")
    print("Ask BIONEXT source database complete")
    print("-------------------------------------")
    print(f"{len(sources)} sources saved to sources.json")


if __name__ == "__main__":
    build_sources()
