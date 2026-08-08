import hashlib
import io
import json
import re
import requests
import pypdf

from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, unquote


# -------------------------------------------------------------------
# Ask BIONEXT 2.0
# Source Builder v3
#
# Builds sources.json from:
# - approved BIONEXT / Oppla pages
# - linked BIONEXT pages
# - linked PDF documents
# - approved external sources
#
# Long documents are broken into smaller chunks to improve retrieval.
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
    "User-Agent": "AskBIONEXT2-SourceBuilder/3.0"
}


# Aim to keep each retrieval unit below the amount currently passed
# from each source into the Ask BIONEXT prompt.
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 350


def clean_text(text):
    """Clean extracted webpage or PDF text."""

    boilerplate = [
        "Skip to main content",
        "Group portal",
        "Decision Making Analysis",
        "News & Articles",
        "Contact",
    ]

    text = re.sub(r"\s+", " ", text)

    for phrase in boilerplate:
        text = text.replace(phrase, " ")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Break long text into smaller overlapping chunks.

    Tries to finish chunks at a sentence boundary where possible.
    """

    text = clean_text(text)

    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        target_end = min(
            start + chunk_size,
            text_length,
        )

        end = target_end

        # If this is not the end of the document, try to stop at
        # a natural sentence boundary.
        if target_end < text_length:
            search_start = max(
                start + int(chunk_size * 0.65),
                start,
            )

            candidate = text.rfind(
                ". ",
                search_start,
                target_end,
            )

            if candidate != -1:
                end = candidate + 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        new_start = max(
            end - overlap,
            start + 1,
        )

        start = new_start

    return chunks


def add_chunked_source(
    output,
    title,
    url,
    source_type,
    text,
):
    """
    Add a document to the output database.

    Short documents remain as one source.
    Long documents become multiple retrieval chunks.
    """

    chunks = chunk_text(text)

    if not chunks:
        return 0

    total_chunks = len(chunks)

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):
        if total_chunks == 1:
            chunk_title = title
        else:
            chunk_title = (
                f"{title} — section {index} of {total_chunks}"
            )

        output.append({
            "title": chunk_title,
            "parent_title": title,
            "url": url,
            "source_type": source_type,
            "chunk_number": index,
            "chunk_count": total_chunks,
            "text": chunk,
        })

    return total_chunks


def extract_page(url):
    """
    Download an approved webpage and extract readable text.
    Returns None if extraction fails.
    """

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=30,
        )
        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser",
        )

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
        print(
            f"Could not extract webpage {url}: {exc}"
        )
        return None


def get_pdf_links(url):
    """
    Find PDF documents linked from a webpage.
    """

    pdf_links = set()

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=30,
        )
        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser",
        )

        for link in soup.find_all(
            "a",
            href=True,
        ):
            absolute_url = urljoin(
                url,
                link["href"],
            )

            clean_url = absolute_url.split("#")[0]

            parsed = urlparse(clean_url)

            if parsed.path.lower().endswith(".pdf"):
                pdf_links.add(clean_url)

    except Exception as exc:
        print(
            f"Could not inspect PDFs on {url}: {exc}"
        )

    return sorted(pdf_links)


def pdf_title_from_url(url):
    """
    Create a readable title from the PDF filename.
    """

    filename = unquote(
        urlparse(url).path.split("/")[-1]
    )

    if filename.lower().endswith(".pdf"):
        filename = filename[:-4]

    filename = filename.replace("_", " ")
    filename = filename.replace("-", " ")

    return filename.strip()


def extract_pdf(url):
    """
    Download a PDF and extract its text using pypdf.
    Returns None if extraction fails.
    """

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=60,
        )
        response.raise_for_status()

        pdf_file = io.BytesIO(
            response.content
        )

        reader = pypdf.PdfReader(
            pdf_file
        )

        pages = []

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):
            try:
                page_text = (
                    page.extract_text()
                    or ""
                )

                page_text = clean_text(
                    page_text
                )

                if page_text:
                    pages.append(
                        f"[Page {page_number}] {page_text}"
                    )

            except Exception as exc:
                print(
                    f"Could not read page "
                    f"{page_number} of {url}: {exc}"
                )

        text = clean_text(
            " ".join(pages)
        )

        if len(text) < 100:
            print(
                "PDF contained too little "
                f"extractable text: {url}"
            )
            return None

        return {
            "title": pdf_title_from_url(url),
            "url": url,
            "text": text,
        }

    except Exception as exc:
        print(
            f"Could not extract PDF {url}: {exc}"
        )
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
            timeout=30,
        )
        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser",
        )

        for link in soup.find_all(
            "a",
            href=True,
        ):
            absolute_url = urljoin(
                url,
                link["href"],
            )

            parsed = urlparse(
                absolute_url
            )

            if parsed.netloc not in [
                "oppla.eu",
                "www.oppla.eu",
            ]:
                continue

            if "/bionext" not in parsed.path.lower():
                continue

            clean_url = (
                absolute_url
                .split("#")[0]
            )

            links.add(clean_url)

    except Exception as exc:
        print(
            "Could not inspect BIONEXT links "
            f"on {url}: {exc}"
        )

    return sorted(links)


def text_fingerprint(text):
    """
    Create a fingerprint used to identify duplicate documents.
    """

    normalised = re.sub(
        r"\s+",
        " ",
        text.lower(),
    ).strip()

    return hashlib.sha256(
        normalised.encode("utf-8")
    ).hexdigest()


def build_sources():
    """Create the Ask BIONEXT source database."""

    raw_documents = []
    seen_urls = set()
    seen_content = set()

    print()
    print("=====================================")
    print("Building Ask BIONEXT 2.0 source database")
    print("=====================================")
    print()

    # ---------------------------------------------------------------
    # 1. Approved starting pages
    # ---------------------------------------------------------------

    print("Reading approved BIONEXT pages...")
    print()

    for source in START_PAGES:
        print(
            f"Reading: {source['url']}"
        )

        extracted = extract_page(
            source["url"]
        )

        if extracted:
            raw_documents.append({
                "title": source["title"],
                "url": source["url"],
                "source_type": source["source_type"],
                "text": extracted["text"],
            })

            seen_urls.add(
                source["url"]
            )

    # ---------------------------------------------------------------
    # 2. Discover linked BIONEXT pages
    # ---------------------------------------------------------------

    print()
    print("Discovering linked BIONEXT pages...")

    discovered_links = get_bionext_links(
        "https://oppla.eu/bionext"
    )

    print(
        f"Found {len(discovered_links)} linked BIONEXT pages"
    )

    # ---------------------------------------------------------------
    # 3. Discover linked PDFs
    # ---------------------------------------------------------------

    print()
    print("Discovering linked PDF documents...")

    pdf_links = set()

    pages_to_check_for_pdfs = (
        [source["url"] for source in START_PAGES]
        + discovered_links
    )

    for url in pages_to_check_for_pdfs:
        discovered_pdfs = get_pdf_links(
            url
        )

        for pdf_url in discovered_pdfs:
            pdf_links.add(pdf_url)

    print(
        f"Found {len(pdf_links)} PDF links"
    )

    # ---------------------------------------------------------------
    # 4. Extract linked BIONEXT pages
    # ---------------------------------------------------------------

    print()
    print("Reading linked BIONEXT pages...")
    print()

    for url in discovered_links:
        if url in seen_urls:
            continue

        print(
            f"Reading linked page: {url}"
        )

        extracted = extract_page(
            url
        )

        if not extracted:
            continue

        raw_documents.append({
            "title": extracted["title"],
            "url": url,
            "source_type": "Linked Oppla BIONEXT page",
            "text": extracted["text"],
        })

        seen_urls.add(url)

    # ---------------------------------------------------------------
    # 5. Extract and deduplicate PDFs
    # ---------------------------------------------------------------

    print()
    print("Reading linked PDF documents...")
    print()

    pdfs_extracted = 0
    pdfs_deduplicated = 0

    for pdf_url in sorted(pdf_links):
        print(
            f"Reading PDF: {pdf_url}"
        )

        extracted = extract_pdf(
            pdf_url
        )

        if not extracted:
            continue

        fingerprint = text_fingerprint(
            extracted["text"]
        )

        if fingerprint in seen_content:
            print(
                "Duplicate PDF skipped: "
                f"{pdf_url}"
            )
            pdfs_deduplicated += 1
            continue

        seen_content.add(
            fingerprint
        )

        raw_documents.append({
            "title": extracted["title"],
            "url": pdf_url,
            "source_type": "BIONEXT PDF",
            "text": extracted["text"],
        })

        pdfs_extracted += 1

    # ---------------------------------------------------------------
    # 6. Approved external sources
    # ---------------------------------------------------------------

    print()
    print("Reading approved external sources...")
    print()

    for source in EXTERNAL_SOURCES:
        print(
            f"Reading: {source['url']}"
        )

        extracted = extract_page(
            source["url"]
        )

        if extracted:
            raw_documents.append({
                "title": source["title"],
                "url": source["url"],
                "source_type": source["source_type"],
                "text": extracted["text"],
            })

    # ---------------------------------------------------------------
    # 7. Chunk documents for retrieval
    # ---------------------------------------------------------------

    print()
    print("Creating retrieval chunks...")
    print()

    sources = []

    for document in raw_documents:
        count = add_chunked_source(
            output=sources,
            title=document["title"],
            url=document["url"],
            source_type=document["source_type"],
            text=document["text"],
        )

        print(
            f"{document['title']}: "
            f"{count} chunk(s)"
        )

    # ---------------------------------------------------------------
    # 8. Save sources.json
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
    print("=====================================")
    print("Ask BIONEXT source database complete")
    print("=====================================")
    print()
    print(
        f"{len(raw_documents)} unique documents collected"
    )
    print(
        f"{len(sources)} retrieval chunks saved to sources.json"
    )
    print(
        f"{len(pdf_links)} PDF links discovered"
    )
    print(
        f"{pdfs_extracted} unique PDFs extracted"
    )
    print(
        f"{pdfs_deduplicated} duplicate PDFs skipped"
    )


if __name__ == "__main__":
    build_sources()