import hashlib
import io
import json
import re
import time

import pypdf
import requests

from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, unquote


# -------------------------------------------------------------------
# Ask BIONEXT 2.0
# Source Builder v5
#
# Sources:
# - recursively discovered pages within /bionext
# - authoritative BIONEXT resource API (group 202)
# - individual BIONEXT resource pages
# - PDFs linked from BIONEXT pages/resources
# - selected approved external sources
#
# Long documents are chunked for semantic retrieval.
# -------------------------------------------------------------------


BIONEXT_ROOT = "https://oppla.eu/bionext"

BIONEXT_RESOURCE_API = (
    "https://oppla.eu/api/resources/groups/202"
)

NODE_BASE = "https://oppla.eu/node/"


START_PAGES = [
    "https://oppla.eu/bionext",
    "https://oppla.eu/bionext/about-bionext",
    "https://oppla.eu/bionext/bionext-resources",
    (
        "https://oppla.eu/bionext/"
        "decision-analysis-environmental-decision-making"
    ),
    "https://oppla.eu/bionext/bionext-news-articles",
]


EXTERNAL_SOURCES = [
    {
        "title": "CORDIS BIONEXT project page",
        "url": "https://cordis.europa.eu/project/id/101059662",
        "source_type": "Approved external source",
    },
    {
        "title": (
            "IPBES Transformative Change Assessment "
            "citation page"
        ),
        "url": (
            "https://ict.ipbes.net/ipbes-ict-guide/"
            "data-and-knowledge-management/"
            "citations-of-ipbes-assessments/"
            "transformative-change-assessment"
        ),
        "source_type": "Approved external IPBES source",
    },
]


HEADERS = {
    "User-Agent": "AskBIONEXT2-SourceBuilder/5.0"
}

CHUNK_SIZE = 3000
CHUNK_OVERLAP = 350

MAX_BIONEXT_PAGES = 500

REQUEST_DELAY = 0.15


# -------------------------------------------------------------------
# HTTP helpers
# -------------------------------------------------------------------


def get_response(url, timeout=30):
    """
    Fetch a URL with a short courtesy delay.
    """

    time.sleep(REQUEST_DELAY)

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=timeout,
        allow_redirects=True,
    )

    response.raise_for_status()

    return response


# -------------------------------------------------------------------
# URL helpers
# -------------------------------------------------------------------


def normalise_url(url):
    """
    Normalise a URL for comparison and deduplication.
    """

    url = url.split("#")[0]

    parsed = urlparse(url)

    path = parsed.path

    if path != "/":
        path = path.rstrip("/")

    return parsed._replace(
        path=path,
        fragment="",
    ).geturl()


def is_bionext_url(url):
    """
    Return True only for pages within the Oppla BIONEXT area.
    """

    parsed = urlparse(url)

    if parsed.netloc not in [
        "oppla.eu",
        "www.oppla.eu",
    ]:
        return False

    return (
        parsed.path == "/bionext"
        or parsed.path.startswith("/bionext/")
    )


def looks_like_resource_page(url):
    """
    Identify an individual BIONEXT resource page.
    """

    parsed = urlparse(url)

    return (
        "/bionext/resource/"
        in parsed.path.lower()
    )


def looks_like_pdf(url):
    """
    Detect a PDF even if query parameters are present.
    """

    parsed = urlparse(url)

    return (
        parsed.path.lower().endswith(".pdf")
    )


# -------------------------------------------------------------------
# Text cleaning
# -------------------------------------------------------------------


def clean_text(text):
    """
    Clean extracted webpage or PDF text.
    """

    boilerplate = [
        "Skip to main content",
        "Group portal",
        "Decision Making Analysis",
        "News & Articles",
        "Contact",
    ]

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    for phrase in boilerplate:
        text = text.replace(
            phrase,
            " ",
        )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# -------------------------------------------------------------------
# Chunking
# -------------------------------------------------------------------


def chunk_text(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP,
):
    """
    Break long text into smaller overlapping chunks.
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

        start = max(
            end - overlap,
            start + 1,
        )

    return chunks


def add_chunked_source(
    output,
    title,
    url,
    source_type,
    text,
    metadata=None,
):
    """
    Add a document to the retrieval database.
    """

    chunks = chunk_text(text)

    if not chunks:
        return 0

    total_chunks = len(chunks)

    metadata = metadata or {}

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):

        if total_chunks == 1:
            chunk_title = title

        else:
            chunk_title = (
                f"{title} — section "
                f"{index} of {total_chunks}"
            )

        record = {
            "title": chunk_title,
            "parent_title": title,
            "url": url,
            "source_type": source_type,
            "chunk_number": index,
            "chunk_count": total_chunks,
            "text": chunk,
        }

        record.update(metadata)

        output.append(record)

    return total_chunks


# -------------------------------------------------------------------
# Web extraction
# -------------------------------------------------------------------


def extract_page(url):
    """
    Download a webpage and extract readable text.
    """

    try:

        response = get_response(
            url,
            timeout=30,
        )

        final_url = normalise_url(
            response.url
        )

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
            soup.title.get_text(
                " ",
                strip=True,
            )
            if soup.title
            else final_url
        )

        text = clean_text(
            soup.get_text(
                " ",
                strip=True,
            )
        )

        if len(text) < 100:
            return None

        return {
            "title": title,
            "url": final_url,
            "text": text,
        }

    except Exception as exc:

        print(
            f"Could not extract webpage "
            f"{url}: {exc}"
        )

        return None


# -------------------------------------------------------------------
# Link extraction
# -------------------------------------------------------------------


def get_links_from_page(url):
    """
    Return links found on a webpage.
    """

    links = set()

    try:

        response = get_response(
            url,
            timeout=30,
        )

        soup = BeautifulSoup(
            response.text,
            "html.parser",
        )

        for link in soup.find_all(
            "a",
            href=True,
        ):

            absolute_url = urljoin(
                response.url,
                link["href"],
            )

            links.add(
                normalise_url(
                    absolute_url
                )
            )

    except Exception as exc:

        print(
            f"Could not inspect links on "
            f"{url}: {exc}"
        )

    return sorted(links)


# -------------------------------------------------------------------
# Recursive BIONEXT crawl
# -------------------------------------------------------------------


def crawl_bionext_site():
    """
    Recursively discover public pages within /bionext.
    """

    queue = [
        normalise_url(url)
        for url in START_PAGES
    ]

    visited = set()
    discovered = set()

    while queue:

        if (
            len(visited)
            >= MAX_BIONEXT_PAGES
        ):

            print(
                "BIONEXT page safety "
                "limit reached."
            )

            break

        current_url = queue.pop(0)

        if current_url in visited:
            continue

        if not is_bionext_url(
            current_url
        ):
            continue

        visited.add(current_url)
        discovered.add(current_url)

        print(
            f"Discovering links: "
            f"{current_url}"
        )

        for link in get_links_from_page(
            current_url
        ):

            if not is_bionext_url(link):
                continue

            if looks_like_pdf(link):
                continue

            if link not in visited:
                queue.append(link)

    return sorted(discovered)


# -------------------------------------------------------------------
# Authoritative BIONEXT resource API
# -------------------------------------------------------------------


def get_bionext_api_resources():
    """
    Retrieve the complete current BIONEXT resource collection
    from the Oppla resource API.
    """

    print(
        "Reading BIONEXT resource API..."
    )

    try:

        response = get_response(
            BIONEXT_RESOURCE_API,
            timeout=30,
        )

        data = response.json()

        resources = data.get(
            "resources",
            [],
        )

        print(
            f"{len(resources)} resource "
            "record(s) returned by API"
        )

        return resources

    except Exception as exc:

        print(
            "Could not read BIONEXT "
            f"resource API: {exc}"
        )

        return []


def resolve_resource_page(resource):
    """
    Resolve an API resource nid to its canonical public
    BIONEXT resource URL.

    Drupal /node/{nid} redirects to the public alias.
    """

    nid = resource.get("nid")

    if not nid:
        return None

    node_url = (
        NODE_BASE
        + str(nid)
    )

    try:

        response = get_response(
            node_url,
            timeout=30,
        )

        final_url = normalise_url(
            response.url
        )

        if not is_bionext_url(
            final_url
        ):
            return None

        return final_url

    except Exception as exc:

        print(
            f"Could not resolve resource "
            f"{nid}: {exc}"
        )

        return None


def discover_api_resource_pages():
    """
    Return canonical resource pages plus API metadata.
    """

    api_resources = (
        get_bionext_api_resources()
    )

    discovered = []

    seen_urls = set()

    for resource in api_resources:

        nid = resource.get("nid")

        title = resource.get(
            "title",
            "",
        )

        print(
            f"Resolving API resource "
            f"{nid}: {title}"
        )

        url = resolve_resource_page(
            resource
        )

        if not url:
            continue

        if url in seen_urls:
            continue

        seen_urls.add(url)

        discovered.append({
            "url": url,
            "nid": nid,
            "api_title": title,
            "teaser_text": resource.get(
                "teaser_text",
                "",
            ),
            "publication_date": (
                resource.get(
                    "publication_date",
                    "",
                )
            ),
            "keywords": resource.get(
                "keywords",
                [],
            ),
            "resource_type_ids": (
                resource.get(
                    "resource_type",
                    [],
                )
            ),
        })

    return discovered


# -------------------------------------------------------------------
# PDF extraction
# -------------------------------------------------------------------


def pdf_title_from_url(url):
    """
    Create a readable title from a PDF filename.
    """

    filename = unquote(
        urlparse(url)
        .path
        .split("/")[-1]
    )

    if filename.lower().endswith(
        ".pdf"
    ):
        filename = filename[:-4]

    filename = filename.replace(
        "_",
        " ",
    )

    filename = filename.replace(
        "-",
        " ",
    )

    return filename.strip()


def extract_pdf(url):
    """
    Download a PDF and extract text using pypdf.
    """

    try:

        response = get_response(
            url,
            timeout=60,
        )

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
                        f"[Page {page_number}] "
                        f"{page_text}"
                    )

            except Exception as exc:

                print(
                    f"Could not read page "
                    f"{page_number} of "
                    f"{url}: {exc}"
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
            "title": pdf_title_from_url(
                url
            ),
            "url": url,
            "text": text,
        }

    except Exception as exc:

        print(
            f"Could not extract PDF "
            f"{url}: {exc}"
        )

        return None


# -------------------------------------------------------------------
# Deduplication
# -------------------------------------------------------------------


def text_fingerprint(text):
    """
    Create a fingerprint for duplicate detection.
    """

    normalised = re.sub(
        r"\s+",
        " ",
        text.lower(),
    ).strip()

    return hashlib.sha256(
        normalised.encode(
            "utf-8"
        )
    ).hexdigest()


# -------------------------------------------------------------------
# Build database
# -------------------------------------------------------------------


def build_sources():
    """
    Create the Ask BIONEXT source database.
    """

    raw_documents = []

    seen_page_urls = set()
    seen_pdf_content = set()

    print()
    print(
        "====================================="
    )
    print(
        "Building Ask BIONEXT 2.0 "
        "source database"
    )
    print(
        "====================================="
    )
    print()

    # ---------------------------------------------------------------
    # 1. Recursively discover BIONEXT pages
    # ---------------------------------------------------------------

    print(
        "Recursively discovering "
        "BIONEXT pages..."
    )
    print()

    bionext_pages = set(
        crawl_bionext_site()
    )

    print()
    print(
        f"{len(bionext_pages)} "
        "BIONEXT pages discovered "
        "by site crawl"
    )

    # ---------------------------------------------------------------
    # 2. Retrieve authoritative resource collection
    # ---------------------------------------------------------------

    print()
    print(
        "Discovering BIONEXT resources "
        "from API..."
    )
    print()

    api_resource_pages = (
        discover_api_resource_pages()
    )

    print()
    print(
        f"{len(api_resource_pages)} "
        "BIONEXT resource page(s) "
        "resolved from API"
    )

    resource_metadata = {}

    for item in api_resource_pages:

        url = item["url"]

        bionext_pages.add(url)

        resource_metadata[url] = item

    # ---------------------------------------------------------------
    # 3. Extract BIONEXT webpages
    # ---------------------------------------------------------------

    print()
    print(
        "Extracting BIONEXT webpages..."
    )
    print()

    pdf_links = set()

    for url in sorted(
        bionext_pages
    ):

        if url in seen_page_urls:
            continue

        print(
            f"Reading webpage: {url}"
        )

        extracted = extract_page(
            url
        )

        if extracted:

            final_url = extracted["url"]

            if final_url in seen_page_urls:
                continue

            metadata = (
                resource_metadata.get(
                    final_url,
                    {}
                )
            )

            if looks_like_resource_page(
                final_url
            ):

                source_type = (
                    "BIONEXT resource page"
                )

            elif (
                "decision-analysis"
                in final_url.lower()
                or "decision-making"
                in final_url.lower()
            ):

                source_type = (
                    "Oppla BIONEXT "
                    "decision-analysis page"
                )

            elif (
                "/article/"
                in final_url.lower()
            ):

                source_type = (
                    "BIONEXT article"
                )

            else:

                source_type = (
                    "Oppla BIONEXT page"
                )

            raw_documents.append({
                "title": extracted["title"],
                "url": final_url,
                "source_type": source_type,
                "text": extracted["text"],
                "metadata": {
                    "resource_nid": (
                        metadata.get("nid")
                    ),
                    "publication_date": (
                        metadata.get(
                            "publication_date",
                            "",
                        )
                    ),
                },
            })

            seen_page_urls.add(
                final_url
            )

        # Discover PDFs linked from this page.
        for link in get_links_from_page(
            url
        ):

            if looks_like_pdf(
                link
            ):

                pdf_links.add(
                    link
                )

    # ---------------------------------------------------------------
    # 4. Extract linked PDFs
    # ---------------------------------------------------------------

    print()
    print(
        "Reading linked PDFs..."
    )
    print()

    pdfs_extracted = 0
    pdfs_deduplicated = 0

    for pdf_url in sorted(
        pdf_links
    ):

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

        if fingerprint in seen_pdf_content:

            print(
                "Duplicate PDF skipped: "
                f"{pdf_url}"
            )

            pdfs_deduplicated += 1

            continue

        seen_pdf_content.add(
            fingerprint
        )

        raw_documents.append({
            "title": extracted["title"],
            "url": pdf_url,
            "source_type": "BIONEXT PDF",
            "text": extracted["text"],
            "metadata": {},
        })

        pdfs_extracted += 1

    # ---------------------------------------------------------------
    # 5. Approved external sources
    # ---------------------------------------------------------------

    print()
    print(
        "Reading approved external "
        "sources..."
    )
    print()

    for source in EXTERNAL_SOURCES:

        print(
            f"Reading: {source['url']}"
        )

        extracted = extract_page(
            source["url"]
        )

        if not extracted:
            continue

        raw_documents.append({
            "title": source["title"],
            "url": source["url"],
            "source_type": (
                source["source_type"]
            ),
            "text": extracted["text"],
            "metadata": {},
        })

    # ---------------------------------------------------------------
    # 6. Chunk for retrieval
    # ---------------------------------------------------------------

    print()
    print(
        "Creating retrieval chunks..."
    )
    print()

    sources = []

    for document in raw_documents:

        count = add_chunked_source(
            output=sources,
            title=document["title"],
            url=document["url"],
            source_type=(
                document["source_type"]
            ),
            text=document["text"],
            metadata=(
                document.get(
                    "metadata",
                    {}
                )
            ),
        )

        print(
            f"{document['title']}: "
            f"{count} chunk(s)"
        )

    # ---------------------------------------------------------------
    # 7. Save
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
    print(
        "====================================="
    )
    print(
        "Ask BIONEXT source database complete"
    )
    print(
        "====================================="
    )
    print()

    print(
        f"{len(bionext_pages)} "
        "total BIONEXT webpages known"
    )

    print(
        f"{len(api_resource_pages)} "
        "BIONEXT resources obtained "
        "from API"
    )

    print(
        f"{len(pdf_links)} "
        "PDF links discovered"
    )

    print(
        f"{pdfs_extracted} "
        "unique PDFs extracted"
    )

    print(
        f"{pdfs_deduplicated} "
        "duplicate PDFs skipped"
    )

    print(
        f"{len(raw_documents)} "
        "documents collected"
    )

    print(
        f"{len(sources)} "
        "retrieval chunks saved "
        "to sources.json"
    )


if __name__ == "__main__":
    build_sources()