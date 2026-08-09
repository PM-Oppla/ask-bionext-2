import json
import math
import os
import re
import time

import streamlit as st
from openai import OpenAI


# -------------------------------------------------------------------
# Ask BIONEXT 2.0
# App v8
#
# Focus of this version:
# - source-bounded answers
# - chunk-based retrieval
# - rate-limit-safe OpenAI embedding search
# - hybrid semantic + keyword ranking
# - BIONEXT-first source weighting
# - route-aware source weighting
# - source-type filtering
# - grouped citations
# - two-stage evidence gate for weak / unsupported questions
#
# Visual branding will be refined separately using the official
# BIONEXT brand assets and guidance.
# -------------------------------------------------------------------


st.set_page_config(
    page_title="Ask BIONEXT 2.0",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------------------------------------------------
# Styling
# -------------------------------------------------------------------


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
      padding: 1.2rem 1.4rem;
      line-height: 1.65;
    }

    .prototype-note {
      background: #f7f9f9;
      border-left: 4px solid var(--green);
      padding: 0.75rem 1rem;
      margin: 0.5rem 0 1.25rem 0;
      border-radius: 4px;
      font-size: 0.95rem;
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
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------


EMBEDDING_MODEL = "text-embedding-3-small"
ANSWER_MODEL = "gpt-4o-mini"
ANSWERABILITY_MODEL = "gpt-4o-mini"

# The OpenAI embeddings endpoint has a per-request token limit.
# The knowledge base is now large enough that embedding every chunk in one
# request can exceed that limit, so source chunks are embedded in batches.
EMBEDDING_BATCH_SIZE = 50

# Only a compact representation of each retrieval chunk is embedded.
# This keeps total embedding-token usage comfortably below API TPM limits
# while retaining the chunk title, source type, and a substantial text sample.
EMBEDDING_TEXT_MAX_CHARS = 1500
EMBEDDING_MAX_RETRIES = 4
EMBEDDING_RETRY_BASE_SECONDS = 2

TOP_CHUNKS = 10
MAX_CHUNKS_PER_DOCUMENT = 2

# BIONEXT-first retrieval: a modest positive prior means that when
# BIONEXT and external chunks are similarly relevant, BIONEXT wins.
BIONEXT_BASE_BONUS = 0.045

# External evidence remains available, but is prevented from dominating
# general BIONEXT queries simply because there are many more IPBES chunks.
DEFAULT_MAX_EXTERNAL_CHUNKS = 4
IPBES_ROUTE_MAX_EXTERNAL_CHUNKS = 8

# Only reject a query when the best retrieval result is extremely weak.
MIN_TOP_RETRIEVAL_SCORE = 0.18

# Hard limits on evidence sent to the answer model.
# These prevent large source documents from creating oversized prompts.
MAX_CHARS_PER_CHUNK = 1200
MAX_CONTEXT_CHARS = 12000
MAX_SOURCES_IN_PROMPT = 8


# -------------------------------------------------------------------
# API
# -------------------------------------------------------------------


def get_api_key():
    try:
        return (
            st.secrets.get("OPENAI_API_KEY", "")
            or os.environ.get("OPENAI_API_KEY", "")
        )
    except Exception:
        return os.environ.get(
            "OPENAI_API_KEY",
            "",
        )


client = OpenAI(
    api_key=get_api_key()
)


# -------------------------------------------------------------------
# Source database
# -------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_sources():
    try:
        with open(
            "sources.json",
            "r",
            encoding="utf-8",
        ) as file:
            raw_sources = json.load(file)

        sources = []

        for source in raw_sources:
            title = (
                source.get(
                    "title",
                    "",
                ).strip()
            )

            parent_title = (
                source.get(
                    "parent_title",
                    title,
                ).strip()
            )

            url = (
                source.get(
                    "url",
                    "",
                ).strip()
            )

            source_type = (
                source.get(
                    "source_type",
                    "Approved source",
                ).strip()
            )

            text = (
                source.get(
                    "text",
                    "",
                ).strip()
            )

            if not title or not url or not text:
                continue

            sources.append({
                "title": title,
                "parent_title": parent_title,
                "url": url,
                "source_type": source_type,
                "chunk_number": source.get(
                    "chunk_number",
                    1,
                ),
                "chunk_count": source.get(
                    "chunk_count",
                    1,
                ),
                "text": text,
            })

        return sources

    except Exception:
        return []


# -------------------------------------------------------------------
# Routing
# -------------------------------------------------------------------


def route_question(question):
    q = question.lower()

    # Explicit IPBES wording should override the generic phrase
    # "transformative change", because BIONEXT also uses that phrase.
    if any(
        term in q
        for term in [
            "ipbes",
            "transformative change assessment",
            "summary for policymakers",
            "chapter 1",
            "chapter 2",
            "chapter 3",
            "chapter 4",
            "chapter 5",
        ]
    ):
        return "IPBES and transformative change route"

    if any(
        term in q
        for term in [
            "decision",
            "mcda",
            "multi-criteria",
            "criteria",
            "trade-off",
            "tradeoff",
            "impact assessment",
            "stakeholder preferences",
            "sensitivity analysis",
        ]
    ):
        return "Decision-analysis route"

    if any(
        term in q
        for term in [
            "resource",
            "policy brief",
            "report",
            "deliverable",
            "publication",
            "output",
            "pdf",
            "fact sheet",
        ]
    ):
        return "BIONEXT resources route"

    if any(
        term in q
        for term in [
            "cordis",
            "grant",
            "funding",
            "eu project",
            "101059662",
        ]
    ):
        return "CORDIS project-information route"

    return "General BIONEXT route"


# -------------------------------------------------------------------
# Source classification and weighting
# -------------------------------------------------------------------


def is_bionext_source(source):
    source_type = source["source_type"].lower()
    return (
        "bionext" in source_type
        or "oppla bionext" in source_type
    )


def is_ipbes_source(source):
    return "ipbes" in source["source_type"].lower()


def is_cordis_source(source):
    return (
        "approved external source"
        in source["source_type"].lower()
        and "cordis" in source["parent_title"].lower()
    )


def source_priority_bonus(route, source):
    """
    Give BIONEXT sources a modest default advantage while still allowing
    route-specific external evidence to rise to the top when appropriate.
    """

    bonus = 0.0

    if is_bionext_source(source):
        bonus += BIONEXT_BASE_BONUS

    source_type = source["source_type"].lower()

    if route == "Decision-analysis route":
        if "decision-analysis" in source_type:
            bonus += 0.060
        elif is_bionext_source(source):
            bonus += 0.015

    elif route == "BIONEXT resources route":
        if "bionext resource page" in source_type:
            bonus += 0.055
        elif "bionext pdf" in source_type:
            bonus += 0.045
        elif is_bionext_source(source):
            bonus += 0.015

    elif route == "IPBES and transformative change route":
        if is_ipbes_source(source):
            bonus += 0.075
        elif is_bionext_source(source):
            bonus += 0.010

    elif route == "CORDIS project-information route":
        if is_cordis_source(source):
            bonus += 0.090
        elif is_bionext_source(source):
            bonus += 0.010

    return bonus


# -------------------------------------------------------------------
# Compact text used for semantic embeddings
# -------------------------------------------------------------------


def make_embedding_text(source):
    """
    Build a compact semantic representation of a source chunk.

    The complete chunk remains available for answer generation. Only the
    representation sent to the embeddings endpoint is shortened.
    """

    body = source["text"]

    if len(body) > EMBEDDING_TEXT_MAX_CHARS:
        head_chars = int(EMBEDDING_TEXT_MAX_CHARS * 0.72)
        tail_chars = EMBEDDING_TEXT_MAX_CHARS - head_chars

        body = (
            body[:head_chars]
            + "\n...\n"
            + body[-tail_chars:]
        )

    return (
        f"Title: {source['parent_title']}\n"
        f"Source type: {source['source_type']}\n"
        f"Text: {body}"
    )


# -------------------------------------------------------------------
# Embeddings
# -------------------------------------------------------------------


@st.cache_data(
    show_spinner=False,
    ttl=3600,
)
def create_source_embeddings(texts):
    """
    Create embeddings for retrieval chunks in safe batches.

    Each text supplied here is already compacted by make_embedding_text().
    Requests are also retried with exponential backoff if the API temporarily
    reports a rate-limit response.
    """

    if not texts:
        return []

    embeddings = []

    for start in range(
        0,
        len(texts),
        EMBEDDING_BATCH_SIZE,
    ):
        batch = texts[
            start:start + EMBEDDING_BATCH_SIZE
        ]

        last_error = None

        for attempt in range(EMBEDDING_MAX_RETRIES):
            try:
                response = client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch,
                )

                ordered = sorted(
                    response.data,
                    key=lambda item: item.index,
                )

                embeddings.extend(
                    item.embedding
                    for item in ordered
                )

                last_error = None
                break

            except Exception as exc:
                last_error = exc

                message = str(exc).lower()

                if (
                    "429" not in message
                    and "rate" not in message
                    and "tpm" not in message
                ):
                    raise

                if attempt >= EMBEDDING_MAX_RETRIES - 1:
                    raise

                wait_seconds = (
                    EMBEDDING_RETRY_BASE_SECONDS
                    * (2 ** attempt)
                )

                time.sleep(wait_seconds)

        if last_error is not None:
            raise last_error

    return embeddings


def create_query_embedding(question):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=question,
    )

    return response.data[0].embedding


# -------------------------------------------------------------------
# Similarity
# -------------------------------------------------------------------


def cosine_similarity(vector_a, vector_b):
    dot_product = sum(
        a * b
        for a, b in zip(
            vector_a,
            vector_b,
        )
    )

    magnitude_a = math.sqrt(
        sum(
            value * value
            for value in vector_a
        )
    )

    magnitude_b = math.sqrt(
        sum(
            value * value
            for value in vector_b
        )
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return (
        dot_product
        / (magnitude_a * magnitude_b)
    )


def keyword_score(question, source):
    """
    Small lexical score used alongside semantic similarity.

    Embeddings do most of the retrieval work; lexical overlap helps with
    specific project terms, document names, acronyms and policy language.
    """

    words = re.findall(
        r"[a-zA-Z][a-zA-Z-]{2,}",
        question.lower(),
    )

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
        "your",
        "into",
        "have",
        "has",
        "how",
        "why",
        "who",
        "are",
        "was",
        "were",
        "tell",
        "give",
        "explain",
    }

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    if not words:
        return 0.0

    title_text = source["parent_title"].lower()
    body_text = source["text"].lower()

    # Exact overlap in the source title is more informative than a match
    # buried somewhere in a long chunk.
    title_matches = sum(
        title_text.count(word)
        for word in words
    )

    body_matches = sum(
        body_text.count(word)
        for word in words
    )

    weighted_matches = (
        title_matches * 2.5
        + body_matches
    )

    return min(
        weighted_matches / max(len(words), 1),
        10,
    ) / 10


# -------------------------------------------------------------------
# Retrieval
# -------------------------------------------------------------------


def retrieve_sources(
    question,
    sources,
    allowed_source_types,
):
    route = route_question(question)

    filtered_sources = [
        source
        for source in sources
        if source["source_type"]
        in allowed_source_types
    ]

    if not filtered_sources:
        return [], route

    texts = [
        make_embedding_text(source)
        for source in filtered_sources
    ]

    source_embeddings = (
        create_source_embeddings(texts)
    )

    question_embedding = (
        create_query_embedding(question)
    )

    scored = []

    for source, embedding in zip(
        filtered_sources,
        source_embeddings,
    ):
        semantic = cosine_similarity(
            question_embedding,
            embedding,
        )

        lexical = keyword_score(
            question,
            source,
        )

        priority = source_priority_bonus(
            route,
            source,
        )

        # Semantic relevance remains dominant. The source prior is deliberately
        # small: it resolves close calls rather than overriding poor relevance.
        combined_score = (
            semantic * 0.86
            + lexical * 0.14
            + priority
        )

        scored.append({
            **source,
            "semantic_score": semantic,
            "keyword_score": lexical,
            "priority_bonus": priority,
            "retrieval_score": combined_score,
        })

    scored.sort(
        key=lambda item: item[
            "retrieval_score"
        ],
        reverse=True,
    )

    selected = []
    document_counts = {}
    external_count = 0

    if route == "IPBES and transformative change route":
        external_limit = IPBES_ROUTE_MAX_EXTERNAL_CHUNKS
    elif route == "CORDIS project-information route":
        external_limit = TOP_CHUNKS
    else:
        external_limit = DEFAULT_MAX_EXTERNAL_CHUNKS

    for source in scored:
        document_key = (
            source["parent_title"],
            source["url"],
        )

        count = document_counts.get(
            document_key,
            0,
        )

        if count >= MAX_CHUNKS_PER_DOCUMENT:
            continue

        external = not is_bionext_source(source)

        if external and external_count >= external_limit:
            continue

        selected.append(source)

        document_counts[
            document_key
        ] = count + 1

        if external:
            external_count += 1

        if len(selected) >= TOP_CHUNKS:
            break

    return selected, route


# -------------------------------------------------------------------
# Group retrieved chunks into citation sources
# -------------------------------------------------------------------


def group_retrieved_sources(retrieved):
    grouped = {}

    for source in retrieved:
        key = (
            source["parent_title"],
            source["url"],
        )

        if key not in grouped:
            grouped[key] = {
                "title": source["parent_title"],
                "url": source["url"],
                "source_type": source[
                    "source_type"
                ],
                "chunks": [],
                "best_score": source[
                    "retrieval_score"
                ],
            }

        grouped[key]["chunks"].append(
            source
        )

        grouped[key]["best_score"] = max(
            grouped[key]["best_score"],
            source["retrieval_score"],
        )

    groups = list(
        grouped.values()
    )

    groups.sort(
        key=lambda item: item[
            "best_score"
        ],
        reverse=True,
    )

    return groups


# -------------------------------------------------------------------
# Prompt context
# -------------------------------------------------------------------


def make_context(groups):
    """
    Build a compact evidence context for the answer model.

    Retrieved chunks can be very large, particularly for long reports.
    This function enforces hard limits on individual passages and on the
    complete prompt context.
    """

    blocks = []
    total_chars = 0

    for source_number, group in enumerate(
        groups[:MAX_SOURCES_IN_PROMPT],
        start=1,
    ):
        passages = []

        for chunk in group["chunks"]:
            text = chunk["text"]

            if len(text) > MAX_CHARS_PER_CHUNK:
                text = (
                    text[:MAX_CHARS_PER_CHUNK]
                    + "\n\n[Text truncated]"
                )

            chunk_label = (
                f"Chunk {chunk['chunk_number']} "
                f"of {chunk['chunk_count']}"
            )

            passage = (
                chunk_label
                + ":\n"
                + text
            )

            projected_total = (
                total_chars
                + sum(len(item) for item in passages)
                + len(passage)
                + len(group["title"])
                + len(group["url"])
                + len(group["source_type"])
                + 100
            )

            if projected_total > MAX_CONTEXT_CHARS:
                break

            passages.append(passage)

        if not passages:
            continue

        block = (
            f"SOURCE [{source_number}]\n"
            f"Title: {group['title']}\n"
            f"URL: {group['url']}\n"
            f"Source type: {group['source_type']}\n\n"
            + "\n\n".join(passages)
        )

        if total_chars + len(block) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - total_chars

            if remaining < 300:
                break

            block = block[:remaining]

        blocks.append(block)
        total_chars += len(block)

        if total_chars >= MAX_CONTEXT_CHARS:
            break

    return "\n\n---\n\n".join(blocks)


# -------------------------------------------------------------------
# Evidence answerability check
# -------------------------------------------------------------------


def evidence_supports_question(question, groups):
    """
    Check whether the retrieved evidence actually contains enough information
    to answer the user's question without relying on outside knowledge.

    This is deliberately separate from retrieval scoring. Embedding similarity
    can sometimes find superficially related chunks for an unrelated question.
    The evidence gate prevents the answer model from falling back to its own
    general knowledge in those cases.
    """

    if not groups:
        return False

    context = make_context(groups)

    if not context.strip():
        return False

    check_prompt = f"""
You are an evidence-gating system for Ask BIONEXT.

Decide whether the supplied evidence contains enough explicit information to
answer the user's question without using any outside or general knowledge.

Rules:
- Judge only the evidence supplied below.
- Do not answer the question.
- Do not use facts you know from elsewhere.
- If the evidence is only loosely related, tangential, or does not explicitly
  support the requested fact or explanation, return UNSUPPORTED.
- Return SUPPORTED only when a careful answer can be grounded in the evidence.
- Reply with exactly one word: SUPPORTED or UNSUPPORTED.

User question:
{question}

Evidence:
{context}
"""

    response = client.chat.completions.create(
        model=ANSWERABILITY_MODEL,
        messages=[
            {
                "role": "user",
                "content": check_prompt,
            }
        ],
        temperature=0,
        max_tokens=5,
    )

    verdict = (
        response
        .choices[0]
        .message
        .content
        .strip()
        .upper()
    )

    return verdict.startswith("SUPPORTED")


# -------------------------------------------------------------------
# Answer generation
# -------------------------------------------------------------------


def generate_answer(
    question,
    allowed_source_types,
):
    sources = load_sources()

    retrieved, route = retrieve_sources(
        question,
        sources,
        allowed_source_types,
    )

    if not retrieved:
        return (
            "I could not find enough information "
            "in the approved Ask BIONEXT sources "
            "to answer that question.",
            route,
            [],
            [],
        )

    if (
        retrieved[0]["retrieval_score"]
        < MIN_TOP_RETRIEVAL_SCORE
    ):
        return (
            "I could not find sufficiently relevant evidence "
            "in the approved Ask BIONEXT sources to answer "
            "that question reliably.",
            route,
            [],
            retrieved,
        )

    groups = group_retrieved_sources(
        retrieved
    )

    if not evidence_supports_question(
        question,
        groups,
    ):
        return (
            "I could not find sufficient evidence within the approved "
            "BIONEXT knowledge base to answer that question.",
            route,
            [],
            retrieved,
        )

    context = make_context(
        groups
    )

    prompt = f"""
You are Ask BIONEXT 2.0, a source-bounded research assistant for the
EU-funded BIONEXT project.

Your task is to answer the user's question using ONLY the approved source
material supplied below.

Evidence policy:

1. Never use outside knowledge, even if you believe you know the answer.
2. Base every factual claim on the supplied source material.
3. Treat BIONEXT project sources as the primary evidence base.
4. When BIONEXT and external sources cover the same issue, lead with the
   BIONEXT evidence and use external evidence to support, contextualise or
   compare it where useful.
5. IPBES material is authoritative external evidence. Do not present an
   IPBES finding as if it were a BIONEXT finding.
6. If the user explicitly asks what IPBES says, answer primarily from the
   supplied IPBES material, while connecting it to BIONEXT only where the
   supplied BIONEXT sources support that connection.
7. CORDIS should primarily be used for formal project information such as
   objectives, funding and grant details.
8. If the supplied evidence is incomplete, ambiguous or insufficient, do not
   answer from memory or general knowledge. Respond only with:
   "I could not find sufficient evidence within the approved BIONEXT knowledge base to answer that question."
9. If sources disagree, explain the disagreement or difference in framing.

Answer style:

10. Answer the question directly before adding detail.
11. Prefer a concise synthesis over a source-by-source summary.
12. Use plain, professional language suitable for researchers, practitioners
    and policy audiences.
13. Use short headings or bullets only when they genuinely improve clarity.
14. Avoid unnecessary repetition and generic introductory language.
15. Where useful, distinguish clearly between BIONEXT findings, guidance,
    project outputs and external IPBES evidence.

Citations:

16. Cite supporting sources using their square-bracket source numbers,
    for example [1] or [2].
17. Put citations immediately after the statement they support.
18. A statement supported by several sources may cite multiple numbers,
    for example [1][3].
19. Do not invent citation numbers.
20. Do not create a bibliography or a separate "Sources used" section in
    the answer; the application displays the source list automatically.

User question:

{question}

Approved source material:

{context}
"""

    response = client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.15,
        max_tokens=1400,
    )

    answer = (
        response
        .choices[0]
        .message
        .content
    )

    return (
        answer,
        route,
        groups,
        retrieved,
    )


# -------------------------------------------------------------------
# Interface
# -------------------------------------------------------------------


st.markdown(
    """
    <div class="header">
      <h1>Ask BIONEXT 2.0</h1>
      <p>
        Source-bounded AI research assistant for BIONEXT
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="prototype-note">
      <strong>Prototype:</strong> Ask BIONEXT searches an approved evidence
      base of BIONEXT project content and selected external sources. BIONEXT
      evidence is prioritised in general searches, while highly relevant
      external evidence remains available where appropriate.
    </div>
    """,
    unsafe_allow_html=True,
)


sources = load_sources()

source_types = sorted(
    {
        source["source_type"]
        for source in sources
    }
)


with st.sidebar:
    st.title(
        "Ask BIONEXT 2.0"
    )

    st.write(
        "Ask questions across the approved "
        "BIONEXT knowledge base, including "
        "project web pages, resources, linked "
        "PDFs and selected IPBES evidence."
    )

    st.markdown(
        "### Source database"
    )

    if sources:
        document_keys = {
            (
                source["parent_title"],
                source["url"],
            )
            for source in sources
        }

        st.success(
            f"{len(document_keys)} documents / "
            f"{len(sources)} retrieval chunks loaded"
        )

    else:
        st.error(
            "sources.json could not be loaded."
        )

    st.markdown("---")

    st.markdown(
        "### Source types"
    )

    selected_source_types = (
        st.multiselect(
            "Search within",
            options=source_types,
            default=source_types,
        )
    )

    st.caption(
        "BIONEXT sources receive a modest priority "
        "in retrieval. Uncheck a source type to "
        "exclude it from the search."
    )


st.write(
    "Ask a question about BIONEXT. "
    "The app searches the approved knowledge base "
    "and synthesises the most relevant evidence."
)


question = st.text_input(
    "Question",
    placeholder=(
        "e.g. What does BIONEXT say about "
        "justice in biodiversity policy?"
    ),
)


if (
    st.button("Ask BIONEXT")
    and question.strip()
):
    if not get_api_key():
        st.error(
            "No OpenAI API key found. "
            "Add OPENAI_API_KEY in Streamlit Secrets."
        )

    elif not selected_source_types:
        st.warning(
            "Please select at least one source type."
        )

    else:
        with st.spinner(
            "Searching BIONEXT evidence..."
        ):
            try:
                (
                    answer,
                    route,
                    groups,
                    retrieved,
                ) = generate_answer(
                    question,
                    selected_source_types,
                )

                st.markdown(
                    '<div class="answer">'
                    + answer
                    + "</div>",
                    unsafe_allow_html=True,
                )

                if groups:
                    st.markdown(
                        "### Sources used"
                    )

                    for number, group in enumerate(
                        groups,
                        start=1,
                    ):
                        st.markdown(
                            f"**[{number}] [{group['title']}]"
                            f"({group['url']})**"
                        )

                        st.caption(
                            group["source_type"]
                        )

                with st.expander(
                    "Technical retrieval details"
                ):
                    st.caption(
                        f"Search route: {route}"
                    )

                    for item in retrieved:
                        st.write(
                            f"**{item['title']}**"
                        )

                        st.caption(
                            f"{item['source_type']} · "
                            f"semantic {item['semantic_score']:.3f} · "
                            f"keyword {item['keyword_score']:.3f} · "
                            f"priority +{item['priority_bonus']:.3f} · "
                            f"combined {item['retrieval_score']:.3f}"
                        )

            except Exception as exc:
                st.error(
                    f"Something went wrong: {exc}"
                )
