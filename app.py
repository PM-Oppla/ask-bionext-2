import json
import math
import os
import re

import streamlit as st
from openai import OpenAI


# -------------------------------------------------------------------
# Ask BIONEXT 2.0
# App v3
#
# Features:
# - source-bounded answers
# - chunk-based retrieval
# - OpenAI embedding search
# - hybrid semantic + keyword ranking
# - source-type filtering
# - grouped citations
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

    .source-card {
      background: #f7f9f9;
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

TOP_CHUNKS = 8

MAX_CHUNKS_PER_DOCUMENT = 2


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

    if any(
        term in q
        for term in [
            "decision",
            "mcda",
            "multi-criteria",
            "criteria",
            "trade-off",
            "tradeoff",
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
        ]
    ):
        return "BIONEXT resources route"

    if any(
        term in q
        for term in [
            "ipbes",
            "transformative change",
            "assessment",
        ]
    ):
        return "IPBES and transformative change route"

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
# Embeddings
# -------------------------------------------------------------------


@st.cache_data(
    show_spinner=False,
    ttl=3600,
)
def create_source_embeddings(texts):
    """
    Create embeddings for all retrieval chunks.

    Streamlit caches the result so the source database does not need
    to be embedded again on every question.
    """

    if not texts:
        return []

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    ordered = sorted(
        response.data,
        key=lambda item: item.index,
    )

    return [
        item.embedding
        for item in ordered
    ]


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

    Embeddings do most of the retrieval work.
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
    }

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    if not words:
        return 0.0

    haystack = (
        source["parent_title"]
        + " "
        + source["text"]
    ).lower()

    matches = sum(
        haystack.count(word)
        for word in words
    )

    return min(
        matches / max(len(words), 1),
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
    filtered_sources = [
        source
        for source in sources
        if source["source_type"]
        in allowed_source_types
    ]

    if not filtered_sources:
        return []

    texts = [
        source["text"]
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

        # Semantic meaning drives ranking.
        # Keyword overlap acts as a small tie-breaker.
        combined_score = (
            semantic * 0.90
            + lexical * 0.10
        )

        scored.append({
            **source,
            "semantic_score": semantic,
            "keyword_score": lexical,
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

        selected.append(source)

        document_counts[
            document_key
        ] = count + 1

        if len(selected) >= TOP_CHUNKS:
            break

    return selected


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
    blocks = []

    for source_number, group in enumerate(
        groups,
        start=1,
    ):
        passages = []

        for chunk in group["chunks"]:
            chunk_label = (
                f"Chunk {chunk['chunk_number']} "
                f"of {chunk['chunk_count']}"
            )

            passages.append(
                chunk_label
                + ":\n"
                + chunk["text"]
            )

        block = (
            f"SOURCE [{source_number}]\n"
            f"Title: {group['title']}\n"
            f"URL: {group['url']}\n"
            f"Source type: "
            f"{group['source_type']}\n\n"
            + "\n\n".join(passages)
        )

        blocks.append(block)

    return "\n\n---\n\n".join(
        blocks
    )


# -------------------------------------------------------------------
# Answer generation
# -------------------------------------------------------------------


def generate_answer(
    question,
    allowed_source_types,
):
    sources = load_sources()

    route = route_question(
        question
    )

    retrieved = retrieve_sources(
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

    groups = group_retrieved_sources(
        retrieved
    )

    context = make_context(
        groups
    )

    prompt = f"""
You are Ask BIONEXT 2.0, a source-bounded research assistant.

Your task is to answer questions about BIONEXT using ONLY the
approved source material supplied below.

Rules:

1. Do not use outside or general knowledge.
2. Base factual claims only on the supplied sources.
3. If the sources do not contain enough evidence to answer the
   question, say so clearly.
4. Synthesize information across sources where useful.
5. Prefer specific evidence over vague summaries.
6. Do not invent project results, publications, findings or dates.
7. Cite supporting sources using square-bracket source numbers,
   for example [1] or [2].
8. Put citations immediately after the statement they support.
9. Do not create a separate bibliography or "Sources used" section;
   the application will display the source list automatically.
10. If sources disagree, explain the disagreement rather than
    choosing one without justification.

User question:

{question}

Approved source material:

{context}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.2,
        max_tokens=1200,
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
        "web pages, project resources and "
        "linked PDFs."
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
        "Uncheck a source type to exclude it "
        "from retrieval."
    )


st.write(
    "Ask a question about BIONEXT. "
    "The app will search the approved knowledge base "
    "and retrieve the most relevant evidence."
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
                    f"**Agent route:** {route}"
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
                            f"**[{number}] "
                            f"{group['title']}**"
                        )

                        st.write(
                            group["url"]
                        )

                        st.caption(
                            group["source_type"]
                        )

                with st.expander(
                    "Retrieval details"
                ):
                    for item in retrieved:
                        st.write(
                            f"**{item['title']}**"
                        )

                        st.caption(
                            f"{item['source_type']} · "
                            f"semantic "
                            f"{item['semantic_score']:.3f} · "
                            f"combined "
                            f"{item['retrieval_score']:.3f}"
                        )

            except Exception as exc:
                st.error(
                    f"Something went wrong: {exc}"
                )