from grounded_llm.models import Document
from grounded_llm.retrieval import BM25Retriever, chunk_documents


def test_chunking_preserves_source_and_line_provenance():
    document = Document(
        source_id="manual",
        text="line one\nline two emergency stop\nline three\nline four",
    )
    chunks = chunk_documents([document], max_words_per_chunk=4, overlap_lines=1)

    assert len(chunks) >= 2
    assert all(chunk.source_id == "manual" for chunk in chunks)
    assert chunks[0].start_line == 1
    assert chunks[0].end_line >= chunks[0].start_line


def test_bm25_ranks_relevant_chunk_first():
    chunks = chunk_documents(
        [
            Document(
                source_id="motion",
                text="Servo motion requires drive enable and motion permission.",
            ),
            Document(source_id="network", text="Ethernet diagnostics expose packet counters."),
        ]
    )
    retriever = BM25Retriever()
    retriever.index(chunks)

    hits = retriever.search("servo motion permission", top_k=2)

    assert hits
    assert hits[0].chunk.source_id == "motion"
    assert hits[0].score > 0
