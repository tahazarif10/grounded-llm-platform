from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=200_000)
    metadata: dict[str, str] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1, max_length=260)
    source_id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=20_000)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)


class RetrievalHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk: DocumentChunk
    score: float = Field(ge=0.0)


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4_000)
    top_k: int = Field(default=5, ge=1, le=12)


class IndexRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    documents: list[Document] = Field(min_length=1, max_length=64)
    max_words_per_chunk: int = Field(default=180, ge=40, le=800)
    overlap_lines: int = Field(default=2, ge=0, le=20)


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4_000)
    evidence: list[DocumentChunk] = Field(min_length=1, max_length=12)


class ModelDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(max_length=8_000)
    cited_chunk_ids: list[str] = Field(default_factory=list, max_length=12)
    confidence: float = Field(ge=0.0, le=1.0)
    abstained: bool


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    source_id: str
    start_line: int
    end_line: int


class GroundedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    citations: list[Citation]
    confidence: float = Field(ge=0.0, le=1.0)
    abstained: bool
    retrieval_count: int = Field(ge=0)
