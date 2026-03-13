import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from loguru import logger

from src.generation.llm_client import LLMClient
from src.retrieval.retriever import Retriever
from src.utils.config import settings

app = typer.Typer(help="DocuRAG Query CLI — powered by Endee vector database")


@app.command()
def main(
    question: str = typer.Argument(..., help="Your natural language question"),
    top_k: int = typer.Option(settings.top_k, "--top-k", "-k", help="Number of chunks to retrieve"),
    show_context: bool = typer.Option(False, "--show-context", help="Print retrieved chunks"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Only show retrieved chunks, skip LLM generation"),
):
    """Ask a question against your ingested documents."""

    retriever = Retriever()

    typer.echo(f"\n Retrieving top-{top_k} chunks from Endee...")
    t0 = time.time()
    results = retriever.retrieve(question, top_k=top_k)
    retrieval_ms = int((time.time() - t0) * 1000)

    if not results:
        typer.echo("No relevant chunks found. Have you ingested documents yet?")
        typer.echo("    Run: python scripts/ingest.py --dir data/sample_docs/")
        raise typer.Exit(1)

    typer.echo(f"   Retrieved {len(results)} chunks in {retrieval_ms}ms\n")

    if show_context or no_llm:
        typer.echo("─" * 60)
        typer.echo("Retrieved Context Chunks:")
        typer.echo("─" * 60)
        for i, r in enumerate(results, 1):
            source = r.meta.get("source", "unknown")
            page = r.meta.get("page_number")
            page_str = f" p.{page}" if page else ""
            text = r.meta.get("text", "")[:300]
            typer.echo(f"\n[{i}] {source}{page_str}  (similarity: {r.similarity:.4f})")
            typer.echo(f"    {text}{'...' if len(r.meta.get('text','')) > 300 else ''}")
        typer.echo("─" * 60)

    if not no_llm:
        llm = LLMClient()
        typer.echo("Generating answer...")
        t1 = time.time()
        answer = llm.generate_answer(question, results)
        gen_ms = int((time.time() - t1) * 1000)

        typer.echo(f"\n{'═' * 60}")
        typer.echo(f"{question}")
        typer.echo(f"{'═' * 60}")
        typer.echo(f"\n{answer}")
        typer.echo(f"\nRetrieval: {retrieval_ms}ms  |  Generation: {gen_ms}ms")

        typer.echo(f"\nSources:")
        seen = set()
        for r in results:
            src = r.meta.get("source", "unknown")
            if src not in seen:
                seen.add(src)
                typer.echo(f"   • {src}  (best match: {r.similarity:.4f})")


if __name__ == "__main__":
    app()