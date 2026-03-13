import sys
import time
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from loguru import logger

from src.retrieval.retriever import IngestionService
from src.utils.config import settings

app = typer.Typer(help="DocuRAG Ingestion CLI — powered by Endee vector database")


@app.command()
def main(
    file: str = typer.Option(None, "--file", "-f", help="Path to a single document (PDF/TXT/MD)"),
    directory: str = typer.Option(None, "--dir", "-d", help="Path to a directory of documents"),
    chunk_size: int = typer.Option(settings.chunk_size, "--chunk-size", help="Max characters per chunk"),
    overlap: int = typer.Option(settings.chunk_overlap, "--overlap", help="Character overlap between chunks"),
    reset: bool = typer.Option(False, "--reset", help="Clear the Endee index before ingesting"),
):
    """Ingest documents into the Endee vector index."""

    if not file and not directory:
        typer.echo("Please specify --file or --dir", err=True)
        raise typer.Exit(code=1)

    # Override config if user passed custom values
    settings.chunk_size = chunk_size
    settings.chunk_overlap = overlap

    if reset:
        from src.retrieval.vector_store import VectorStore
        typer.echo("Resetting Endee index...")
        VectorStore().reset()
        typer.echo("Index cleared.")

    svc = IngestionService()
    t0 = time.time()

    if file:
        typer.echo(f"Ingesting file: {file}")
        total = svc.ingest_file(file)
    else:
        typer.echo(f"Ingesting directory: {directory}")
        total = svc.ingest_directory(directory)

    elapsed = round(time.time() - t0, 2)
    typer.echo(f"\nDone! {total} chunks indexed in {elapsed}s")
    typer.echo(f"   Endee index: {settings.endee_index_name}")
    typer.echo(f"   Documents:   {', '.join(svc.list_documents())}")


if __name__ == "__main__":
    app()