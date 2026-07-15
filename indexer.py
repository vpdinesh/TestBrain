#!/usr/bin/env python3
"""
TestBrain — Document Indexer
==============================
Indexes test reports, documentation, and results into ChromaDB
for RAG-powered natural language querying.

Usage:
    python3 indexer.py --path /path/to/your/test/repo
    python3 indexer.py  (uses TEST_REPO_PATH env variable or current directory)
"""

import os
import sys
import json
import glob
import argparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# ── Configuration ─────────────────────────────────────────────────────────────

CHROMA_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")

# File patterns to index (customize for your project)
DEFAULT_PATTERNS = [
    "**/*.md",
    "**/reports/*.json",
    "**/test-reports/*.json",
    "**/docs/*.md",
]

# Skip patterns
SKIP_PATTERNS = ["node_modules", ".git", "__pycache__", "package-lock", "chroma_db", "rag_env"]

# Max file size to index (500KB)
MAX_FILE_SIZE = 500_000


def load_documents(repo_path, patterns=None):
    """Load all documents from the specified repository path."""
    documents = []
    patterns = patterns or DEFAULT_PATTERNS

    for pattern in patterns:
        full_pattern = os.path.join(repo_path, pattern)
        files = glob.glob(full_pattern, recursive=True)

        for filepath in files:
            # Skip unwanted files
            if any(skip in filepath for skip in SKIP_PATTERNS):
                continue
            if os.path.getsize(filepath) > MAX_FILE_SIZE:
                continue

            try:
                with open(filepath, "r", errors="ignore") as f:
                    content = f.read()

                if not content.strip():
                    continue

                # For JSON files, extract key info
                if filepath.endswith(".json"):
                    try:
                        data = json.loads(content)
                        if isinstance(data, dict) and "summary" in data:
                            content = f"Test Report: {os.path.basename(filepath)}\n"
                            content += f"Summary: {json.dumps(data['summary'], indent=2)}\n"
                            if "results" in data:
                                for r in data["results"][:20]:
                                    content += f"- {r.get('id', '')}: {r.get('name', '')} — {r.get('status', '')} — {r.get('notes', '')}\n"
                        elif isinstance(data, list):
                            content = f"Data from: {os.path.basename(filepath)}\n{json.dumps(data[:10], indent=2)}"
                    except json.JSONDecodeError:
                        pass

                # Create document with metadata
                rel_path = os.path.relpath(filepath, repo_path)
                documents.append({
                    "content": content[:10000],
                    "metadata": {
                        "source": rel_path,
                        "type": "markdown" if filepath.endswith(".md") else "json" if filepath.endswith(".json") else "text",
                    }
                })
            except Exception as e:
                print(f"  Skip: {filepath} ({e})")

    return documents


def index_documents(repo_path):
    """Index all documents into ChromaDB."""
    print("=" * 60)
    print("  TestBrain — Document Indexer")
    print("=" * 60)
    print(f"  Source: {repo_path}")
    print(f"  DB:     {CHROMA_DB_PATH}")
    print()

    # Load documents
    print("  Loading documents...")
    docs = load_documents(repo_path)
    print(f"  Found {len(docs)} documents")

    if not docs:
        print("  No documents found! Check your --path argument.")
        print(f"  Looking in: {repo_path}")
        sys.exit(1)

    # Split into chunks
    print("  Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "]
    )

    chunks = []
    metadatas = []
    for doc in docs:
        splits = splitter.split_text(doc["content"])
        for split in splits:
            chunks.append(split)
            metadatas.append(doc["metadata"])

    print(f"  Created {len(chunks)} chunks")

    # Create embeddings and store in ChromaDB
    print("  Creating embeddings (first run downloads model ~90MB)...")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Clear existing DB
    if os.path.exists(CHROMA_DB_PATH):
        import shutil
        shutil.rmtree(CHROMA_DB_PATH)

    # Create vector store
    Chroma.from_texts(
        texts=chunks,
        metadatas=metadatas,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
    )

    print(f"\n  Indexed {len(chunks)} chunks into ChromaDB")
    print(f"  Database: {CHROMA_DB_PATH}")
    print("  Done! Now run: streamlit run app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TestBrain — Index your test documents")
    parser.add_argument("--path", type=str,
                        default=os.environ.get("TEST_REPO_PATH", os.getcwd()),
                        help="Path to your test automation repository")
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"  Error: Path does not exist: {args.path}")
        sys.exit(1)

    index_documents(args.path)
