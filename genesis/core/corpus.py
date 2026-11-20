"""Import, conversion et lecture en flux du corpus.

Rien n'est jamais chargé entièrement en mémoire :
    - les fichiers sont copiés puis lus par blocs (1 Mio) ;
    - la tokenisation écrit directement un fichier binaire de tokens (tokens.bin)
      qui sera ensuite lu en mémoire virtuelle (numpy.memmap) pendant
      l'entraînement. La RAM utilisée reste donc constante, même avec 10 Go de
      corpus.

Formats acceptés nativement : texte brut, Markdown, CSV/TSV, JSON/JSONL, HTML,
XML, EPUB. Extensions optionnelles si les paquets sont installés : .docx
(python-docx), .pdf (pypdf).
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import unicodedata
import zipfile
from pathlib import Path
from typing import Callable, Iterable, Iterator

from .tokenizers import SPECIAL, Tokenizer

CHUNK_CHARS = 1 << 20  # 1 Mio de texte par bloc

TEXT_EXTS = {".txt", ".text", ".md", ".markdown", ".rst", ".log", ".tex", ".asc"}
STRUCTURED_EXTS = {".csv", ".tsv", ".json", ".jsonl", ".ndjson"}
MARKUP_EXTS = {".html", ".htm", ".xhtml", ".xml", ".sgml", ".svg"}
ODD_EXTS = {".epub", ".docx", ".pdf"}

SUPPORTED_EXTS = TEXT_EXTS | STRUCTURED_EXTS | MARKUP_EXTS | ODD_EXTS

FORMAT_LABELS = {
    ".txt": "Texte",
    ".text": "Texte",
    ".md": "Markdown",
    ".markdown": "Markdown",
    ".rst": "reStructuredText",
    ".log": "Journal",
    ".tex": "LaTeX",
    ".asc": "Texte",
    ".csv": "CSV",
    ".tsv": "TSV",
    ".json": "JSON",
    ".jsonl": "JSONL",
    ".ndjson": "JSONL",
    ".html": "HTML",
    ".htm": "HTML",
    ".xhtml": "HTML",
    ".xml": "XML",
    ".sgml": "SGML",
    ".svg": "SVG",
    ".epub": "EPUB",
    ".docx": "Word",
    ".pdf": "PDF",
}

_TAG_RE = re.compile(r"<script.*?</script>|<style.*?</style>|<[^>]+>", re.S | re.I)
_WS_RE = re.compile(r"[ \t\r\f\v]+")


# ---------------------------------------------------------------------------
# Détection d'encodage & lecture texte en flux
# ---------------------------------------------------------------------------
def detect_encoding(path: Path) -> str:
    with open(path, "rb") as f:
        head = f.read(65536)
    if head.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if head.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    try:
        head.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        pass
    try:
        head.decode("cp1252")
        return "cp1252"
    except UnicodeDecodeError:
        return "latin-1"


def _iter_text_file(path: Path, block: int = CHUNK_CHARS) -> Iterator[str]:
    encoding = detect_encoding(path)
    with open(path, "r", encoding=encoding, errors="replace", newline="") as f:
        while True:
            chunk = f.read(block)
            if not chunk:
                break
            yield unicodedata.normalize("NFC", chunk)


def _iter_rows(path: Path, delimiter: str | None = None) -> Iterator[str]:
    encoding = detect_encoding(path)
    with open(path, "r", encoding=encoding, errors="replace", newline="") as f:
        if delimiter is None:
            sample = f.read(8192)
            f.seek(0)
            try:
                delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
            except csv.Error:
                delimiter = ","
        reader = csv.reader(f, delimiter=delimiter)
        buf: list[str] = []
        size = 0
        for row in reader:
            line = " ".join(cell.strip() for cell in row if cell and cell.strip())
            if not line:
                continue
            buf.append(line)
            size += len(line) + 1
            if size >= CHUNK_CHARS:
                yield "\n".join(buf) + "\n"
                buf, size = [], 0
        if buf:
            yield "\n".join(buf) + "\n"


def _extract_strings(obj: object) -> Iterator[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _extract_strings(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _extract_strings(value)


def _iter_jsonl(path: Path) -> Iterator[str]:
    for chunk in _iter_text_file(path):
        for line in chunk.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                yield "\n".join(_extract_strings(json.loads(line))) + "\n"
            except json.JSONDecodeError:
                yield line + "\n"


def _iter_json(path: Path) -> Iterator[str]:
    if path.stat().st_size > 512 * 1024 * 1024:
        raise ValueError("JSON de plus de 512 Mo : convertissez-le en JSONL pour un flux mémoire constant.")
    encoding = detect_encoding(path)
    data = json.loads(path.read_text(encoding=encoding, errors="replace"))
    buf: list[str] = []
    size = 0
    for text in _extract_strings(data):
        buf.append(text)
        size += len(text) + 1
        if size >= CHUNK_CHARS:
            yield "\n".join(buf) + "\n"
            buf, size = [], 0
    if buf:
        yield "\n".join(buf) + "\n"


def _strip_markup(text: str) -> str:
    import html as html_mod

    text = _TAG_RE.sub("\n", text)
    text = html_mod.unescape(text)
    return _WS_RE.sub(" ", text)


def _iter_markup(path: Path) -> Iterator[str]:
    buf = ""
    for chunk in _iter_text_file(path):
        buf += chunk
        if len(buf) >= CHUNK_CHARS:
            cut = buf.rfind(">")
            if cut < 0:
                cut = len(buf) - 1
            piece, buf = buf[: cut + 1], buf[cut + 1 :]
            cleaned = _strip_markup(piece)
            if cleaned.strip():
                yield cleaned + "\n"
    if buf:
        cleaned = _strip_markup(buf)
        if cleaned.strip():
            yield cleaned + "\n"


def _iter_docx(path: Path) -> Iterator[str]:
    try:
        import docx  # python-docx
    except ImportError as exc:  # pragma: no cover
        raise ValueError("Format .docx : installez python-docx (pip install python-docx) "
                         "ou enregistrez le fichier en .txt.") from exc
    document = docx.Document(str(path))
    buf: list[str] = []
    size = 0
    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        buf.append(text)
        size += len(text) + 1
        if size >= CHUNK_CHARS:
            yield "\n".join(buf) + "\n"
            buf, size = [], 0
    for table in document.tables:
        for row in table.rows:
            line = " | ".join(cell.text.strip() for cell in row.cells)
            if line.strip((" |")):
                yield line + "\n"
    if buf:
        yield "\n".join(buf) + "\n"


def _iter_pdf(path: Path) -> Iterator[str]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise ValueError("Format .pdf : installez pypdf (pip install pypdf) "
                         "ou convertissez le fichier en .txt.") from exc
    reader = PdfReader(str(path))
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:  # page illisible : on continue
            continue
        if text.strip():
            yield text + "\n"


def _iter_epub(path: Path) -> Iterator[str]:
    with zipfile.ZipFile(path) as zf:
        for name in sorted(zf.namelist()):
            if not name.lower().endswith((".xhtml", ".html", ".htm", ".xml")):
                continue
            with zf.open(name) as fh:
                raw = fh.read()
            text = raw.decode("utf-8", errors="replace")
            cleaned = _strip_markup(text)
            if cleaned.strip():
                yield cleaned + "\n"


def iter_text_chunks(path: Path) -> Iterator[str]:
    """Renvoie le texte d'un fichier par blocs, quel que soit son format."""
    ext = path.suffix.lower()
    if ext in TEXT_EXTS or ext == "":
        yield from _iter_text_file(path)
    elif ext == ".csv":
        yield from _iter_rows(path, ",")
    elif ext == ".tsv":
        yield from _iter_rows(path, "\t")
    elif ext in (".jsonl", ".ndjson"):
        yield from _iter_jsonl(path)
    elif ext == ".json":
        yield from _iter_json(path)
    elif ext in MARKUP_EXTS:
        yield from _iter_markup(path)
    elif ext == ".docx":
        yield from _iter_docx(path)
    elif ext == ".pdf":
        yield from _iter_pdf(path)
    elif ext == ".epub":
        yield from _iter_epub(path)
    else:
        # tentative en texte brut pour les extensions inconnues
        yield from _iter_text_file(path)


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTS or path.is_dir()


def format_label(path: Path | str) -> str:
    p = Path(path)
    return FORMAT_LABELS.get(p.suffix.lower(), p.suffix.lstrip(".").upper() or "Fichier")


# ---------------------------------------------------------------------------
# Import dans un projet
# ---------------------------------------------------------------------------
def _unique_name(folder: Path, name: str) -> str:
    candidate = name
    stem, suffix = Path(name).stem, Path(name).suffix
    i = 1
    while (folder / candidate).exists():
        candidate = f"{stem}_{i}{suffix}"
        i += 1
    return candidate


def import_file(corpus_dir: Path, src: Path, progress: Callable[[int], None] | None = None) -> dict:
    """Copie un fichier dans le corpus du projet et calcule ses statistiques."""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = re.sub(r"[^0-9A-Za-zÀ-ÿ._ -]", "_", src.stem)[:80] or "fichier"
    stored = _unique_name(corpus_dir, f"{safe_stem}{src.suffix.lower()}")
    dest = corpus_dir / stored
    shutil.copy2(src, dest)

    chars = 0
    lines = 0
    try:
        for chunk in iter_text_chunks(dest):
            chars += len(chunk)
            lines += chunk.count("\n")
    except ValueError:
        raise
    except Exception:
        pass
    return {
        "name": src.name,
        "stored": stored,
        "ext": src.suffix.lower(),
        "format": format_label(src),
        "size": dest.stat().st_size,
        "chars": chars,
        "lines": lines,
    }


def import_paths(corpus_dir: Path, paths: Iterable[Path],
                 progress: Callable[[int, int, str], None] | None = None) -> tuple[list[dict], list[str]]:
    """Importe fichiers et dossiers (récursivement). Renvoie (importés, erreurs)."""
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob("*")
                                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS))
        elif path.is_file():
            files.append(path)
    imported: list[dict] = []
    errors: list[str] = []
    for i, path in enumerate(files):
        if progress:
            progress(i, len(files), path.name)
        try:
            imported.append(import_file(corpus_dir, path))
        except Exception as exc:  # un fichier cassé ne bloque pas l'import
            errors.append(f"{path.name} : {exc}")
    return imported, errors


# ---------------------------------------------------------------------------
# Lecture d'un corpus de projet
# ---------------------------------------------------------------------------
def iter_corpus_text(files: Iterable[dict], corpus_dir: Path,
                     add_separators: bool = True) -> Iterator[str]:
    for info in files:
        path = corpus_dir / info["stored"]
        if not path.exists():
            continue
        try:
            for chunk in iter_text_chunks(path):
                yield chunk
        except Exception as exc:
            yield f"\n[genesis] fichier ignoré ({path.name}) : {exc}\n"
        if add_separators:
            yield f"\n{SPECIAL}\n"


def corpus_stats(files: Iterable[dict]) -> dict:
    files = list(files)
    return {
        "files": len(files),
        "chars": sum(int(f.get("chars", 0)) for f in files),
        "lines": sum(int(f.get("lines", 0)) for f in files),
        "bytes": sum(int(f.get("size", 0)) for f in files),
    }


def build_token_file(
    files: list[dict],
    corpus_dir: Path,
    tokenizer: Tokenizer,
    out_path: Path,
    total_chars: int = 0,
    progress: Callable[[float, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict:
    """Tokenise tout le corpus vers un fichier binaire de tokens (flux constant)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".bin.part")
    import array

    n_tokens = 0
    chars_done = 0
    source = iter_corpus_text(files, corpus_dir, add_separators=True)
    with open(tmp_path, "wb") as fh:
        for chunk in source:
            if should_stop and should_stop():
                break
            ids = tokenizer.encode(chunk)
            if ids:
                arr = array.array(tokenizer.array_typecode, ids)
                fh.write(arr.tobytes())
                n_tokens += len(ids)
            chars_done += len(chunk)
            if progress and total_chars:
                progress(min(0.99, chars_done / total_chars), f"{n_tokens:,} tokens")
    os.replace(tmp_path, out_path)
    return {
        "tokens": n_tokens,
        "chars": chars_done,
        "dtype": tokenizer.dtype,
        "vocab_size": tokenizer.vocab_size,
        "path": str(out_path),
    }


def preview_corpus(files: Iterable[dict], corpus_dir: Path, max_chars: int = 1200) -> str:
    out: list[str] = []
    total = 0
    for chunk in iter_corpus_text(files, corpus_dir, add_separators=False):
        out.append(chunk)
        total += len(chunk)
        if total >= max_chars:
            break
    text = "".join(out)[:max_chars]
    return text.strip()


def load_text_array(path: Path, dtype: str):
    """Ouvre le fichier de tokens sans le charger en RAM (mémoire virtuelle)."""
    import numpy as np

    return np.memmap(str(path), dtype=dtype, mode="r")
