#!/usr/bin/env python3
"""Build the Mooré plain-text JSONL for continued pretraining (`--data_format text`).

One row per document: {"text", "source", "doc_id"}. Sources:

- council: Council of Ministers reports, Mooré sections of `madoss/conseil-ministres-parsed`.
- raamde: Raamde news articles, the Mooré run moore-web's `segment_news_data.py` split off.
- ministereduc: Ministry of Education Facebook posts, `mos.txt` of the ministereduc-bf mirror.
- faso_docs: Mooré running text with a text layer from the faso-web-docs mirror (FASO_DOCS).

Held-out sentences (moore-web validation/test, FLORES+ dev/devtest, Bouquet dev/test) are
removed sentence by sentence, not document by document: moore-web splits by row, so every
Council session holds some of them. The build fails if any held-out sentence of 4+ words and
20+ characters is still found inside an output document.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from evaluate_translation import bouquet_rows
from mix_translation_data import (
    MOORE_WEB,
    flores_texts_by_id,
    normalize_text,
    read_hf_dataset,
)

load_dotenv()

COUNCIL_DATASET = "madoss/conseil-ministres-parsed"
COUNCIL_REVISION = "841e61738c1afb82a96355dc4d0dc17978dd4904"  # 93 Mooré sessions, up to 2026-09-10
OPEN_SOURCE = Path(__file__).resolve().parents[2]

# Mooré running text with a text layer, relative to the faso-web-docs root. Left out: dictionaries
# and indexes (lexicon entries), burkinadoc scans (need OCR), the two-column SIDA book and the
# expert batch (bilingual layouts the line filter can't separate).
FASO_DOCS = [
    "mooreburkina-priority/apps/mos-contes-volume-1/text/*.txt",
    "mooreburkina-priority/apps/mos-contes-volume-2/text/*.txt",
    "mooreburkina-priority/apps/mos-proverbes-volume-1/text/*.txt",  # French gloss lines filtered
    "mooreburkina-priority/pdf/16-pag-ne-a-sid-viim.pdf",
    "mooreburkina-priority/pdf/20-sida-histoire-de-poko-en-moore.pdf",
    "kade-facilitator-manual/SIDA Histoire de Poko en moore livre du Facilitateur.pdf",
    # Not the three "Du Mooré au Français" parts: French lessons, mostly French fragments.
    "burkinadoc/du-moore-literacy-series/Livre_de_Lecture_Moore_*.pdf",
    "burkinadoc/du-moore-literacy-series/Livre_de_Grammaire_Moore_*.pdf",
    "universal-declaration-human-rights/udhr-mos.txt",
    "messages-nouvel-an/2024/Message_nouvel_an_moore.txt",
    "abcburkina-contes/text/267-les-co-epouses-moore.txt",
]
# Bilingual apps with one language per page: only the pages narrated in Mooré, i.e. those with
# audio timings in segments.jsonl (tale volume 5: 30 Mooré pages; its 30 French pages have none).
FASO_NARRATED_APPS = ["mooreburkina-priority/apps/mos-contes-volume-5"]

# French function words that are not Mooré words ("la", "a", "ne", "me" are, so they are left out).
FRENCH_WORDS = {
    "le", "les", "de", "des", "du", "et", "est", "un", "une", "en", "au", "aux", "pour", "que",
    "qui", "dans", "par", "sur", "il", "elle", "ils", "elles", "nous", "vous", "on", "pas",
    "plus", "ce", "cette", "ces", "son", "sa", "ses", "leur", "leurs", "avec", "mais", "ou",
    "où", "été", "être", "sont", "était", "comme", "tout", "tous", "très", "aussi", "je", "tu",
    "se", "lui", "y"
}
FRENCH_SHARE = 0.25
# Look-alikes some PDFs use for Mooré letters (script g, Greek iota, upsilon for ʋ), and no-break spaces.
CHAR_FIXES = str.maketrans({"\u0261": "g", "\u1fd6": "ĩ", "\u028a": "ʋ", "\u01b1": "Ʋ", "\xa0": " "})
SENTENCE_END = re.compile(r"(?<=[.!?»;:])\s+|\n+")
# Mooré-only letters. A paragraph of 20+ words without any is French or English (job ads glued
# without spaces, English credits, French tale versions); shorter Mooré often has none (syllable
# drills, dialogue), so the rule is not applied to sentences.
MOORE_LETTERS = re.compile(r"[ɩʋẽãõĩũɛƐƲƖẼÃÕĨŨ]")
MIN_WORDS_WITHOUT_MOORE_LETTERS = 20
MIN_DOCUMENT_WORDS = 5
MIN_SUBSTRING_WORDS = 4
MIN_SUBSTRING_CHARS = 20  # shorter ones ("a b c d", list letters) only match whole sentences


@dataclass
class Document:
    source: str
    doc_id: str
    paragraphs: list[str]


def fix_characters(text: str) -> str:
    """NFC (e + combining tilde -> ẽ) and the CHAR_FIXES look-alikes."""
    return unicodedata.normalize("NFC", text.translate(CHAR_FIXES))


def held_out_sentences(flores_dataset: str) -> set[str]:
    """Normalized Mooré sentences of every evaluation set."""
    texts = set()
    for split in ("validation", "test"):
        texts.update(normalize_text(row["moore"] or "") for row in read_hf_dataset(MOORE_WEB, split=split, config="mos-fra"))
    for split in ("dev", "devtest"):
        texts.update(normalize_text(t) for t in flores_texts_by_id(flores_dataset, split, "mos").values())
    for level in ("sentence_level", "paragraph_level"):
        for split in ("dev", "test"):
            rows = bouquet_rows(level, split)
            for text in [r["ref_text"] for r in rows[("fra", "mos")]] + [r["src_text"] for r in rows[("mos", "fra")]]:
                texts.add(normalize_text(text))
                texts.update(normalize_text(s) for s in SENTENCE_END.split(text))
    texts.discard("")
    return texts


class Decontaminator:
    """Drops sentences equal to a held-out sentence or containing a long one, and French sentences."""

    def __init__(self, held_out: set[str]):
        self.exact = held_out
        self.long = [h for h in held_out if len(h.split()) >= MIN_SUBSTRING_WORDS and len(h) >= MIN_SUBSTRING_CHARS]
        self.by_trigram: dict[str, list[str]] = defaultdict(list)
        for h in self.long:
            self.by_trigram[" ".join(h.split()[:3])].append(h)

    def contains_held_out(self, normalized: str) -> list[str]:
        words = normalized.split()
        padded = f" {normalized} "
        found = []
        for i in range(len(words) - 2):
            for h in self.by_trigram.get(" ".join(words[i : i + 3]), ()):
                if f" {h} " in padded:
                    found.append(h)
        return found

    def clean(self, paragraphs: list[str], stats: Counter) -> list[str]:
        kept = []  # one list of sentences per paragraph
        for paragraph in paragraphs:
            sentences = []
            for sentence in SENTENCE_END.split(paragraph):
                key = normalize_text(sentence)
                if not key:
                    continue
                if key in self.exact or self.contains_held_out(key):
                    stats["held_out_sentences_removed"] += 1
                elif is_french(sentence):  # e.g. French left in Council Mooré sections
                    stats["french_sentences_dropped"] += 1
                else:
                    sentences.append(sentence.strip())
            kept.append(sentences)
        # A held-out sentence that our splitter (e.g. at ":" inside quoted speech) or a paragraph
        # break (a Raamde title and its first line) cut across: find it in the document's
        # normalized text and drop every sentence whose span overlaps it.
        spans, joined, offset = [], [], 0
        for i, sentences in enumerate(kept):
            for j, sentence in enumerate(sentences):
                key = normalize_text(sentence)
                spans.append((offset, offset + len(key), i, j))
                joined.append(key)
                offset += len(key) + 1
        text = " ".join(joined)
        drop = set()
        for h in set(self.contains_held_out(text)):
            start = text.find(h)
            while start != -1:
                end = start + len(h)
                drop.update((i, j) for s, e, i, j in spans if s < end and start < e)
                start = text.find(h, start + 1)
        stats["held_out_pieces_removed"] += len(drop)
        kept = [[s for j, s in enumerate(sentences) if (i, j) not in drop] for i, sentences in enumerate(kept)]
        return [" ".join(sentences) for sentences in kept if sentences]


def is_french(line: str) -> bool:
    words = re.findall(r"\w+", line.lower())
    return len(words) >= 3 and sum(w in FRENCH_WORDS for w in words) / len(words) >= FRENCH_SHARE


def paragraphs_from_text(text: str, stats: Counter) -> list[str]:
    """Blank lines and form feeds separate paragraphs; wrapped lines are joined; French lines dropped."""
    paragraphs = []
    for block in re.split(r"\n\s*\n|\f", text):
        joined = ""
        for line in block.splitlines():
            line = line.strip()
            if not line or re.fullmatch(r"[\W\d_]+", line):
                continue
            if is_french(line):
                stats["french_lines_dropped"] += 1
                continue
            joined = joined + line if joined.endswith("-") or not joined else f"{joined} {line}"
        if joined:
            paragraphs.append(joined)
    return paragraphs


def council_documents(council_file: str | None, revision: str) -> Iterator[Document]:
    if council_file is None:
        from huggingface_hub import hf_hub_download

        council_file = hf_hub_download(COUNCIL_DATASET, "corpus.json", repo_type="dataset", revision=revision)
    for session in json.loads(Path(council_file).read_text()):
        paragraphs = [" ".join(([s["title"]] if s.get("title") else []) + s["sentences"]) for s in session["mos"]]
        if paragraphs:
            yield Document("council", session["date"], paragraphs)


def raamde_documents(path: Path) -> Iterator[Document]:
    """moore-web's Mooré run (`segments.moore`): the units before the "Kibarã yii" credit marker,
    else before the first French unit. Per-unit GlotLID labels mislabel credit lines and miss
    unlabelled articles; the run is complete."""
    for article in json.loads(path.read_text()):
        if "recrutement" in article["url"]:  # French job ads, not news (4 pages)
            continue
        paragraphs = (article.get("segments") or {}).get("moore") or []
        if paragraphs:
            yield Document("raamde", article["url"], paragraphs)


def ministereduc_documents(root: Path, stats: Counter) -> Iterator[Document]:
    for path in sorted(root.glob("20*/*/*/mos.txt")):
        status = json.loads((path.parent / "meta.json").read_text()).get("status")
        if status == "rejected":
            continue
        yield Document("ministereduc", path.parent.name, paragraphs_from_text(path.read_text(), stats))


def faso_text(path: Path) -> str:
    if path.suffix == ".pdf":
        return subprocess.run(["pdftotext", str(path), "-"], capture_output=True, text=True, check=True).stdout
    text = path.read_text()
    if path.parent.name == "text" and "abcburkina" in str(path):
        text = re.sub(r"\A(?:(?:Title|Source|Languages|Fetched): .*\n)+", "", text)
    return text.replace("&1", "")  # UDHR article 12 placeholder, see faso-web-docs SOURCES.md


def faso_documents(root: Path, stats: Counter) -> Iterator[Document]:
    if shutil.which("pdftotext") is None:
        sys.exit("pdftotext not found (install poppler-utils).")
    paths = []
    for pattern in FASO_DOCS:
        matches = sorted(root.glob(pattern))
        if not matches:
            sys.exit(f"No faso-web-docs file matches {pattern!r} under {root}.")
        paths += matches
    for app in FASO_NARRATED_APPS:
        segments = (root / app / "segments.jsonl").read_text().splitlines()
        pages = dict.fromkeys(json.loads(line)["page"] for line in segments)
        paths += [root / app / "text" / Path(page).with_suffix(".txt").name for page in pages]
    for path in paths:
        paragraphs = paragraphs_from_text(faso_text(path), stats)
        yield Document("faso_docs", str(path.relative_to(root)), paragraphs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("data/cpt_mos_v1.jsonl"))
    parser.add_argument("--validation-output", type=Path, default=Path("data/cpt_mos_validation_v1.jsonl"))
    parser.add_argument("--validation-docs", type=int, default=20, help="Documents held out for CPT eval loss.")
    parser.add_argument("--sources", nargs="+", default=["council", "raamde", "ministereduc", "faso_docs"])
    parser.add_argument("--council-file", help="A local corpus.json (e.g. a newer build) instead of the Hub release.")
    parser.add_argument("--council-revision", default=COUNCIL_REVISION)
    parser.add_argument("--raamde-file", type=Path, default=OPEN_SOURCE / "moore-web/data/raamde/raamde_corpus_with_glotlid.json")
    parser.add_argument("--ministereduc-dir", type=Path, default=OPEN_SOURCE / "ministereduc_bf/data")
    parser.add_argument("--faso-docs-dir", type=Path, default=OPEN_SOURCE / "faso-web-docs")
    parser.add_argument("--flores-dataset", default="openlanguagedata/flores_plus")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats: Counter = Counter()
    readers = {
        "council": lambda: council_documents(args.council_file, args.council_revision),
        "raamde": lambda: raamde_documents(args.raamde_file),
        "ministereduc": lambda: ministereduc_documents(args.ministereduc_dir, stats),
        "faso_docs": lambda: faso_documents(args.faso_docs_dir, stats),
    }
    unknown = set(args.sources) - set(readers)
    if unknown:
        sys.exit(f"Unknown sources {sorted(unknown)}; choose from {sorted(readers)}.")

    decontaminator = Decontaminator(held_out_sentences(args.flores_dataset))
    rows, seen = [], set()
    words, docs = Counter(), Counter()
    for source in args.sources:
        for doc in readers[source]():
            paragraphs = decontaminator.clean([fix_characters(p) for p in doc.paragraphs], stats)
            kept = [p for p in paragraphs if len(p.split()) < MIN_WORDS_WITHOUT_MOORE_LETTERS or MOORE_LETTERS.search(p)]
            stats["paragraphs_without_moore_letters_dropped"] += len(paragraphs) - len(kept)
            text = "\n\n".join(kept)
            if len(text.split()) < MIN_DOCUMENT_WORDS:
                stats["documents_under_5_words"] += 1
                continue
            key = hashlib.sha256(normalize_text(text).encode()).hexdigest()
            if not text or key in seen:
                stats["empty_or_duplicate_documents"] += 1
                continue
            seen.add(key)
            leaks = decontaminator.contains_held_out(normalize_text(text))
            if leaks:
                sys.exit(f"{doc.source} {doc.doc_id} still contains held-out text: {leaks[:3]}")
            rows.append({"text": text, "source": doc.source, "doc_id": doc.doc_id})
            words[source] += len(text.split())
            docs[source] += 1

    random.Random(args.seed).shuffle(rows)
    validation, train = rows[: args.validation_docs], rows[args.validation_docs :]
    for path, part in ((args.output, train), (args.validation_output, validation)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in part)
        print(f"Wrote {len(part)} documents to {path}")
    for source in args.sources:
        print(f"  {source:13s} {docs[source]:5d} docs {words[source]:9,d} words")
    print(f"  total         {sum(docs.values()):5d} docs {sum(words.values()):9,d} words")
    for name, count in sorted(stats.items()):
        print(f"  {name}: {count:,}")


if __name__ == "__main__":
    main()
