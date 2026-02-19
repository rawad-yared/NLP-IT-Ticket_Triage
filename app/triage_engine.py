from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yake
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)


HEADER_LINE_RE = re.compile(r"(?i)^(from|sent|to|subject|cc|bcc):")
SEPARATOR_LINE_RE = re.compile(r"^[-_]{2,}$")
GREETING_LINE_RE = re.compile(r"(?i)^(dear|hi|hello)\b")
SIGNOFF_MARKER_RE = re.compile(
    r"(?i)\b(best regards|kind regards|regards|thanks(?: and regards)?|thank you|sincerely)\b"
)
DISCLAIMER_MARKER_RE = re.compile(
    r"(?i)\b(this email and any attachments are confidential|do not reply to this email)\b"
)

FILLER_STOPWORDS = {
    "i",
    "im",
    "i'm",
    "me",
    "my",
    "we",
    "our",
    "please",
    "hello",
    "thanks",
    "thank",
    "message",
    "reaching",
    "reach",
    "hope",
    "team",
    "contacting",
}
EMAIL_STOPWORDS = {
    "hi",
    "dear",
    "regards",
    "sent",
    "forwarded",
    "subject",
    "from",
    "to",
    "cc",
    "bcc",
}
GENERIC_IT_STOPWORDS = {
    "issue",
    "problem",
    "help",
    "support",
}
GENERIC_NOISE_TOKENS = {
    "customer",
    "service",
    "project",
    "management",
    "platform",
    "team",
    "message",
}


class TicketTriageRuntime:
    def __init__(
        self,
        base_dir: Path | None = None,
        max_length: int = 256,
        include_generic_it_stopwords: bool = True,
    ) -> None:
        self.base_dir = (base_dir or Path(__file__).resolve().parents[1]).resolve()
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        default_stopwords = yake.KeywordExtractor(lan="en").stopword_set
        self._tag_stopwords = set(default_stopwords) | set(FILLER_STOPWORDS) | set(EMAIL_STOPWORDS)
        if include_generic_it_stopwords:
            self._tag_stopwords |= set(GENERIC_IT_STOPWORDS)

        self._classifier_cache: dict[str, dict[str, Any]] = {}
        self._summarizer = None
        self._yake = yake.KeywordExtractor(
            lan="en",
            n=3,
            dedup_lim=0.8,
            dedup_func="seqm",
            window_size=1,
            top=12,
            stopwords=self._tag_stopwords,
        )

    @staticmethod
    def _is_boilerplate_line(line: str) -> bool:
        line = line.strip()
        if not line:
            return False
        if HEADER_LINE_RE.match(line):
            return True
        if SEPARATOR_LINE_RE.match(line):
            return True
        if GREETING_LINE_RE.match(line):
            words = line.rstrip(",:").split()
            if len(words) <= 6 and len(line) <= 45:
                return True
        return False

    @staticmethod
    def _trim_trailing_boilerplate(text: str, marker_re: re.Pattern, min_ratio: float) -> str:
        matches = list(marker_re.finditer(text))
        if not matches:
            return text
        cutoff = matches[-1].start()
        if cutoff >= int(len(text) * min_ratio):
            return text[:cutoff]
        return text

    def normalize_ticket_text(self, text: str) -> str:
        text = "" if pd.isna(text) else str(text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", text)

        lines = [line.strip() for line in re.split(r"[\r\n]+", text) if line.strip()]
        lines = [line for line in lines if not self._is_boilerplate_line(line)]
        text = " ".join(lines)
        text = self._trim_trailing_boilerplate(text, DISCLAIMER_MARKER_RE, min_ratio=0.55)
        text = self._trim_trailing_boilerplate(text, SIGNOFF_MARKER_RE, min_ratio=0.60)

        text = re.sub(r"[^\w\s\.,:;!?\-/]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _task_model_dir(self, task_name: str) -> Path:
        return self.base_dir / "models" / f"{task_name}_model" / "best"

    @staticmethod
    def _normalize_id2label(id2label: dict[Any, str]) -> dict[int, str]:
        out: dict[int, str] = {}
        for k, v in id2label.items():
            out[int(k)] = str(v)
        return out

    def _load_classifier(self, task_name: str) -> dict[str, Any]:
        if task_name in self._classifier_cache:
            return self._classifier_cache[task_name]

        model_dir = self._task_model_dir(task_name)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Missing model directory: {model_dir}. "
                "Run the training notebook first to export best checkpoints."
            )

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        model.to(self.device)
        model.eval()

        id2label = self._normalize_id2label(model.config.id2label)
        runtime = {"tokenizer": tokenizer, "model": model, "id2label": id2label}
        self._classifier_cache[task_name] = runtime
        return runtime

    def predict_label(self, task_name: str, clean_text: str) -> dict[str, Any]:
        runtime = self._load_classifier(task_name)
        tokenizer = runtime["tokenizer"]
        model = runtime["model"]
        id2label = runtime["id2label"]

        encoded = tokenizer(
            clean_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = model(**encoded).logits[0]
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

        pred_id = int(np.argmax(probs))
        confidence = float(probs[pred_id])
        label = id2label.get(pred_id, str(pred_id))
        return {"label": label, "confidence": round(confidence, 4)}

    @staticmethod
    def _normalize_tag_phrase(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _prepare_text_for_tags(self, text: str) -> str:
        clean_text = self.normalize_ticket_text(text).lower()
        clean_text = re.sub(r"[^\w\s]", " ", clean_text)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()
        return clean_text

    @staticmethod
    def _token_overlap_ratio(a: str, b: str) -> float:
        a_tokens = set(a.split())
        b_tokens = set(b.split())
        if not a_tokens or not b_tokens:
            return 0.0
        return len(a_tokens & b_tokens) / min(len(a_tokens), len(b_tokens))

    @classmethod
    def _is_overlapping_variant(cls, candidate: str, kept: str) -> bool:
        if candidate == kept:
            return True
        if candidate in kept or kept in candidate:
            return True
        c_tokens = set(candidate.split())
        k_tokens = set(kept.split())
        overlap = len(c_tokens & k_tokens)
        if overlap >= 2 and cls._token_overlap_ratio(candidate, kept) >= 0.66:
            return True
        return False

    def _dedupe_overlapping_tags(self, candidates: list[tuple[str, float]], top_k: int) -> list[str]:
        best_score_by_phrase: dict[str, float] = {}
        for phrase, score in candidates:
            if phrase not in best_score_by_phrase or score < best_score_by_phrase[phrase]:
                best_score_by_phrase[phrase] = score

        ordered = sorted(
            best_score_by_phrase.items(),
            key=lambda x: (-len(x[0].split()), x[1], -len(x[0])),
        )

        kept: list[str] = []
        for phrase, _score in ordered:
            if any(self._is_overlapping_variant(phrase, existing) for existing in kept):
                continue
            kept.append(phrase)
            if len(kept) >= top_k:
                break
        return kept

    def extract_tags(self, text: str, top_k: int = 5) -> list[str]:
        tag_text = self._prepare_text_for_tags(text)
        if not tag_text:
            return []

        keywords = self._yake.extract_keywords(tag_text)
        candidates: list[tuple[str, float]] = []

        for phrase, score in keywords:
            normalized = self._normalize_tag_phrase(phrase)
            if len(normalized) < 3:
                continue

            filtered_tokens = [
                token
                for token in normalized.split()
                if token not in self._tag_stopwords and len(token) > 1
            ]
            filtered_tokens = list(dict.fromkeys(filtered_tokens))
            informative_tokens = [
                token for token in filtered_tokens if token not in GENERIC_NOISE_TOKENS
            ]
            if not informative_tokens:
                continue

            filtered = " ".join(filtered_tokens).strip()
            if len(filtered) < 3:
                continue

            candidates.append((filtered, float(score)))

        return self._dedupe_overlapping_tags(candidates, top_k=top_k)

    def _get_summarizer(self):
        if self._summarizer is None:
            device = 0 if torch.cuda.is_available() else -1
            self._summarizer = pipeline(
                "summarization",
                model="t5-small",
                tokenizer="t5-small",
                device=device,
            )
        return self._summarizer

    def summarize_text(self, text: str) -> str:
        clean_text = self.normalize_ticket_text(text)
        if not clean_text:
            return ""

        words = clean_text.split()
        if len(words) < 24:
            return clean_text

        input_text = f"summarize: {clean_text[:3000]}"
        max_len = min(72, max(24, int(len(words) * 0.6)))
        min_len = min(max_len - 4, max(10, int(len(words) * 0.2)))

        summarizer = self._get_summarizer()
        output = summarizer(
            input_text,
            max_length=max_len,
            min_length=min_len,
            do_sample=False,
            truncation=True,
        )
        return self.normalize_ticket_text(output[0]["summary_text"])

    def triage_ticket(
        self,
        ticket_text: str,
        ticket_id: str = "ad_hoc",
        top_k_tags: int = 5,
        include_summary: bool = True,
    ) -> dict[str, Any]:
        clean_text = self.normalize_ticket_text(ticket_text)
        if not clean_text:
            raise ValueError("Ticket text is empty after cleanup. Please provide more detail.")

        department = self.predict_label("department", clean_text)
        urgency = self.predict_label("urgency", clean_text)
        tags = self.extract_tags(clean_text, top_k=top_k_tags)
        summary = self.summarize_text(clean_text) if include_summary else ""

        return {
            "ticket_id": str(ticket_id),
            "department": department,
            "urgency": urgency,
            "tags": tags,
            "summary": summary,
        }
