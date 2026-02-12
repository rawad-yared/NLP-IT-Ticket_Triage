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


class TicketTriageRuntime:
    def __init__(self, base_dir: Path | None = None, max_length: int = 256) -> None:
        self.base_dir = (base_dir or Path(__file__).resolve().parents[1]).resolve()
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._classifier_cache: dict[str, dict[str, Any]] = {}
        self._summarizer = None
        self._yake = yake.KeywordExtractor(
            lan="en",
            n=3,
            dedupLim=0.9,
            dedupFunc="seqm",
            windowsSize=1,
            top=12,
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

    def extract_tags(self, text: str, top_k: int = 5) -> list[str]:
        clean_text = self.normalize_ticket_text(text)
        if not clean_text:
            return []

        keywords = self._yake.extract_keywords(clean_text)
        tags: list[str] = []
        seen: set[str] = set()

        for phrase, _score in keywords:
            candidate = self.normalize_ticket_text(phrase).lower()
            if len(candidate) < 3:
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            tags.append(candidate)
            if len(tags) >= top_k:
                break
        return tags

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
