#!/usr/bin/env python3
from __future__ import annotations

import base64
import csv
import datetime as dt
import io
import json
import os
import queue
import re
import threading
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import fitz  # PyMuPDF
import requests
import tkinter as tk
from PIL import Image
from tkinter import filedialog, messagebox, simpledialog, ttk
from tkinter.scrolledtext import ScrolledText

APP_NAME = "Microfiche Overlap Extractor"
APP_VERSION = "0.1.0"

DEFAULT_CLASSIFY_PROMPT = (
    "Task: detect microfiche page overlap.\n"
    "Definition of overlap: two different record cards are merged/superimposed in one scan, "
    "including clear side-by-side merge OR ghost/partial superimposition.\n"
    "Return strict JSON only with keys:\n"
    "- is_overlap: boolean\n"
    "- confidence: number in [0,1]\n"
    "- overlap_type: one of [clear_double_card, ghost_superimposition, none]\n"
    "- signatures: array of 0..2 concise identity strings (e.g. NAME|DOB|STUDENTNO)\n"
    "- reason: short string\n"
)


def now_ts() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def now_file_ts() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def app_data_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", str(Path.home())))
        root = base / "MicroficheOverlapExtractor"
    else:
        root = Path.home() / ".microfiche_overlap_extractor"
    root.mkdir(parents=True, exist_ok=True)
    return root


@dataclass
class ModelProfile:
    name: str
    base_url: str
    model: str
    api_key: str = ""
    timeout_sec: int = 120


class Storage:
    def __init__(self) -> None:
        self.root = app_data_dir()
        self.models_path = self.root / "models_config.json"
        self.memory_path = self.root / "memory_store.json"
        self.last_scan_path = self.root / "last_scan.json"

    def load_models(self) -> List[ModelProfile]:
        if not self.models_path.exists():
            models = self.default_models()
            self.save_models(models)
            return models
        try:
            raw = json.loads(self.models_path.read_text(encoding="utf-8"))
            out: List[ModelProfile] = []
            for x in raw:
                out.append(
                    ModelProfile(
                        name=str(x.get("name", "")),
                        base_url=str(x.get("base_url", "")),
                        model=str(x.get("model", "")),
                        api_key=str(x.get("api_key", "")),
                        timeout_sec=int(x.get("timeout_sec", 120)),
                    )
                )
            return out or self.default_models()
        except Exception:
            return self.default_models()

    def save_models(self, models: List[ModelProfile]) -> None:
        self.models_path.write_text(
            json.dumps([asdict(m) for m in models], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def load_memory(self) -> Dict[str, Any]:
        if not self.memory_path.exists():
            data = {"global_notes": [], "overrides": {}}
            self.save_memory(data)
            return data
        try:
            data = json.loads(self.memory_path.read_text(encoding="utf-8"))
            data.setdefault("global_notes", [])
            data.setdefault("overrides", {})
            return data
        except Exception:
            data = {"global_notes": [], "overrides": {}}
            self.save_memory(data)
            return data

    def save_memory(self, memory: Dict[str, Any]) -> None:
        self.memory_path.write_text(
            json.dumps(memory, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def save_last_scan(self, records: List[Dict[str, Any]]) -> None:
        self.last_scan_path.write_text(
            json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def load_last_scan(self) -> List[Dict[str, Any]]:
        if not self.last_scan_path.exists():
            return []
        try:
            return json.loads(self.last_scan_path.read_text(encoding="utf-8"))
        except Exception:
            return []

    @staticmethod
    def default_models() -> List[ModelProfile]:
        return [
            ModelProfile(
                name="GPT-5.3 (Third-party GPT Pool)",
                base_url="https://gmn.chuangzuoli.com/v1",
                model="gpt-5.3",
                api_key="sk-c459a469d12f8896aeaa45ea94e0e5f7b3eafbe6fcc98dfff86fad74e724bd5d",
                timeout_sec=120,
            ),
            ModelProfile(
                name="GPT-5.2 (Third-party GPT Pool)",
                base_url="https://gmn.chuangzuoli.com/v1",
                model="gpt-5.2",
                api_key="sk-c459a469d12f8896aeaa45ea94e0e5f7b3eafbe6fcc98dfff86fad74e724bd5d",
                timeout_sec=120,
            ),
            ModelProfile(
                name="Claude-Opus-4.6 (Third-party Claude Pool)",
                base_url="https://cursor.scihub.edu.kg/api/v1",
                model="claude-opus-4-6",
                api_key="cr_56c958bfb141949f0a7e3ce7bf9e83315fe7695edf95749683c05b234c594000",
                timeout_sec=150,
            ),
            ModelProfile(
                name="Kimi-K2.5 (Bailian)",
                base_url="https://coding.dashscope.aliyuncs.com/v1",
                model="kimi-k2.5",
                api_key="sk-sp-a745d056ce96479c899d2b5d9c40d345",
                timeout_sec=120,
            ),
            ModelProfile(
                name="GLM-5 (Bailian)",
                base_url="https://coding.dashscope.aliyuncs.com/v1",
                model="glm-5",
                api_key="sk-sp-a745d056ce96479c899d2b5d9c40d345",
                timeout_sec=120,
            ),
            ModelProfile(
                name="Minimax-M2.5 (Bailian)",
                base_url="https://coding.dashscope.aliyuncs.com/v1",
                model="MiniMax-M2.5",
                api_key="sk-sp-a745d056ce96479c899d2b5d9c40d345",
                timeout_sec=120,
            ),
        ]


def parse_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        text = m.group(0)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {}


def norm_sig(sig: str) -> str:
    s = re.sub(r"[^A-Za-z0-9|]+", " ", sig.upper()).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def sig_tokens(sig: str) -> set[str]:
    return set([x for x in re.split(r"[|\s]+", norm_sig(sig)) if x])


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    uni = len(a | b)
    return inter / uni if uni else 0.0


def list_pdfs(root: Path, recursive: bool) -> List[Path]:
    if recursive:
        return sorted([p for p in root.rglob("*.pdf") if p.is_file()])
    return sorted([p for p in root.glob("*.pdf") if p.is_file()])


def render_page_jpeg(page: fitz.Page, dpi: int = 220, max_width: int = 960, quality: int = 55) -> bytes:
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    if img.width > max_width:
        h = int(img.height * max_width / img.width)
        img = img.resize((max_width, h), Image.Resampling.LANCZOS)
    bio = io.BytesIO()
    img.save(bio, format="JPEG", quality=quality, optimize=True)
    return bio.getvalue()


class OpenAICompatibleClient:
    def __init__(self, profile: ModelProfile):
        self.profile = profile

    def _post(self, payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any], str]:
        url = self.profile.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.profile.api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.profile.timeout_sec,
            )
            txt = resp.text
            try:
                obj = resp.json()
            except Exception:
                obj = {}
            return resp.status_code, obj, txt
        except Exception as exc:
            return -1, {}, f"request_error: {type(exc).__name__}: {exc}"

    def classify_page(
        self,
        image_jpeg: bytes,
        file_name: str,
        page_no: int,
        custom_prompt: str,
        memory_notes: List[str],
    ) -> Dict[str, Any]:
        b64 = base64.b64encode(image_jpeg).decode("ascii")
        rules = ""
        if memory_notes:
            joined = "\n".join([f"- {x}" for x in memory_notes[:20]])
            rules = f"\nLearned rules from prior corrections:\n{joined}\n"

        prompt = (
            DEFAULT_CLASSIFY_PROMPT
            + rules
            + f"\nfile={file_name}\npage={page_no}\n"
        )
        if custom_prompt.strip():
            prompt += f"\nCustom instructions:\n{custom_prompt.strip()}\n"

        payload_openai = {
            "model": self.profile.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 320,
        }

        status, obj, raw = self._post(payload_openai)
        if status == 200 and obj:
            msg = obj.get("choices", [{}])[0].get("message", {}).get("content", "")
            parsed = parse_json_object(msg)
            if parsed:
                return {
                    "ok": True,
                    "status": status,
                    "raw": msg,
                    "json": parsed,
                    "usage": obj.get("usage", {}),
                }
            # non-json soft fail
            return {
                "ok": False,
                "status": status,
                "raw": msg,
                "error": "Model response is not valid JSON.",
                "usage": obj.get("usage", {}),
            }

        # fallback shape for some gateways that proxy Anthropic-like format
        payload_alt = {
            "model": self.profile.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64,
                            },
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 320,
        }
        status2, obj2, raw2 = self._post(payload_alt)
        if status2 == 200 and obj2:
            msg = obj2.get("choices", [{}])[0].get("message", {}).get("content", "")
            parsed = parse_json_object(msg)
            if parsed:
                return {
                    "ok": True,
                    "status": status2,
                    "raw": msg,
                    "json": parsed,
                    "usage": obj2.get("usage", {}),
                }
            return {
                "ok": False,
                "status": status2,
                "raw": msg,
                "error": "Fallback response is not valid JSON.",
                "usage": obj2.get("usage", {}),
            }

        return {
            "ok": False,
            "status": status2 if status2 != -1 else status,
            "raw": raw2 if raw2 else raw,
            "error": "Image classification failed on both request formats.",
        }

    def quick_test(self) -> Tuple[bool, str]:
        # 1x1 png transparent pixel
        tiny = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc``\xf8\x0f\x00\x01\x04"
            b"\x01\x00\x18\xdd\x8d\x18\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        b64 = base64.b64encode(tiny).decode("ascii")
        payload = {
            "model": self.profile.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Reply exactly OK."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 8,
        }
        status, obj, raw = self._post(payload)
        if status == 200 and obj:
            msg = obj.get("choices", [{}])[0].get("message", {}).get("content", "")
            return True, f"HTTP 200. Reply: {msg[:120]!r}"
        return False, f"HTTP {status}. {raw[:220]}"


class OverlapEngine:
    def __init__(
        self,
        client: OpenAICompatibleClient,
        memory: Dict[str, Any],
        logger,
        cancel_event: threading.Event,
        progress_cb,
    ) -> None:
        self.client = client
        self.memory = memory
        self.log = logger
        self.cancel_event = cancel_event
        self.progress_cb = progress_cb

    def memory_override(self, file_name: str, page_no: int) -> Optional[Dict[str, Any]]:
        key = f"{file_name.lower()}::{page_no}"
        return self.memory.get("overrides", {}).get(key)

    def scan_pdfs(
        self,
        pdf_paths: List[Path],
        scope: str,
        custom_prompt: str,
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        total_pages = 0
        for p in pdf_paths:
            try:
                total_pages += len(fitz.open(str(p)))
            except Exception:
                pass
        done = 0
        self.progress_cb(done, max(total_pages, 1))

        for pdf_path in pdf_paths:
            if self.cancel_event.is_set():
                self.log("Scan cancelled.")
                break

            self.log(f"Scanning: {pdf_path}")
            try:
                doc = fitz.open(str(pdf_path))
            except Exception as exc:
                self.log(f"Failed to open {pdf_path}: {exc}")
                continue

            for idx in range(len(doc)):
                if self.cancel_event.is_set():
                    self.log("Scan cancelled.")
                    break
                page_no = idx + 1
                file_name = pdf_path.name
                override = self.memory_override(file_name, page_no)
                if override:
                    rec = {
                        "source_directory": str(pdf_path.parent),
                        "file_name": file_name,
                        "file_path": str(pdf_path),
                        "page": page_no,
                        "is_overlap": bool(override.get("is_overlap", False)),
                        "confidence": float(override.get("confidence", 1.0)),
                        "overlap_type": str(override.get("overlap_type", "manual_override")),
                        "signatures": override.get("signatures", []),
                        "reason": str(override.get("note", "manual memory override")),
                        "scope": scope,
                        "model": self.client.profile.model,
                        "status": "memory_override",
                    }
                    records.append(rec)
                    done += 1
                    self.progress_cb(done, max(total_pages, 1))
                    continue

                try:
                    image_jpeg = render_page_jpeg(doc[idx])
                except Exception as exc:
                    self.log(f"Render failed {pdf_path} p{page_no}: {exc}")
                    rec = {
                        "source_directory": str(pdf_path.parent),
                        "file_name": file_name,
                        "file_path": str(pdf_path),
                        "page": page_no,
                        "is_overlap": False,
                        "confidence": 0.0,
                        "overlap_type": "none",
                        "signatures": [],
                        "reason": f"render_error: {exc}",
                        "scope": scope,
                        "model": self.client.profile.model,
                        "status": "error",
                    }
                    records.append(rec)
                    done += 1
                    self.progress_cb(done, max(total_pages, 1))
                    continue

                result = self.client.classify_page(
                    image_jpeg=image_jpeg,
                    file_name=file_name,
                    page_no=page_no,
                    custom_prompt=custom_prompt,
                    memory_notes=self.memory.get("global_notes", []),
                )

                if not result.get("ok"):
                    self.log(
                        f"LLM failed {file_name} p{page_no}: "
                        f"status={result.get('status')} err={result.get('error')}"
                    )
                    rec = {
                        "source_directory": str(pdf_path.parent),
                        "file_name": file_name,
                        "file_path": str(pdf_path),
                        "page": page_no,
                        "is_overlap": False,
                        "confidence": 0.0,
                        "overlap_type": "none",
                        "signatures": [],
                        "reason": f"llm_error: {result.get('error', 'unknown')}",
                        "scope": scope,
                        "model": self.client.profile.model,
                        "status": "llm_error",
                    }
                else:
                    obj = result.get("json", {})
                    sigs = obj.get("signatures", [])
                    if not isinstance(sigs, list):
                        sigs = []
                    sigs = [norm_sig(str(s)) for s in sigs if str(s).strip()][:2]
                    rec = {
                        "source_directory": str(pdf_path.parent),
                        "file_name": file_name,
                        "file_path": str(pdf_path),
                        "page": page_no,
                        "is_overlap": bool(obj.get("is_overlap", False)),
                        "confidence": float(obj.get("confidence", 0.0)),
                        "overlap_type": str(obj.get("overlap_type", "none")),
                        "signatures": sigs,
                        "reason": str(obj.get("reason", ""))[:500],
                        "scope": scope,
                        "model": self.client.profile.model,
                        "status": "ok",
                    }
                records.append(rec)
                done += 1
                self.progress_cb(done, max(total_pages, 1))

            doc.close()

        return records


def write_overlap_csv(records: List[Dict[str, Any]], out_csv: Path) -> int:
    overlaps = [r for r in records if r.get("is_overlap") and r.get("scope") == "source"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source_directory",
        "file_name",
        "file_path",
        "page",
        "is_overlap",
        "confidence",
        "overlap_type",
        "signatures",
        "reason",
        "model",
        "status",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in overlaps:
            row = dict(r)
            row["signatures"] = " | ".join(r.get("signatures", []))
            w.writerow({k: row.get(k, "") for k in fields})
    return len(overlaps)


def export_overlap_pages(records: List[Dict[str, Any]], logger) -> int:
    targets = [r for r in records if r.get("scope") == "source" and r.get("is_overlap")]
    by_file: Dict[str, List[int]] = {}
    for r in targets:
        by_file.setdefault(r["file_path"], []).append(int(r["page"]))

    created = 0
    for file_path, pages in by_file.items():
        src = Path(file_path)
        try:
            doc = fitz.open(str(src))
        except Exception as exc:
            logger(f"Open failed {src}: {exc}")
            continue
        for page in sorted(set(pages)):
            out = src.parent / f"O_{src.stem}_P{page}.pdf"
            try:
                one = fitz.open()
                one.insert_pdf(doc, from_page=page - 1, to_page=page - 1)
                one.save(str(out))
                one.close()
                created += 1
            except Exception as exc:
                logger(f"Export overlap page failed {src.name} p{page}: {exc}")
        doc.close()
    return created


def export_extracted_non_overlap(records: List[Dict[str, Any]], logger) -> int:
    by_file: Dict[str, Dict[int, bool]] = {}
    for r in records:
        if r.get("scope") != "source":
            continue
        by_file.setdefault(r["file_path"], {})[int(r["page"])] = bool(r.get("is_overlap"))

    created = 0
    for file_path, marks in by_file.items():
        src = Path(file_path)
        try:
            doc = fitz.open(str(src))
        except Exception as exc:
            logger(f"Open failed {src}: {exc}")
            continue
        keep_pages = [p for p, is_ov in sorted(marks.items()) if not is_ov]
        if not keep_pages:
            logger(f"No non-overlap pages for {src.name}, skip E_ output.")
            doc.close()
            continue
        out = src.parent / f"E_{src.name}"
        try:
            out_doc = fitz.open()
            for p in keep_pages:
                out_doc.insert_pdf(doc, from_page=p - 1, to_page=p - 1)
            out_doc.save(str(out))
            out_doc.close()
            created += 1
        except Exception as exc:
            logger(f"Create E_ file failed {src.name}: {exc}")
        doc.close()
    return created


def find_best_replacement(
    overlap_rec: Dict[str, Any],
    candidate_recs: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    target_sigs = [sig_tokens(s) for s in overlap_rec.get("signatures", []) if s]
    if not target_sigs:
        return None

    best: Optional[Dict[str, Any]] = None
    best_score = 0.0
    for c in candidate_recs:
        if c.get("is_overlap"):
            continue
        if c.get("file_path") == overlap_rec.get("file_path") and int(c.get("page", 0)) == int(
            overlap_rec.get("page", 0)
        ):
            continue
        cand_sigs = [sig_tokens(s) for s in c.get("signatures", []) if s]
        if not cand_sigs:
            continue

        score = 0.0
        for t in target_sigs:
            for cs in cand_sigs:
                score = max(score, jaccard(t, cs))

        # Slight preference for high-confidence non-overlap candidates.
        score += min(0.2, max(0.0, float(c.get("confidence", 0.0)) * 0.2))

        if score > best_score:
            best_score = score
            best = c

    if best_score < 0.35:
        return None
    return best


def replace_overlap_pages(
    source_records: List[Dict[str, Any]],
    all_records: List[Dict[str, Any]],
    logger,
) -> Tuple[int, Path]:
    source_by_file: Dict[str, List[Dict[str, Any]]] = {}
    for r in source_records:
        source_by_file.setdefault(r["file_path"], []).append(r)

    candidates = [r for r in all_records if not r.get("is_overlap")]

    report_rows: List[Dict[str, Any]] = []
    replaced_files = 0
    doc_cache: Dict[str, fitz.Document] = {}

    def get_doc(path: str) -> fitz.Document:
        if path not in doc_cache:
            doc_cache[path] = fitz.open(path)
        return doc_cache[path]

    for file_path, recs in source_by_file.items():
        src = Path(file_path)
        overlap_pages = [r for r in recs if r.get("is_overlap")]
        if not overlap_pages:
            continue
        page_replacements: Dict[int, Optional[Dict[str, Any]]] = {}
        for ov in overlap_pages:
            cand = find_best_replacement(ov, candidates)
            page_replacements[int(ov["page"])] = cand
            report_rows.append(
                {
                    "source_file": str(src),
                    "source_page": int(ov["page"]),
                    "source_signatures": " | ".join(ov.get("signatures", [])),
                    "replacement_file": cand.get("file_path") if cand else "",
                    "replacement_page": int(cand.get("page", 0)) if cand else "",
                    "status": "replaced" if cand else "not_found",
                }
            )

        try:
            src_doc = get_doc(str(src))
            out_doc = fitz.open()
            for i in range(len(src_doc)):
                p = i + 1
                cand = page_replacements.get(p)
                if cand:
                    cand_doc = get_doc(cand["file_path"])
                    out_doc.insert_pdf(cand_doc, from_page=int(cand["page"]) - 1, to_page=int(cand["page"]) - 1)
                else:
                    out_doc.insert_pdf(src_doc, from_page=i, to_page=i)
            out = src.parent / f"R_{src.name}"
            out_doc.save(str(out))
            out_doc.close()
            replaced_files += 1
        except Exception as exc:
            logger(f"Replacement failed for {src.name}: {exc}")

    for d in doc_cache.values():
        try:
            d.close()
        except Exception:
            pass

    report_path = app_data_dir() / f"replacement_report_{now_file_ts()}.csv"
    with report_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "source_file",
                "source_page",
                "source_signatures",
                "replacement_file",
                "replacement_page",
                "status",
            ],
        )
        w.writeheader()
        w.writerows(report_rows)

    return replaced_files, report_path


def import_training_csv(memory: Dict[str, Any], csv_path: Path) -> Tuple[int, int]:
    added = 0
    notes_added = 0
    overrides = memory.setdefault("overrides", {})
    notes = memory.setdefault("global_notes", [])
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            file_name = (
                row.get("file_name")
                or row.get("file")
                or Path(row.get("file_path", "")).name
                or ""
            ).strip()
            page_raw = row.get("page") or row.get("source_page") or ""
            lbl_raw = row.get("is_overlap") or row.get("label") or ""
            note = (row.get("note") or row.get("reason") or "").strip()
            if not file_name:
                continue
            try:
                page = int(str(page_raw).strip())
            except Exception:
                continue

            lbl_s = str(lbl_raw).strip().lower()
            is_overlap = lbl_s in {"1", "true", "yes", "y", "overlap", "ov"}
            key = f"{file_name.lower()}::{page}"
            overrides[key] = {
                "is_overlap": is_overlap,
                "confidence": 1.0,
                "overlap_type": "manual_override",
                "signatures": [],
                "note": note or "imported training label",
                "updated_at": now_ts(),
            }
            added += 1
            if note and note not in notes:
                notes.append(note)
                notes_added += 1
    return added, notes_added


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(f"{APP_NAME} {APP_VERSION}")
        self.geometry("1200x820")
        self.minsize(1080, 760)

        self.storage = Storage()
        self.models = self.storage.load_models()
        self.memory = self.storage.load_memory()

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.cancel_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None

        self._build_ui()
        self._load_defaults()
        self.after(150, self._drain_logs)

    def _build_ui(self) -> None:
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=10, pady=8)

        ttk.Label(top, text="Model Profile").grid(row=0, column=0, sticky="w")
        self.model_name_var = tk.StringVar()
        self.model_combo = ttk.Combobox(top, textvariable=self.model_name_var, state="readonly", width=40)
        self.model_combo.grid(row=0, column=1, sticky="we", padx=6)
        self.model_combo.bind("<<ComboboxSelected>>", self.on_select_model)

        self.add_model_btn = ttk.Button(top, text="Add Model", command=self.add_model)
        self.add_model_btn.grid(row=0, column=2, padx=4)
        self.del_model_btn = ttk.Button(top, text="Delete Model", command=self.delete_model)
        self.del_model_btn.grid(row=0, column=3, padx=4)
        self.save_model_btn = ttk.Button(top, text="Save Model", command=self.save_model)
        self.save_model_btn.grid(row=0, column=4, padx=4)
        self.test_model_btn = ttk.Button(top, text="Test Model", command=self.test_model)
        self.test_model_btn.grid(row=0, column=5, padx=4)

        ttk.Label(top, text="Base URL").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.base_url_var = tk.StringVar()
        self.base_url_entry = ttk.Entry(top, textvariable=self.base_url_var, width=88)
        self.base_url_entry.grid(row=1, column=1, columnspan=5, sticky="we", padx=6, pady=(8, 0))

        ttk.Label(top, text="Model ID").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.model_id_var = tk.StringVar()
        self.model_id_entry = ttk.Entry(top, textvariable=self.model_id_var, width=35)
        self.model_id_entry.grid(row=2, column=1, sticky="w", padx=6, pady=(6, 0))

        ttk.Label(top, text="API Key").grid(row=2, column=2, sticky="e", pady=(6, 0))
        self.api_key_var = tk.StringVar()
        self.api_key_entry = ttk.Entry(top, textvariable=self.api_key_var, show="*", width=40)
        self.api_key_entry.grid(row=2, column=3, sticky="we", padx=6, pady=(6, 0))

        ttk.Label(top, text="Timeout(s)").grid(row=2, column=4, sticky="e", pady=(6, 0))
        self.timeout_var = tk.StringVar(value="120")
        self.timeout_entry = ttk.Entry(top, textvariable=self.timeout_var, width=8)
        self.timeout_entry.grid(row=2, column=5, sticky="w", padx=6, pady=(6, 0))

        top.columnconfigure(1, weight=1)
        top.columnconfigure(3, weight=1)

        scan_frame = ttk.LabelFrame(self, text="Scan + Actions")
        scan_frame.pack(fill=tk.X, padx=10, pady=8)

        ttk.Label(scan_frame, text="Source Directory").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        self.source_dir_var = tk.StringVar()
        self.source_dir_entry = ttk.Entry(scan_frame, textvariable=self.source_dir_var, width=90)
        self.source_dir_entry.grid(row=0, column=1, sticky="we", padx=6, pady=6)
        ttk.Button(scan_frame, text="Browse", command=self.pick_source_dir).grid(row=0, column=2, padx=6)

        ttk.Label(scan_frame, text="Replacement Directory (optional)").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        self.replacement_dir_var = tk.StringVar()
        self.replacement_dir_entry = ttk.Entry(scan_frame, textvariable=self.replacement_dir_var, width=90)
        self.replacement_dir_entry.grid(row=1, column=1, sticky="we", padx=6, pady=6)
        ttk.Button(scan_frame, text="Browse", command=self.pick_replacement_dir).grid(row=1, column=2, padx=6)

        ttk.Label(scan_frame, text="CSV Output Path").grid(row=2, column=0, sticky="w", padx=6, pady=6)
        self.csv_path_var = tk.StringVar()
        self.csv_path_entry = ttk.Entry(scan_frame, textvariable=self.csv_path_var, width=90)
        self.csv_path_entry.grid(row=2, column=1, sticky="we", padx=6, pady=6)
        ttk.Button(scan_frame, text="Pick", command=self.pick_csv_path).grid(row=2, column=2, padx=6)

        self.recursive_var = tk.BooleanVar(value=False)
        self.act_csv_var = tk.BooleanVar(value=True)
        self.act_identify_var = tk.BooleanVar(value=True)
        self.act_extract_var = tk.BooleanVar(value=True)
        self.act_replace_var = tk.BooleanVar(value=False)

        row3 = ttk.Frame(scan_frame)
        row3.grid(row=3, column=0, columnspan=3, sticky="w", padx=6, pady=4)
        ttk.Checkbutton(row3, text="Recursive scan", variable=self.recursive_var).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(row3, text="1) Generate overlap CSV", variable=self.act_csv_var).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(row3, text="2) Export O_*.pdf pages", variable=self.act_identify_var).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(row3, text="3) Export E_*.pdf (remove overlaps)", variable=self.act_extract_var).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(row3, text="4) Replace overlaps -> R_*.pdf", variable=self.act_replace_var).pack(side=tk.LEFT, padx=8)

        scan_frame.columnconfigure(1, weight=1)

        prompt_frame = ttk.LabelFrame(self, text="Custom Prompt (optional)")
        prompt_frame.pack(fill=tk.BOTH, padx=10, pady=8)
        self.prompt_text = ScrolledText(prompt_frame, height=7)
        self.prompt_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        mem_frame = ttk.LabelFrame(self, text="Memory / Training")
        mem_frame.pack(fill=tk.X, padx=10, pady=8)

        ttk.Button(mem_frame, text="Import Training CSV", command=self.import_training).grid(row=0, column=0, padx=6, pady=6)
        ttk.Button(mem_frame, text="Add Memory Note", command=self.add_memory_note).grid(row=0, column=1, padx=6, pady=6)
        ttk.Button(mem_frame, text="Save Memory", command=self.save_memory).grid(row=0, column=2, padx=6, pady=6)
        ttk.Button(mem_frame, text="Show Memory Stats", command=self.show_memory_stats).grid(row=0, column=3, padx=6, pady=6)

        self.memory_info_var = tk.StringVar(value="Memory: 0 notes, 0 overrides")
        ttk.Label(mem_frame, textvariable=self.memory_info_var).grid(row=0, column=4, padx=8, pady=6, sticky="w")

        run_frame = ttk.Frame(self)
        run_frame.pack(fill=tk.X, padx=10, pady=8)
        self.run_btn = ttk.Button(run_frame, text="Run", command=self.run_pipeline)
        self.run_btn.pack(side=tk.LEFT, padx=4)
        self.stop_btn = ttk.Button(run_frame, text="Stop", command=self.stop_pipeline)
        self.stop_btn.pack(side=tk.LEFT, padx=4)
        self.progress = ttk.Progressbar(run_frame, mode="determinate")
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        log_frame = ttk.LabelFrame(self, text="Log")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)
        self.log_text = ScrolledText(log_frame, height=14)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

    def _load_defaults(self) -> None:
        names = [m.name for m in self.models]
        self.model_combo["values"] = names
        if names:
            self.model_combo.current(0)
            self.on_select_model()

        default_dir = str(Path.cwd())
        self.source_dir_var.set(default_dir)
        self.csv_path_var.set(str(Path(default_dir) / f"overlap_report_{now_file_ts()}.csv"))
        self._refresh_memory_info()

    def _refresh_memory_info(self) -> None:
        notes = len(self.memory.get("global_notes", []))
        overrides = len(self.memory.get("overrides", {}))
        self.memory_info_var.set(f"Memory: {notes} notes, {overrides} overrides")

    def _drain_logs(self) -> None:
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_text.insert(tk.END, f"[{now_ts()}] {msg}\n")
                self.log_text.see(tk.END)
        except queue.Empty:
            pass
        self.after(150, self._drain_logs)

    def log(self, msg: str) -> None:
        self.log_queue.put(msg)

    def get_selected_model_index(self) -> Optional[int]:
        name = self.model_name_var.get().strip()
        for i, m in enumerate(self.models):
            if m.name == name:
                return i
        return None

    def on_select_model(self, *_args) -> None:
        idx = self.get_selected_model_index()
        if idx is None:
            return
        m = self.models[idx]
        self.base_url_var.set(m.base_url)
        self.model_id_var.set(m.model)
        self.api_key_var.set(m.api_key)
        self.timeout_var.set(str(m.timeout_sec))

    def add_model(self) -> None:
        name = simpledialog.askstring("Add Model", "Profile name:", parent=self)
        if not name:
            return
        if any(x.name == name for x in self.models):
            messagebox.showerror("Error", "Profile name already exists.")
            return
        base_url = simpledialog.askstring("Add Model", "Base URL:", parent=self)
        if not base_url:
            return
        model = simpledialog.askstring("Add Model", "Model ID:", parent=self)
        if not model:
            return
        api_key = simpledialog.askstring("Add Model", "API Key (optional):", parent=self, show="*") or ""
        timeout = simpledialog.askinteger("Add Model", "Timeout seconds:", initialvalue=120, parent=self)
        timeout = timeout or 120

        self.models.append(
            ModelProfile(name=name.strip(), base_url=base_url.strip(), model=model.strip(), api_key=api_key, timeout_sec=timeout)
        )
        self.storage.save_models(self.models)
        self.model_combo["values"] = [m.name for m in self.models]
        self.model_name_var.set(name.strip())
        self.on_select_model()
        self.log(f"Added model profile: {name}")

    def delete_model(self) -> None:
        idx = self.get_selected_model_index()
        if idx is None:
            return
        name = self.models[idx].name
        if not messagebox.askyesno("Confirm", f"Delete model profile '{name}'?"):
            return
        del self.models[idx]
        self.storage.save_models(self.models)
        names = [m.name for m in self.models]
        self.model_combo["values"] = names
        if names:
            self.model_combo.current(0)
            self.on_select_model()
        else:
            self.model_name_var.set("")
            self.base_url_var.set("")
            self.model_id_var.set("")
            self.api_key_var.set("")
            self.timeout_var.set("120")
        self.log(f"Deleted model profile: {name}")

    def save_model(self) -> None:
        idx = self.get_selected_model_index()
        if idx is None:
            messagebox.showerror("Error", "Select a model profile first.")
            return
        try:
            timeout = int(self.timeout_var.get().strip())
        except Exception:
            messagebox.showerror("Error", "Timeout must be integer.")
            return
        m = self.models[idx]
        m.base_url = self.base_url_var.get().strip()
        m.model = self.model_id_var.get().strip()
        m.api_key = self.api_key_var.get().strip()
        m.timeout_sec = max(10, timeout)
        self.storage.save_models(self.models)
        self.log(f"Saved model profile: {m.name}")
        messagebox.showinfo("Saved", "Model profile saved.")

    def current_profile(self) -> Optional[ModelProfile]:
        idx = self.get_selected_model_index()
        if idx is None:
            return None
        try:
            timeout = int(self.timeout_var.get().strip())
        except Exception:
            timeout = 120
        return ModelProfile(
            name=self.models[idx].name,
            base_url=self.base_url_var.get().strip(),
            model=self.model_id_var.get().strip(),
            api_key=self.api_key_var.get().strip(),
            timeout_sec=max(10, timeout),
        )

    def test_model(self) -> None:
        profile = self.current_profile()
        if not profile:
            messagebox.showerror("Error", "No model selected.")
            return
        if not profile.api_key:
            messagebox.showerror("Error", "API key is empty.")
            return

        def worker() -> None:
            self.log(f"Testing model: {profile.name} / {profile.model}")
            client = OpenAICompatibleClient(profile)
            ok, msg = client.quick_test()
            if ok:
                self.log(f"Model test OK: {msg}")
            else:
                self.log(f"Model test failed: {msg}")

        threading.Thread(target=worker, daemon=True).start()

    def pick_source_dir(self) -> None:
        d = filedialog.askdirectory(title="Select Source Directory")
        if d:
            self.source_dir_var.set(d)
            if not self.csv_path_var.get().strip():
                self.csv_path_var.set(str(Path(d) / f"overlap_report_{now_file_ts()}.csv"))

    def pick_replacement_dir(self) -> None:
        d = filedialog.askdirectory(title="Select Replacement Directory")
        if d:
            self.replacement_dir_var.set(d)

    def pick_csv_path(self) -> None:
        p = filedialog.asksaveasfilename(
            title="Select CSV Output Path",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if p:
            self.csv_path_var.set(p)

    def import_training(self) -> None:
        p = filedialog.askopenfilename(
            title="Select Training CSV",
            filetypes=[("CSV", "*.csv")],
        )
        if not p:
            return
        try:
            added, notes = import_training_csv(self.memory, Path(p))
            self.storage.save_memory(self.memory)
            self._refresh_memory_info()
            self.log(f"Imported training CSV: {added} overrides, {notes} notes added.")
            messagebox.showinfo("Training Imported", f"Overrides added: {added}\nNotes added: {notes}")
        except Exception as exc:
            messagebox.showerror("Import Error", str(exc))

    def add_memory_note(self) -> None:
        note = simpledialog.askstring(
            "Add Memory Note",
            "Enter a rule/note to improve future overlap detection:",
            parent=self,
        )
        if not note:
            return
        note = note.strip()
        if not note:
            return
        notes = self.memory.setdefault("global_notes", [])
        if note not in notes:
            notes.append(note)
            self.storage.save_memory(self.memory)
            self._refresh_memory_info()
            self.log("Added memory note.")

    def save_memory(self) -> None:
        self.storage.save_memory(self.memory)
        self._refresh_memory_info()
        self.log("Memory saved.")
        messagebox.showinfo("Saved", "Memory saved.")

    def show_memory_stats(self) -> None:
        notes = len(self.memory.get("global_notes", []))
        overrides = len(self.memory.get("overrides", {}))
        messagebox.showinfo("Memory Stats", f"Global notes: {notes}\nOverrides: {overrides}")

    def stop_pipeline(self) -> None:
        self.cancel_event.set()
        self.log("Stop requested.")

    def run_pipeline(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showwarning("Busy", "A pipeline is already running.")
            return

        profile = self.current_profile()
        if not profile:
            messagebox.showerror("Error", "No model selected.")
            return
        if not profile.api_key:
            messagebox.showerror("Error", "API key is empty.")
            return

        source_dir = Path(self.source_dir_var.get().strip())
        if not source_dir.exists() or not source_dir.is_dir():
            messagebox.showerror("Error", "Source directory does not exist.")
            return

        csv_path = Path(self.csv_path_var.get().strip()) if self.csv_path_var.get().strip() else source_dir / f"overlap_report_{now_file_ts()}.csv"
        replacement_dir = self.replacement_dir_var.get().strip()
        replacement_path = Path(replacement_dir) if replacement_dir else None

        custom_prompt = self.prompt_text.get("1.0", tk.END).strip()
        recursive = bool(self.recursive_var.get())

        self.cancel_event.clear()
        self.progress["value"] = 0

        def progress_cb(done: int, total: int) -> None:
            def _set() -> None:
                self.progress["maximum"] = total
                self.progress["value"] = done

            self.after(0, _set)

        def worker() -> None:
            try:
                self.log("Starting pipeline...")
                src_pdfs = list_pdfs(source_dir, recursive=recursive)
                if not src_pdfs:
                    self.log("No PDF files found in source directory.")
                    return
                self.log(f"Source PDFs: {len(src_pdfs)}")

                extra_pdfs: List[Path] = []
                if self.act_replace_var.get() and replacement_path and replacement_path.exists() and replacement_path.is_dir():
                    extra_pdfs = list_pdfs(replacement_path, recursive=recursive)
                    # avoid duplicates
                    src_set = {str(p.resolve()) for p in src_pdfs}
                    extra_pdfs = [p for p in extra_pdfs if str(p.resolve()) not in src_set]
                    if extra_pdfs:
                        self.log(f"Replacement candidate PDFs: {len(extra_pdfs)}")

                client = OpenAICompatibleClient(profile)
                engine = OverlapEngine(
                    client=client,
                    memory=self.memory,
                    logger=self.log,
                    cancel_event=self.cancel_event,
                    progress_cb=progress_cb,
                )

                source_records = engine.scan_pdfs(src_pdfs, scope="source", custom_prompt=custom_prompt)
                all_records = list(source_records)

                if extra_pdfs and not self.cancel_event.is_set():
                    self.log("Scanning replacement directory for candidate pages...")
                    extra_records = engine.scan_pdfs(extra_pdfs, scope="replacement", custom_prompt=custom_prompt)
                    all_records.extend(extra_records)

                self.storage.save_last_scan(all_records)
                self.log(f"Scan complete. Total pages processed: {len(all_records)}")

                if self.cancel_event.is_set():
                    self.log("Pipeline stopped before actions.")
                    return

                if self.act_csv_var.get():
                    count = write_overlap_csv(source_records, csv_path)
                    self.log(f"Action 1 done: CSV saved to {csv_path} (overlaps={count})")

                if self.act_identify_var.get():
                    cnt = export_overlap_pages(source_records, self.log)
                    self.log(f"Action 2 done: exported O_ pages = {cnt}")

                if self.act_extract_var.get():
                    cnt = export_extracted_non_overlap(source_records, self.log)
                    self.log(f"Action 3 done: created E_ files = {cnt}")

                if self.act_replace_var.get():
                    cnt, rep_csv = replace_overlap_pages(source_records, all_records, self.log)
                    self.log(f"Action 4 done: created R_ files = {cnt}")
                    self.log(f"Replacement report: {rep_csv}")

                self.log("Pipeline finished.")
            except Exception:
                self.log("Pipeline crashed:\n" + traceback.format_exc())
            finally:
                self.after(0, lambda: self.progress.configure(value=0))

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
