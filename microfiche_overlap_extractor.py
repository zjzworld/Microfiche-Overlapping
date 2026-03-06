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
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import fitz  # PyMuPDF
import requests
import tkinter as tk
from PIL import Image
from tkinter import filedialog, font as tkfont, messagebox, simpledialog, ttk
from tkinter.scrolledtext import ScrolledText

APP_NAME = "Microfiche Overlap Extractor"
APP_VERSION = "0.3.0"

DEFAULT_CLASSIFY_PROMPT = (
    "Task: classify each microfiche page as overlap, blurry, clean, or uncertain.\n"
    "Definition of overlap: two different record cards are merged/superimposed in one scan, "
    "including clear side-by-side merge OR ghost/partial superimposition.\n"
    "Typical clean transcript card content is often around 1000:447 in horizontal-to-vertical proportion "
    "(about 2.24:1). If the visible card area looks materially longer/wider than a normal transcript card, "
    "treat that as a strong overlap cue.\n"
    "When a page looks horizontally stretched or longer than normal, do an OCR-style verification mentally: "
    "read for duplicated/conflicting names, grades, headings, or two superimposed text layers before returning clean.\n"
    "A second strong overlap cue is page-boundary overflow: identify the main page/card boundary, especially the right edge. "
    "If the right boundary of the first/main page is visible but there is still text, numbers, headings, or another page/frame "
    "to the right of that boundary, that is strong evidence of overlap.\n"
    "Definition of blurry: text is so unreadable that student name and grades cannot be recognized at all. "
    "If any student name or any grades can be seen even partially, it is NOT blurry.\n"
    "You must choose exactly one decision class.\n"
    "Return strict JSON only with keys:\n"
    "- decision: one of [overlap, blurry, clean, uncertain]\n"
    "- is_overlap: boolean\n"
    "- is_blurry: boolean\n"
    "- confidence: number in [0,1]\n"
    "- overlap_type: one of [clear_double_card, ghost_superimposition, none]\n"
    "- signatures: array of 0..2 concise identity strings (e.g. NAME|DOB|STUDENTNO)\n"
    "- reason: short string\n"
)

# Display name -> candidate real model IDs.
# The GUI only shows display names. Request routing uses this internal mapping.
MODEL_NAME_TO_IDS: Dict[str, List[str]] = {
    "GPT-5.4": ["gpt-5.4", "GPT-5.4"],
    "GPT-5.3": ["gpt-5.3", "GPT-5.3"],
    "Claude-Opus-4.6": ["claude-opus-4-6", "claude-opus-4-5-20251101"],
    "Kimi-K2.5": ["kimi-k2.5"],
    "GLM-5": ["glm-5"],
    "MiniMax-M2.5": ["MiniMax-M2.5", "minimax-m2.5"],
}

UI_TOKENS: Dict[str, str] = {
    "canvas": "#F6F1EF",
    "canvas_soft": "#ECE8E6",
    "card": "#FBFAF8",
    "card_soft": "#F4F0EE",
    "card_strong": "#FFFFFF",
    "line": "#DAD4D0",
    "line_soft": "#E7E1DC",
    "ink": "#1D232B",
    "ink_soft": "#5C6671",
    "muted": "#7D8792",
    "accent": "#6D88A6",
    "accent_soft": "#E4ECF4",
    "rose_soft": "#F1DDE6",
    "ice_soft": "#DCEAF2",
    "sand_soft": "#EEE7DE",
    "run": "#617E99",
    "danger": "#B27570",
    "success": "#6C836D",
}


def resolve_model_candidates(display_model: str) -> List[str]:
    key = (display_model or "").strip()
    if not key:
        return []
    if key in MODEL_NAME_TO_IDS:
        return MODEL_NAME_TO_IDS[key][:]
    # case-insensitive fallback
    for k, ids in MODEL_NAME_TO_IDS.items():
        if k.lower() == key.lower():
            return ids[:]
    # custom model profile: use as-is
    return [key]


def normalize_display_model_name(name: str, model: str) -> str:
    n = (name or "").strip()
    m = (model or "").strip()
    # Exact match on display names.
    if n in MODEL_NAME_TO_IDS:
        return n
    if m in MODEL_NAME_TO_IDS:
        return m
    # Reverse mapping from actual model id to display name.
    for display, ids in MODEL_NAME_TO_IDS.items():
        for mid in ids:
            if mid.lower() == m.lower():
                return display
    # Legacy profile names created by earlier app versions.
    legacy = n.lower()
    if legacy.startswith("gpt-5.4"):
        return "GPT-5.4"
    if legacy.startswith("gpt-5.3") and "codex" not in legacy:
        return "GPT-5.3"
    if "claude-opus-4.6" in legacy or "claude-opus-4-6" in legacy:
        return "Claude-Opus-4.6"
    if "kimi-k2.5" in legacy:
        return "Kimi-K2.5"
    if legacy.startswith("glm-5"):
        return "GLM-5"
    if "minimax" in legacy and "2.5" in legacy:
        return "MiniMax-M2.5"
    return n or m


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
            changed = False
            for x in raw:
                raw_name = str(x.get("name", ""))
                raw_model = str(x.get("model", ""))
                raw_name_l = raw_name.strip().lower()
                raw_model_l = raw_model.strip().lower()
                if "codex" in raw_name_l or "codex" in raw_model_l:
                    changed = True
                    continue
                display_model = normalize_display_model_name(raw_name, raw_model)
                if display_model in {"GPT-5.2"}:
                    changed = True
                    continue
                if display_model and (raw_name != display_model or raw_model != display_model):
                    changed = True
                out.append(
                    ModelProfile(
                        name=display_model,
                        base_url=str(x.get("base_url", "")),
                        model=display_model,
                        api_key=str(x.get("api_key", "")),
                        timeout_sec=int(x.get("timeout_sec", 120)),
                    )
                )
            defaults_by_name = {m.name: m for m in self.default_models()}
            if not any(m.name == "GPT-5.4" for m in out):
                out.insert(0, defaults_by_name["GPT-5.4"])
                changed = True
            if not any(m.name == "GPT-5.3" for m in out):
                out.insert(1 if out else 0, defaults_by_name["GPT-5.3"])
                changed = True
            if changed and out:
                self.save_models(out)
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
            data = {"global_notes": [], "overrides": {}, "correction_history": []}
            self.save_memory(data)
            return data
        try:
            data = json.loads(self.memory_path.read_text(encoding="utf-8"))
            data.setdefault("global_notes", [])
            data.setdefault("overrides", {})
            data.setdefault("correction_history", [])
            return data
        except Exception:
            data = {"global_notes": [], "overrides": {}, "correction_history": []}
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
                name="GPT-5.4",
                base_url="https://ai.last.ee",
                model="GPT-5.4",
                api_key="sk-9b06f0ac4851ba8cdef2498ba269978ae5c64e099720b2a1b32d0d1b5f6631b4",
                timeout_sec=120,
            ),
            ModelProfile(
                name="GPT-5.3",
                base_url="https://ai.last.ee",
                model="GPT-5.3",
                api_key="sk-9b06f0ac4851ba8cdef2498ba269978ae5c64e099720b2a1b32d0d1b5f6631b4",
                timeout_sec=120,
            ),
            ModelProfile(
                name="Claude-Opus-4.6",
                base_url="https://cursor.scihub.edu.kg/api/v1",
                model="Claude-Opus-4.6",
                api_key="cr_56c958bfb141949f0a7e3ce7bf9e83315fe7695edf95749683c05b234c594000",
                timeout_sec=150,
            ),
            ModelProfile(
                name="Kimi-K2.5",
                base_url="https://coding.dashscope.aliyuncs.com/v1",
                model="Kimi-K2.5",
                api_key="sk-sp-a745d056ce96479c899d2b5d9c40d345",
                timeout_sec=120,
            ),
            ModelProfile(
                name="GLM-5",
                base_url="https://coding.dashscope.aliyuncs.com/v1",
                model="GLM-5",
                api_key="sk-sp-a745d056ce96479c899d2b5d9c40d345",
                timeout_sec=120,
            ),
            ModelProfile(
                name="MiniMax-M2.5",
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


def is_localish_base_url(base_url: str) -> bool:
    raw = (base_url or "").strip()
    if not raw:
        return False
    if "://" not in raw:
        raw = "http://" + raw
    try:
        parsed = urlparse(raw)
    except Exception:
        return False
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return False
    return host in {"127.0.0.1", "localhost", "0.0.0.0", "::1"} or host.endswith(".local")


def measure_page_visual_cues(image_jpeg: bytes) -> Dict[str, Any]:
    try:
        img = Image.open(io.BytesIO(image_jpeg)).convert("L")
        w, h = img.size
        mask = img.point(lambda px: 255 if px < 242 else 0, mode="L")
        bbox = mask.getbbox()
        cues: Dict[str, Any] = {
            "image_width": w,
            "image_height": h,
            "image_ratio": round(w / max(h, 1), 3),
        }
        if bbox:
            bw = max(1, bbox[2] - bbox[0])
            bh = max(1, bbox[3] - bbox[1])
            crop = mask.crop(bbox)
            crop_w, crop_h = crop.size
            cols = [0] * crop_w
            pixels = crop.load()
            for x in range(crop_w):
                dark = 0
                for y in range(crop_h):
                    if pixels[x, y]:
                        dark += 1
                cols[x] = dark
            search_start = max(0, int(crop_w * 0.50))
            search_end = max(search_start + 1, int(crop_w * 0.92))
            boundary_local_x = max(range(search_start, search_end), key=lambda i: cols[i])
            boundary_strength = cols[boundary_local_x] / max(crop_h, 1)
            outside_dark = 0
            outside_area = max(1, (crop_w - boundary_local_x - 1) * crop_h)
            outside_col_max = 0
            if boundary_local_x + 1 < crop_w:
                for x in range(boundary_local_x + 1, crop_w):
                    outside_col_max = max(outside_col_max, cols[x])
                    outside_dark += cols[x]
            cues.update(
                {
                    "content_width": bw,
                    "content_height": bh,
                    "content_ratio": round(bw / bh, 3),
                    "content_bbox": [int(x) for x in bbox],
                    "right_boundary_x": int(bbox[0] + boundary_local_x),
                    "right_boundary_strength": round(boundary_strength, 3),
                    "right_of_boundary_dark_ratio": round(outside_dark / outside_area, 4),
                    "right_of_boundary_colmax_ratio": round(outside_col_max / max(crop_h, 1), 4),
                }
            )
        return cues
    except Exception:
        return {}


OVERLAP_CSV_FIELDS = [
    "source_directory",
    "file_name",
    "file_path",
    "page",
    "decision",
    "is_overlap",
    "is_blurry",
    "confidence",
    "overlap_type",
    "signatures",
    "reason",
    "model",
    "resolved_model",
    "endpoint",
    "status",
    "error_detail",
]


def overlap_row_for_csv(rec: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(rec)
    row["signatures"] = " | ".join(rec.get("signatures", []))
    return {k: row.get(k, "") for k in OVERLAP_CSV_FIELDS}


def normalize_decision_fields(obj: Dict[str, Any]) -> Tuple[str, bool, bool]:
    decision = str(obj.get("decision", "")).strip().lower()
    is_overlap = bool(obj.get("is_overlap", False))
    is_blurry = bool(obj.get("is_blurry", False))

    if decision not in {"overlap", "blurry", "clean", "uncertain"}:
        if is_overlap:
            decision = "overlap"
        elif is_blurry:
            decision = "blurry"
        else:
            decision = "clean"

    if decision == "overlap":
        return decision, True, False
    if decision == "blurry":
        return decision, False, True
    if decision == "clean":
        return decision, False, False
    return "uncertain", False, False


def summarize_page_result(rec: Dict[str, Any]) -> str:
    return (
        f"{rec.get('file_name')} p{int(rec.get('page', 0)):03d}: "
        f"decision={rec.get('decision')} overlap={rec.get('is_overlap')} "
        f"blurry={rec.get('is_blurry')} conf={float(rec.get('confidence', 0.0)):.2f} "
        f"model={rec.get('resolved_model') or rec.get('model')} "
        f"endpoint={rec.get('endpoint', '')} "
        f"reason={str(rec.get('reason', ''))[:120]}"
    )


def ensure_memory_schema(memory: Dict[str, Any]) -> Dict[str, Any]:
    memory.setdefault("global_notes", [])
    memory.setdefault("overrides", {})
    memory.setdefault("correction_history", [])
    return memory


def flags_from_decision(decision: str) -> Tuple[bool, bool]:
    d = (decision or "").strip().lower()
    return d == "overlap", d == "blurry"


def correction_summary(entry: Dict[str, Any]) -> str:
    file_name = str(entry.get("file_name", "unknown.pdf"))
    page = int(entry.get("page", 0))
    previous = str(entry.get("previous_decision", "unknown"))
    corrected = str(entry.get("corrected_decision", "unknown"))
    note = str(entry.get("note", "")).strip()
    bits = [f"{file_name} p{page}: corrected {previous} -> {corrected}"]
    if note:
        bits.append(f"note={note}")
    sigs = [str(s).strip() for s in entry.get("signatures", []) if str(s).strip()]
    if sigs:
        bits.append("signatures=" + " | ".join(sigs[:2]))
    return "; ".join(bits)


def build_memory_notes(memory: Dict[str, Any], file_name: str) -> List[str]:
    ensure_memory_schema(memory)
    notes: List[str] = []
    seen: set[str] = set()

    def add(note: str) -> None:
        note = note.strip()
        if not note:
            return
        if note in seen:
            return
        seen.add(note)
        notes.append(note)

    for note in memory.get("global_notes", [])[:10]:
        add(str(note))

    target = (file_name or "").strip().lower()
    history = [x for x in memory.get("correction_history", []) if isinstance(x, dict)]
    same_file = [x for x in reversed(history) if str(x.get("file_name", "")).strip().lower() == target]
    recent = list(reversed(history))

    for entry in same_file[:8]:
        add("Same-file correction memory: " + correction_summary(entry))
    for entry in recent[:8]:
        add("Recent correction memory: " + correction_summary(entry))

    return notes[:18]


def remember_page_correction(memory: Dict[str, Any], rec: Dict[str, Any], corrected_decision: str, note: str) -> Dict[str, Any]:
    ensure_memory_schema(memory)
    corrected_decision = corrected_decision.strip().lower()
    if corrected_decision not in {"overlap", "blurry", "clean", "uncertain"}:
        raise ValueError("Corrected decision must be one of overlap, blurry, clean, uncertain.")

    file_name = str(rec.get("file_name", "")).strip()
    page_no = int(rec.get("page", 0))
    if not file_name or page_no <= 0:
        raise ValueError("Correction target is missing file_name or page.")

    is_overlap, is_blurry = flags_from_decision(corrected_decision)
    key = f"{file_name.lower()}::{page_no}"
    override = {
        "decision": corrected_decision,
        "is_overlap": is_overlap,
        "is_blurry": is_blurry,
        "confidence": 1.0,
        "overlap_type": "manual_override",
        "signatures": list(rec.get("signatures", []))[:2],
        "note": note.strip() or f"manual correction from {rec.get('decision', 'unknown')} to {corrected_decision}",
        "updated_at": now_ts(),
    }
    memory["overrides"][key] = override

    history = memory.setdefault("correction_history", [])
    history.append(
        {
            "file_name": file_name,
            "file_path": str(rec.get("file_path", "")),
            "page": page_no,
            "previous_decision": str(rec.get("decision", "unknown")),
            "corrected_decision": corrected_decision,
            "note": note.strip(),
            "signatures": list(rec.get("signatures", []))[:2],
            "overlap_type": str(rec.get("overlap_type", "none")),
            "updated_at": now_ts(),
        }
    )
    if len(history) > 300:
        del history[:-300]

    global_notes = memory.setdefault("global_notes", [])
    if note and note.strip() and note.strip() not in global_notes:
        global_notes.append(note.strip())
    auto_note = (
        f"If a microfiche page looks unusually wide or stretched compared with a normal transcript card, "
        f"do not mark it clean until OCR-style reading rules out overlap. Example memory: {file_name} p{page_no} -> {corrected_decision}."
    )
    if auto_note not in global_notes:
        global_notes.append(auto_note)

    return override


def find_last_scan_record(records: List[Dict[str, Any]], file_ref: str, page_no: int) -> Optional[Dict[str, Any]]:
    file_ref = file_ref.strip().lower()
    candidates = []
    for rec in records:
        rec_file = str(rec.get("file_name", "")).strip().lower()
        rec_path = str(rec.get("file_path", "")).strip().lower()
        rec_page = int(rec.get("page", 0))
        if rec_page != page_no:
            continue
        if file_ref in {rec_file, rec_path, Path(rec_path).stem.lower(), Path(rec_path).name.lower()}:
            return rec
        if file_ref and (file_ref in rec_file or file_ref in rec_path):
            candidates.append(rec)
    if len(candidates) == 1:
        return candidates[0]
    return None


class CorrectionPicker(tk.Toplevel):
    def __init__(self, parent: "App", records: List[Dict[str, Any]]) -> None:
        super().__init__(parent)
        self.parent = parent
        self.records = [r for r in records if r.get("scope") == "source"]
        self.filtered_records = list(self.records)
        self.title("Correct Last Scan Page")
        self.geometry("1080x720")
        self.minsize(940, 620)
        self.configure(bg=parent.ui["canvas"])
        self.transient(parent)
        self.grab_set()

        self.filter_var = tk.StringVar()
        self.corrected_var = tk.StringVar(value="overlap")
        self.status_var = tk.StringVar(value="Select a page, then save a correction into local memory.")

        shell = tk.Frame(self, bg=parent.ui["canvas"])
        shell.pack(fill=tk.BOTH, expand=True, padx=18, pady=18)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(1, weight=1)

        top = tk.Frame(shell, bg=parent.ui["card"])
        top.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        tk.Label(top, text="Correct Last Scan Page", font=parent.font_heading, fg=parent.ui["ink"], bg=parent.ui["card"]).grid(row=0, column=0, sticky="w", padx=16, pady=(14, 4))
        tk.Label(
            top,
            text="Filter by file name, page, decision, or reason. Corrections are stored locally and reused in later scans.",
            font=parent.font_caption,
            fg=parent.ui["muted"],
            bg=parent.ui["card"],
        ).grid(row=1, column=0, sticky="w", padx=16, pady=(0, 14))

        filter_row = tk.Frame(shell, bg=parent.ui["canvas"])
        filter_row.grid(row=1, column=0, sticky="nsew")
        filter_row.columnconfigure(0, weight=3)
        filter_row.columnconfigure(1, weight=2)
        filter_row.rowconfigure(0, weight=1)

        left = tk.Frame(filter_row, bg=parent.ui["card"], highlightbackground=parent.ui["line"], highlightthickness=1, bd=0)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)

        search_row = tk.Frame(left, bg=parent.ui["card"])
        search_row.grid(row=0, column=0, sticky="ew", padx=14, pady=14)
        ttk.Entry(search_row, textvariable=self.filter_var, style="Glass.TEntry").pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(search_row, text="Filter", style="Glass.TButton", command=self.refresh).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(search_row, text="Reset", style="Glass.TButton", command=self.reset_filter).pack(side=tk.LEFT, padx=(8, 0))
        self.filter_var.trace_add("write", lambda *_args: self.refresh())

        columns = ("file_name", "page", "decision", "confidence", "reason")
        self.tree = ttk.Treeview(left, columns=columns, show="headings", height=18)
        self.tree.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))
        self.tree.heading("file_name", text="File")
        self.tree.heading("page", text="Page")
        self.tree.heading("decision", text="Current")
        self.tree.heading("confidence", text="Conf")
        self.tree.heading("reason", text="Reason")
        self.tree.column("file_name", width=240, anchor="w")
        self.tree.column("page", width=64, anchor="center")
        self.tree.column("decision", width=92, anchor="center")
        self.tree.column("confidence", width=60, anchor="center")
        self.tree.column("reason", width=360, anchor="w")
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self.on_select())
        self.tree.bind("<Double-1>", lambda _e: self.save())

        right = tk.Frame(filter_row, bg=parent.ui["card"], highlightbackground=parent.ui["line"], highlightthickness=1, bd=0)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(4, weight=1)

        tk.Label(right, text="Selected Page", font=parent.font_heading, fg=parent.ui["ink"], bg=parent.ui["card"]).grid(row=0, column=0, sticky="w", padx=16, pady=(14, 8))
        self.selection_label = tk.Label(
            right,
            text="No page selected",
            font=parent.font_status,
            fg=parent.ui["ink_soft"],
            bg=parent.ui["card"],
            justify="left",
            wraplength=320,
        )
        self.selection_label.grid(row=1, column=0, sticky="w", padx=16)

        label_row = tk.Frame(right, bg=parent.ui["card"])
        label_row.grid(row=2, column=0, sticky="w", padx=16, pady=(14, 10))
        tk.Label(label_row, text="Corrected Label", font=parent.font_label, fg=parent.ui["ink_soft"], bg=parent.ui["card"]).pack(anchor="w")
        for val in ("overlap", "blurry", "clean", "uncertain"):
            tk.Radiobutton(
                label_row,
                text=val,
                value=val,
                variable=self.corrected_var,
                bg=parent.ui["card"],
                activebackground=parent.ui["card"],
                highlightthickness=0,
                fg=parent.ui["ink"],
                selectcolor=parent.ui["card_strong"],
            ).pack(anchor="w")

        tk.Label(right, text="Memory Note", font=parent.font_label, fg=parent.ui["ink_soft"], bg=parent.ui["card"]).grid(row=3, column=0, sticky="w", padx=16)
        self.note_text = ScrolledText(right, height=10)
        self.note_text.grid(row=4, column=0, sticky="nsew", padx=16, pady=(6, 10))
        parent._style_text_widget(self.note_text)

        bottom = tk.Frame(right, bg=parent.ui["card"])
        bottom.grid(row=5, column=0, sticky="ew", padx=16, pady=(0, 14))
        ttk.Button(bottom, text="Save Correction", style="Accent.TButton", command=self.save).pack(side=tk.LEFT)
        ttk.Button(bottom, text="Close", style="Glass.TButton", command=self.destroy).pack(side=tk.LEFT, padx=(8, 0))
        tk.Label(bottom, textvariable=self.status_var, font=parent.font_caption, fg=parent.ui["muted"], bg=parent.ui["card"]).pack(side=tk.RIGHT)

        self.refresh()

    def reset_filter(self) -> None:
        self.filter_var.set("")
        self.refresh()

    def refresh(self) -> None:
        query = self.filter_var.get().strip().lower()
        if query:
            self.filtered_records = []
            for rec in self.records:
                hay = " ".join(
                    [
                        str(rec.get("file_name", "")),
                        str(rec.get("file_path", "")),
                        str(rec.get("page", "")),
                        str(rec.get("decision", "")),
                        str(rec.get("reason", "")),
                    ]
                ).lower()
                if query in hay:
                    self.filtered_records.append(rec)
        else:
            self.filtered_records = list(self.records)

        for item in self.tree.get_children():
            self.tree.delete(item)
        for idx, rec in enumerate(self.filtered_records):
            self.tree.insert(
                "",
                tk.END,
                iid=str(idx),
                values=(
                    rec.get("file_name", ""),
                    int(rec.get("page", 0)),
                    rec.get("decision", ""),
                    f"{float(rec.get('confidence', 0.0)):.2f}",
                    str(rec.get("reason", ""))[:80],
                ),
            )
        self.status_var.set(f"{len(self.filtered_records)} pages shown")

    def selected_record(self) -> Optional[Dict[str, Any]]:
        sel = self.tree.selection()
        if not sel:
            return None
        idx = int(sel[0])
        if 0 <= idx < len(self.filtered_records):
            return self.filtered_records[idx]
        return None

    def on_select(self) -> None:
        rec = self.selected_record()
        if not rec:
            self.selection_label.configure(text="No page selected")
            return
        self.selection_label.configure(
            text=(
                f"{rec.get('file_name')} p{int(rec.get('page', 0)):03d}\n"
                f"Current: {rec.get('decision')} | Conf: {float(rec.get('confidence', 0.0)):.2f}\n"
                f"Endpoint: {rec.get('endpoint', '') or '-'}"
            )
        )
        self.corrected_var.set(str(rec.get("decision", "clean")))
        self.note_text.delete("1.0", tk.END)
        self.note_text.insert("1.0", str(rec.get("reason", "")))

    def save(self) -> None:
        rec = self.selected_record()
        if not rec:
            messagebox.showerror("No Selection", "Select a page first.", parent=self)
            return
        corrected = self.corrected_var.get().strip().lower()
        note = self.note_text.get("1.0", tk.END).strip()
        try:
            remember_page_correction(self.parent.memory, rec, corrected, note)
            self.parent.storage.save_memory(self.parent.memory)
            self.parent._refresh_memory_info()
            self.parent.log(
                f"Stored correction memory for {rec.get('file_name')} p{int(rec.get('page', 0)):03d}: "
                f"{rec.get('decision')} -> {corrected}"
            )
            self.status_var.set("Correction saved to local memory")
            messagebox.showinfo(
                "Correction Saved",
                f"Saved correction for {rec.get('file_name')} p{int(rec.get('page', 0))}: "
                f"{rec.get('decision')} -> {corrected}",
                parent=self,
            )
        except Exception as exc:
            messagebox.showerror("Correction Error", str(exc), parent=self)


class OpenAICompatibleClient:
    def __init__(self, profile: ModelProfile):
        self.profile = profile
        alias = profile.model.strip() or profile.name.strip()
        self.model_candidates = resolve_model_candidates(alias)
        base = (profile.base_url or "").lower()
        self.responses_only = "ai.last.ee" in base

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.profile.api_key.strip():
            headers["Authorization"] = f"Bearer {self.profile.api_key}"
        return headers

    def _post_json(self, endpoint: str, payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any], str]:
        base = self.profile.base_url.rstrip("/")
        parsed = urlparse(base)
        if parsed.path.rstrip("/") == "":
            url = base + "/v1" + endpoint
        else:
            url = base + endpoint
        headers = self._headers()
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

    def _post_chat(self, payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any], str]:
        return self._post_json("/chat/completions", payload)

    def _post_responses(self, payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any], str]:
        return self._post_json("/responses", payload)

    @staticmethod
    def _extract_chat_text(obj: Dict[str, Any]) -> str:
        msg = obj.get("choices", [{}])[0].get("message", {}).get("content", "")
        if isinstance(msg, str):
            return msg
        if isinstance(msg, list):
            parts = []
            for item in msg:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            return "\n".join([p for p in parts if p])
        return ""

    @staticmethod
    def _extract_responses_text(obj: Dict[str, Any]) -> str:
        direct = obj.get("output_text")
        if isinstance(direct, str) and direct.strip():
            return direct
        parts: List[str] = []
        for item in obj.get("output", []):
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []):
                if not isinstance(content, dict):
                    continue
                text = content.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text)
        return "\n".join(parts).strip()

    def _try_responses(self, model_id: str, prompt: str, b64: str, max_output_tokens: int) -> Tuple[bool, Dict[str, Any]]:
        payload = {
            "model": model_id,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"},
                    ],
                }
            ],
            "max_output_tokens": max_output_tokens,
        }
        status, obj, raw = self._post_responses(payload)
        if status == 200 and obj:
            msg = self._extract_responses_text(obj)
            parsed = parse_json_object(msg)
            if parsed:
                return True, {
                    "ok": True,
                    "status": status,
                    "raw": msg,
                    "json": parsed,
                    "usage": obj.get("usage", {}),
                    "resolved_model": model_id,
                    "resolved_endpoint": "responses",
                }
            return False, {"error": f"{model_id}: responses non-json", "raw": raw[:240], "status": status}
        return False, {"error": f"{model_id}: responses status={status} raw={raw[:240]}", "raw": raw[:240], "status": status}

    def _try_chat(self, model_id: str, prompt: str, b64: str, max_output_tokens: int) -> Tuple[bool, Dict[str, Any]]:
        headers = {
            "Authorization": f"Bearer {self.profile.api_key}",
            "Content-Type": "application/json",
        }
        _ = headers
        payload_openai = {
            "model": model_id,
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
            "max_tokens": max_output_tokens,
        }
        status, obj, raw = self._post_chat(payload_openai)
        if status == 200 and obj:
            msg = self._extract_chat_text(obj)
            parsed = parse_json_object(msg)
            if parsed:
                return True, {
                    "ok": True,
                    "status": status,
                    "raw": msg,
                    "json": parsed,
                    "usage": obj.get("usage", {}),
                    "resolved_model": model_id,
                    "resolved_endpoint": "chat_completions",
                }
        payload_alt = {
            "model": model_id,
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
            "max_tokens": max_output_tokens,
        }
        status2, obj2, raw2 = self._post_chat(payload_alt)
        if status2 == 200 and obj2:
            msg2 = self._extract_chat_text(obj2)
            parsed2 = parse_json_object(msg2)
            if parsed2:
                return True, {
                    "ok": True,
                    "status": status2,
                    "raw": msg2,
                    "json": parsed2,
                    "usage": obj2.get("usage", {}),
                    "resolved_model": model_id,
                    "resolved_endpoint": "chat_completions_alt",
                }
        err = f"{model_id}: chat status={status} raw={raw[:160]} || alt status={status2} raw={raw2[:160]}"
        return False, {"error": err, "status": status2 if status2 != 200 else status, "raw": raw2[:240] or raw[:240]}

    def classify_page(
        self,
        image_jpeg: bytes,
        file_name: str,
        page_no: int,
        custom_prompt: str,
        memory_notes: List[str],
        page_cues: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        b64 = base64.b64encode(image_jpeg).decode("ascii")
        rules = ""
        if memory_notes:
            joined = "\n".join([f"- {x}" for x in memory_notes[:20]])
            rules = f"\nLearned rules from prior corrections:\n{joined}\n"
        cues = ""
        if page_cues:
            cue_lines = [f"- rendered_size={page_cues.get('image_width')}x{page_cues.get('image_height')}"]
            if page_cues.get("content_width") and page_cues.get("content_height"):
                cue_lines.append(
                    f"- content_bbox_size={page_cues.get('content_width')}x{page_cues.get('content_height')} "
                    f"(ratio={page_cues.get('content_ratio')})"
                )
            if page_cues.get("right_boundary_x") is not None:
                cue_lines.append(
                    f"- estimated_right_boundary_x={page_cues.get('right_boundary_x')} "
                    f"strength={page_cues.get('right_boundary_strength')}"
                )
                cue_lines.append(
                    f"- right_of_boundary_dark_ratio={page_cues.get('right_of_boundary_dark_ratio')} "
                    f"right_of_boundary_colmax_ratio={page_cues.get('right_of_boundary_colmax_ratio')}"
                )
            cue_lines.append(
                "- heuristic_hint=normal clean card is often around 1000:447 (~2.24:1); materially wider/longer content is a strong overlap cue"
            )
            cue_lines.append(
                "- heuristic_hint=if the main right page boundary is identifiable but there is still visible structure or text beyond it, treat that as strong overlap evidence"
            )
            cues = "\nMeasured visual cues:\n" + "\n".join(cue_lines) + "\n"
        prompt = (
            DEFAULT_CLASSIFY_PROMPT
            + rules
            + cues
            + f"\nfile={file_name}\npage={page_no}\n"
        )
        if custom_prompt.strip():
            prompt += f"\nCustom instructions:\n{custom_prompt.strip()}\n"

        errors: List[str] = []
        for model_id in self.model_candidates:
            ok, result = self._try_responses(model_id, prompt, b64, 320)
            if ok:
                return result
            errors.append(str(result.get("error", ""))[:220])
            if self.responses_only:
                continue

            ok, result = self._try_chat(model_id, prompt, b64, 320)
            if ok:
                return result
            errors.append(str(result.get("error", ""))[:220])

        return {
            "ok": False,
            "status": -1,
            "raw": "",
            "error": "Image classification failed. " + " || ".join(errors[:4]),
        }

    def quick_test(self) -> Tuple[bool, str]:
        tiny_img = Image.new("RGB", (12, 12), color="white")
        bio = io.BytesIO()
        tiny_img.save(bio, format="JPEG", quality=70)
        b64 = base64.b64encode(bio.getvalue()).decode("ascii")
        errs = []
        for model_id in self.model_candidates:
            payload = {
                "model": model_id,
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Reply exactly OK."},
                            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"},
                        ],
                    }
                ],
                "max_output_tokens": 12,
            }
            status, obj, raw = self._post_responses(payload)
            if status == 200 and obj:
                msg = self._extract_responses_text(obj)
                return True, f"HTTP 200 via '{model_id}' on responses. Reply: {msg[:120]!r}"
            errs.append(f"{model_id}: responses {status} {raw[:80]}")
            if self.responses_only:
                continue

            ok, result = self._try_chat(model_id, "Reply exactly OK.", b64, 12)
            if ok:
                return True, f"HTTP 200 via '{model_id}' on {result.get('resolved_endpoint')}. Reply: {str(result.get('raw', ''))[:120]!r}"
            errs.append(f"{model_id}: chat fallback failed")
        return False, " ; ".join(errs[:3])


class OverlapEngine:
    def __init__(
        self,
        client: OpenAICompatibleClient,
        memory: Dict[str, Any],
        logger,
        cancel_event: threading.Event,
        progress_cb,
        render_dpi: int = 220,
    ) -> None:
        self.client = client
        self.memory = ensure_memory_schema(memory)
        self.log = logger
        self.cancel_event = cancel_event
        self.progress_cb = progress_cb
        self.render_dpi = max(120, min(360, int(render_dpi)))

    def memory_override(self, file_name: str, page_no: int) -> Optional[Dict[str, Any]]:
        key = f"{file_name.lower()}::{page_no}"
        return self.memory.get("overrides", {}).get(key)

    def scan_pdfs(
        self,
        pdf_paths: List[Path],
        scope: str,
        custom_prompt: str,
        on_page_result: Optional[Callable[[Dict[str, Any], Path, fitz.Document], None]] = None,
        on_file_done: Optional[Callable[[Path, fitz.Document, List[Dict[str, Any]]], None]] = None,
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
            file_records: List[Dict[str, Any]] = []

            for idx in range(len(doc)):
                if self.cancel_event.is_set():
                    self.log("Scan cancelled.")
                    break
                page_no = idx + 1
                file_name = pdf_path.name
                override = self.memory_override(file_name, page_no)
                if override:
                    override_decision = str(override.get("decision", "")).strip().lower()
                    if override_decision not in {"overlap", "blurry", "clean", "uncertain"}:
                        override_overlap = bool(override.get("is_overlap", False))
                        override_blurry = bool(override.get("is_blurry", False))
                        override_decision = "overlap" if override_overlap else ("blurry" if override_blurry else "clean")
                    override_overlap = bool(override.get("is_overlap", override_decision == "overlap"))
                    override_blurry = bool(override.get("is_blurry", override_decision == "blurry"))
                    rec = {
                        "source_directory": str(pdf_path.parent),
                        "file_name": file_name,
                        "file_path": str(pdf_path),
                        "page": page_no,
                        "decision": override_decision,
                        "is_overlap": override_overlap,
                        "is_blurry": override_blurry,
                        "confidence": float(override.get("confidence", 1.0)),
                        "overlap_type": str(override.get("overlap_type", "manual_override")),
                        "signatures": override.get("signatures", []),
                        "reason": str(override.get("note", "manual memory override")),
                        "scope": scope,
                        "model": self.client.profile.model,
                        "resolved_model": "memory_override",
                        "endpoint": "memory_override",
                        "status": "memory_override",
                        "error_detail": "",
                    }
                    records.append(rec)
                    file_records.append(rec)
                    self.log("Page result (memory): " + summarize_page_result(rec))
                    if on_page_result:
                        try:
                            on_page_result(rec, pdf_path, doc)
                        except Exception as cb_exc:
                            self.log(f"on_page_result callback failed: {cb_exc}")
                    done += 1
                    self.progress_cb(done, max(total_pages, 1))
                    continue

                try:
                    image_jpeg = render_page_jpeg(doc[idx], dpi=self.render_dpi)
                    page_cues = measure_page_visual_cues(image_jpeg)
                except Exception as exc:
                    self.log(f"Render failed {pdf_path} p{page_no}: {exc}")
                    rec = {
                        "source_directory": str(pdf_path.parent),
                        "file_name": file_name,
                        "file_path": str(pdf_path),
                        "page": page_no,
                        "decision": "uncertain",
                        "is_overlap": False,
                        "is_blurry": False,
                        "confidence": 0.0,
                        "overlap_type": "none",
                        "signatures": [],
                        "reason": f"render_error: {exc}",
                        "scope": scope,
                        "model": self.client.profile.model,
                        "resolved_model": "",
                        "endpoint": "",
                        "status": "error",
                        "error_detail": f"render_error: {exc}",
                    }
                    records.append(rec)
                    file_records.append(rec)
                    self.log("Page result (error): " + summarize_page_result(rec))
                    if on_page_result:
                        try:
                            on_page_result(rec, pdf_path, doc)
                        except Exception as cb_exc:
                            self.log(f"on_page_result callback failed: {cb_exc}")
                    done += 1
                    self.progress_cb(done, max(total_pages, 1))
                    continue

                result = self.client.classify_page(
                    image_jpeg=image_jpeg,
                    file_name=file_name,
                    page_no=page_no,
                    custom_prompt=custom_prompt,
                    memory_notes=build_memory_notes(self.memory, file_name),
                    page_cues=page_cues,
                )

                if not result.get("ok"):
                    rec = {
                        "source_directory": str(pdf_path.parent),
                        "file_name": file_name,
                        "file_path": str(pdf_path),
                        "page": page_no,
                        "decision": "uncertain",
                        "is_overlap": False,
                        "is_blurry": False,
                        "confidence": 0.0,
                        "overlap_type": "none",
                        "signatures": [],
                        "reason": "llm_error",
                        "scope": scope,
                        "model": self.client.profile.model,
                        "resolved_model": result.get("resolved_model", ""),
                        "endpoint": result.get("resolved_endpoint", ""),
                        "status": "llm_error",
                        "error_detail": f"status={result.get('status')} err={result.get('error', 'unknown')} raw={str(result.get('raw', ''))[:240]}",
                    }
                    self.log(
                        f"LLM failed {file_name} p{page_no}: "
                        f"{rec['error_detail']}"
                    )
                else:
                    obj = result.get("json", {})
                    sigs = obj.get("signatures", [])
                    if not isinstance(sigs, list):
                        sigs = []
                    sigs = [norm_sig(str(s)) for s in sigs if str(s).strip()][:2]
                    decision, is_overlap, is_blurry = normalize_decision_fields(obj)
                    rec = {
                        "source_directory": str(pdf_path.parent),
                        "file_name": file_name,
                        "file_path": str(pdf_path),
                        "page": page_no,
                        "decision": decision,
                        "is_overlap": is_overlap,
                        "is_blurry": is_blurry,
                        "confidence": float(obj.get("confidence", 0.0)),
                        "overlap_type": str(obj.get("overlap_type", "none")),
                        "signatures": sigs,
                        "reason": str(obj.get("reason", ""))[:500],
                        "scope": scope,
                        "model": self.client.profile.model,
                        "resolved_model": result.get("resolved_model", ""),
                        "endpoint": result.get("resolved_endpoint", ""),
                        "status": "ok",
                        "error_detail": "",
                    }
                    if rec.get("decision") == "uncertain":
                        self.log("Page result (uncertain): " + summarize_page_result(rec))
                    else:
                        self.log("Page result: " + summarize_page_result(rec))
                records.append(rec)
                file_records.append(rec)
                if on_page_result:
                    try:
                        on_page_result(rec, pdf_path, doc)
                    except Exception as cb_exc:
                        self.log(f"on_page_result callback failed: {cb_exc}")
                done += 1
                self.progress_cb(done, max(total_pages, 1))

            if on_file_done:
                try:
                    on_file_done(pdf_path, doc, file_records)
                except Exception as cb_exc:
                    self.log(f"on_file_done callback failed: {cb_exc}")
            doc.close()

        return records


def write_overlap_csv(records: List[Dict[str, Any]], out_csv: Path) -> int:
    overlaps = [r for r in records if r.get("is_overlap") and r.get("scope") == "source"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=OVERLAP_CSV_FIELDS)
        w.writeheader()
        for r in overlaps:
            w.writerow(overlap_row_for_csv(r))
    return len(overlaps)


def export_single_tagged_page_from_doc(
    doc: fitz.Document, src_path: Path, page: int, prefix: str, logger
) -> bool:
    out = src_path.parent / f"{prefix}_{src_path.stem}_P{page}.pdf"
    try:
        one = fitz.open()
        one.insert_pdf(doc, from_page=page - 1, to_page=page - 1)
        one.save(str(out))
        one.close()
        return True
    except Exception as exc:
        logger(f"Export {prefix}_ page failed {src_path.name} p{page}: {exc}")
        return False


def export_single_overlap_page_from_doc(
    doc: fitz.Document, src_path: Path, page: int, logger
) -> bool:
    return export_single_tagged_page_from_doc(doc, src_path, page, "O", logger)


def export_single_blurry_page_from_doc(
    doc: fitz.Document, src_path: Path, page: int, logger
) -> bool:
    return export_single_tagged_page_from_doc(doc, src_path, page, "B", logger)


def export_extracted_non_overlap_for_file(
    doc: fitz.Document, src_path: Path, file_records: List[Dict[str, Any]], logger
) -> bool:
    marks: Dict[int, bool] = {}
    for r in file_records:
        if r.get("scope") != "source":
            continue
        marks[int(r["page"])] = bool(r.get("is_overlap"))

    keep_pages = [p for p, is_ov in sorted(marks.items()) if not is_ov]
    if not keep_pages:
        logger(f"No non-overlap pages for {src_path.name}, skip E_ output.")
        return False

    out = src_path.parent / f"E_{src_path.name}"
    try:
        out_doc = fitz.open()
        for p in keep_pages:
            out_doc.insert_pdf(doc, from_page=p - 1, to_page=p - 1)
        out_doc.save(str(out))
        out_doc.close()
        return True
    except Exception as exc:
        logger(f"Create E_ file failed {src_path.name}: {exc}")
        return False


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


def export_blurry_pages(records: List[Dict[str, Any]], logger) -> int:
    targets = [r for r in records if r.get("scope") == "source" and r.get("is_blurry")]
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
            if export_single_blurry_page_from_doc(doc, src, page, logger):
                created += 1
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
        if c.get("is_overlap") or c.get("is_blurry"):
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

    candidates = [r for r in all_records if not r.get("is_overlap") and not r.get("is_blurry")]

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
    ensure_memory_schema(memory)
    added = 0
    notes_added = 0
    overrides = memory.setdefault("overrides", {})
    notes = memory.setdefault("global_notes", [])
    history = memory.setdefault("correction_history", [])
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
            is_blurry = lbl_s in {"blurry", "blur", "unreadable"}
            decision = "overlap" if is_overlap else ("blurry" if is_blurry else "clean")
            key = f"{file_name.lower()}::{page}"
            overrides[key] = {
                "decision": decision,
                "is_overlap": is_overlap,
                "is_blurry": is_blurry,
                "confidence": 1.0,
                "overlap_type": "manual_override",
                "signatures": [],
                "note": note or "imported training label",
                "updated_at": now_ts(),
            }
            history.append(
                {
                    "file_name": file_name,
                    "file_path": str(row.get("file_path", "")),
                    "page": page,
                    "previous_decision": "unknown",
                    "corrected_decision": decision,
                    "note": note or "imported training label",
                    "signatures": [],
                    "overlap_type": "manual_override",
                    "updated_at": now_ts(),
                }
            )
            added += 1
            if note and note not in notes:
                notes.append(note)
                notes_added += 1
    if len(history) > 300:
        del history[:-300]
    return added, notes_added


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.ui = UI_TOKENS.copy()
        self.title(f"{APP_NAME} {APP_VERSION}")
        self.geometry("1320x860")
        self.minsize(1160, 780)
        self.configure(bg=self.ui["canvas"])

        self.storage = Storage()
        self.models = self.storage.load_models()
        self.memory = ensure_memory_schema(self.storage.load_memory())

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.cancel_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None
        self._bg_after_id: Optional[str] = None

        self._setup_fonts()
        self._configure_styles()
        self._build_backdrop()

        self._build_ui()
        self._load_defaults()
        self.bind("<Configure>", self._on_root_configure)
        self.after(60, self._draw_backdrop)
        self.after(150, self._drain_logs)

    def _setup_fonts(self) -> None:
        available = {name.lower(): name for name in tkfont.families()}

        def pick(*candidates: str) -> str:
            for candidate in candidates:
                if candidate.lower() in available:
                    return available[candidate.lower()]
            return "TkDefaultFont"

        self.font_ui = tkfont.Font(family=pick("Aptos", "Segoe UI Variable Text", "Segoe UI", "Helvetica Neue"), size=10)
        self.font_small = tkfont.Font(family=self.font_ui.actual("family"), size=9)
        self.font_label = tkfont.Font(family=self.font_ui.actual("family"), size=9, weight="bold")
        self.font_title = tkfont.Font(family=pick("Aptos Display", "Segoe UI Variable Display", "Bahnschrift", "Helvetica Neue"), size=23, weight="normal")
        self.font_heading = tkfont.Font(family=self.font_ui.actual("family"), size=12, weight="bold")
        self.font_overline = tkfont.Font(family=self.font_ui.actual("family"), size=9, weight="bold")
        self.font_status = tkfont.Font(family=self.font_ui.actual("family"), size=10)
        self.font_caption = tkfont.Font(family=self.font_ui.actual("family"), size=9)

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        self.option_add("*Font", self.font_ui)
        self.option_add("*TCombobox*Listbox.font", self.font_ui)
        style.configure(".", background=self.ui["canvas"], foreground=self.ui["ink"])
        style.configure(
            "Glass.TEntry",
            fieldbackground=self.ui["card_strong"],
            background=self.ui["card_strong"],
            foreground=self.ui["ink"],
            bordercolor=self.ui["line"],
            lightcolor=self.ui["line"],
            darkcolor=self.ui["line"],
            insertcolor=self.ui["ink"],
            padding=(10, 8),
            relief="flat",
        )
        style.configure(
            "Glass.TCombobox",
            fieldbackground=self.ui["card_strong"],
            background=self.ui["card_strong"],
            foreground=self.ui["ink"],
            bordercolor=self.ui["line"],
            lightcolor=self.ui["line"],
            darkcolor=self.ui["line"],
            arrowcolor=self.ui["ink_soft"],
            padding=(10, 8),
            relief="flat",
        )
        style.map(
            "Glass.TCombobox",
            fieldbackground=[("readonly", self.ui["card_strong"])],
            background=[("readonly", self.ui["card_strong"])],
            foreground=[("readonly", self.ui["ink"])],
        )
        style.configure(
            "Glass.TButton",
            background=self.ui["card_soft"],
            foreground=self.ui["ink"],
            bordercolor=self.ui["line"],
            lightcolor=self.ui["line"],
            darkcolor=self.ui["line"],
            focuscolor=self.ui["card_soft"],
            padding=(12, 8),
            relief="flat",
        )
        style.map(
            "Glass.TButton",
            background=[("active", self.ui["card_strong"]), ("pressed", self.ui["card_soft"])],
            bordercolor=[("active", self.ui["accent"])],
        )
        style.configure(
            "Accent.TButton",
            background=self.ui["run"],
            foreground=self.ui["card_strong"],
            bordercolor=self.ui["run"],
            lightcolor=self.ui["run"],
            darkcolor=self.ui["run"],
            focuscolor=self.ui["run"],
            padding=(14, 9),
            relief="flat",
        )
        style.map(
            "Accent.TButton",
            background=[("active", self.ui["accent"]), ("pressed", self.ui["run"])],
            foreground=[("active", self.ui["card_strong"])],
        )
        style.configure(
            "Danger.TButton",
            background=self.ui["danger"],
            foreground=self.ui["card_strong"],
            bordercolor=self.ui["danger"],
            lightcolor=self.ui["danger"],
            darkcolor=self.ui["danger"],
            focuscolor=self.ui["danger"],
            padding=(14, 9),
            relief="flat",
        )
        style.map(
            "Danger.TButton",
            background=[("active", "#C0857F"), ("pressed", self.ui["danger"])],
            foreground=[("active", self.ui["card_strong"])],
        )
        style.configure(
            "Glass.Horizontal.TProgressbar",
            troughcolor=self.ui["card_soft"],
            background=self.ui["accent"],
            bordercolor=self.ui["line"],
            lightcolor=self.ui["accent"],
            darkcolor=self.ui["accent"],
            thickness=10,
        )

    def _build_backdrop(self) -> None:
        self.backdrop = tk.Canvas(self, bg=self.ui["canvas"], highlightthickness=0, bd=0)
        self.backdrop.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.backdrop.lower()

    def _on_root_configure(self, event: tk.Event) -> None:
        if event.widget is not self:
            return
        if self._bg_after_id:
            try:
                self.after_cancel(self._bg_after_id)
            except Exception:
                pass
        self._bg_after_id = self.after(20, self._draw_backdrop)

    def _draw_backdrop(self) -> None:
        self._bg_after_id = None
        w = max(self.winfo_width(), 1)
        h = max(self.winfo_height(), 1)
        self.backdrop.delete("all")
        self.backdrop.create_rectangle(0, 0, w, h, fill=self.ui["canvas"], outline="")
        self.backdrop.create_oval(-140, int(h * 0.08), int(w * 0.42), int(h * 0.82), fill=self.ui["ice_soft"], outline="")
        self.backdrop.create_oval(int(w * 0.26), -140, int(w * 0.92), int(h * 0.54), fill=self.ui["rose_soft"], outline="")
        self.backdrop.create_oval(int(w * 0.68), int(h * 0.42), int(w + 120), h + 120, fill=self.ui["sand_soft"], outline="")
        self.backdrop.create_oval(int(w * 0.72), -60, w + 140, int(h * 0.38), fill=self.ui["ice_soft"], outline="")

    def _create_card(self, parent: tk.Widget, title: str, subtitle: str = "", badge: str = "") -> Tuple[tk.Frame, tk.Frame]:
        card = tk.Frame(parent, bg=self.ui["card"], highlightbackground=self.ui["line"], highlightthickness=1, bd=0)
        header = tk.Frame(card, bg=self.ui["card"])
        header.pack(fill=tk.X, padx=18, pady=(16, 10))

        title_box = tk.Frame(header, bg=self.ui["card"])
        title_box.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(title_box, text=title, font=self.font_heading, fg=self.ui["ink"], bg=self.ui["card"]).pack(anchor="w")
        if subtitle:
            tk.Label(
                title_box,
                text=subtitle,
                font=self.font_caption,
                fg=self.ui["muted"],
                bg=self.ui["card"],
                wraplength=720,
                justify="left",
            ).pack(anchor="w", pady=(4, 0))
        if badge:
            self._make_pill(header, badge, self.ui["accent_soft"], self.ui["accent"]).pack(side=tk.RIGHT, padx=(12, 0))

        body = tk.Frame(card, bg=self.ui["card"])
        body.pack(fill=tk.BOTH, expand=True, padx=18, pady=(0, 18))
        return card, body

    def _create_soft_panel(self, parent: tk.Widget, title: str) -> tk.Frame:
        panel = tk.Frame(parent, bg=self.ui["card_soft"], highlightbackground=self.ui["line_soft"], highlightthickness=1, bd=0)
        tk.Label(panel, text=title, font=self.font_label, fg=self.ui["ink_soft"], bg=self.ui["card_soft"]).pack(anchor="w", padx=14, pady=(12, 8))
        return panel

    def _make_pill(self, parent: tk.Widget, text: str, bg: str, fg: str) -> tk.Label:
        return tk.Label(
            parent,
            text=text,
            font=self.font_small,
            fg=fg,
            bg=bg,
            padx=10,
            pady=5,
            bd=0,
        )

    def _make_field_label(self, parent: tk.Widget, text: str) -> tk.Label:
        return tk.Label(parent, text=text, font=self.font_label, fg=self.ui["ink_soft"], bg=parent.cget("bg"))

    def _make_check(
        self,
        parent: tk.Widget,
        text: str,
        variable: tk.BooleanVar,
        command: Optional[Callable[[], None]] = None,
    ) -> tk.Checkbutton:
        return tk.Checkbutton(
            parent,
            text=text,
            variable=variable,
            command=command,
            bg=parent.cget("bg"),
            activebackground=parent.cget("bg"),
            fg=self.ui["ink"],
            activeforeground=self.ui["ink"],
            selectcolor=self.ui["card_strong"],
            highlightthickness=0,
            bd=0,
            padx=2,
            pady=4,
            font=self.font_ui,
        )

    def _style_text_widget(self, widget: ScrolledText) -> None:
        widget.configure(
            bg=self.ui["card_strong"],
            fg=self.ui["ink"],
            insertbackground=self.ui["ink"],
            relief="flat",
            bd=0,
            padx=12,
            pady=10,
            highlightthickness=1,
            highlightbackground=self.ui["line"],
            wrap=tk.WORD,
        )
        try:
            widget.vbar.configure(
                bg=self.ui["card_soft"],
                activebackground=self.ui["accent_soft"],
                troughcolor=self.ui["canvas_soft"],
                relief="flat",
                bd=0,
                width=11,
            )
        except Exception:
            pass

    def _set_status(self, text: str) -> None:
        self.after(0, lambda: self.status_var.set(text))

    def _build_ui(self) -> None:
        self.shell = tk.Frame(self, bg=self.ui["canvas"])
        self.shell.pack(fill=tk.BOTH, expand=True, padx=28, pady=24)
        self.shell.columnconfigure(0, weight=1)
        self.shell.rowconfigure(1, weight=1)

        hero_card, hero_body = self._create_card(
            self.shell,
            "A calmer liquid-glass shell for overlap review",
            "LLM vision workflow for microfiche transcript PDFs. The classifier marks overlap, blurry, clean, or uncertain page states and can export results immediately while scanning.",
            f"v{APP_VERSION}",
        )
        hero_card.grid(row=0, column=0, sticky="ew", pady=(0, 16))
        tk.Label(hero_body, text="MICROFICHE VISION WORKFLOW", font=self.font_overline, fg=self.ui["muted"], bg=self.ui["card"]).pack(anchor="w")
        pill_row = tk.Frame(hero_body, bg=self.ui["card"])
        pill_row.pack(anchor="w", pady=(14, 0))
        self._make_pill(pill_row, "Pure LLM detection", self.ui["accent_soft"], self.ui["accent"]).pack(side=tk.LEFT, padx=(0, 8))
        self._make_pill(pill_row, "Live O_ / B_ / E_ output", self.ui["rose_soft"], self.ui["ink"]).pack(side=tk.LEFT, padx=(0, 8))
        self._make_pill(pill_row, "Per-page decision log", self.ui["sand_soft"], self.ui["ink_soft"]).pack(side=tk.LEFT)

        dashboard = tk.Frame(self.shell, bg=self.ui["canvas"])
        dashboard.grid(row=1, column=0, sticky="nsew")
        dashboard.columnconfigure(0, weight=3)
        dashboard.columnconfigure(1, weight=2)
        dashboard.rowconfigure(3, weight=1)

        model_card, model_body = self._create_card(
            dashboard,
            "Model Profiles",
            "Display names stay simple. Profiles can point to remote API providers or local OpenAI-compatible VLM endpoints such as Ollama or vLLM.",
        )
        model_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=(0, 10))
        model_body.columnconfigure(1, weight=1)
        model_body.columnconfigure(3, weight=1)

        self.model_name_var = tk.StringVar()
        self._make_field_label(model_body, "Profile").grid(row=0, column=0, sticky="w")
        self.model_combo = ttk.Combobox(model_body, textvariable=self.model_name_var, state="readonly", style="Glass.TCombobox", width=38)
        self.model_combo.grid(row=0, column=1, sticky="ew", padx=(0, 10))
        self.model_combo.bind("<<ComboboxSelected>>", self.on_select_model)

        btn_strip = tk.Frame(model_body, bg=self.ui["card"])
        btn_strip.grid(row=0, column=2, columnspan=2, sticky="e")
        self.add_model_btn = ttk.Button(btn_strip, text="Add", style="Glass.TButton", command=self.add_model)
        self.add_model_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.del_model_btn = ttk.Button(btn_strip, text="Delete", style="Glass.TButton", command=self.delete_model)
        self.del_model_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.save_model_btn = ttk.Button(btn_strip, text="Save", style="Glass.TButton", command=self.save_model)
        self.save_model_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.test_model_btn = ttk.Button(btn_strip, text="Test", style="Glass.TButton", command=self.test_model)
        self.test_model_btn.pack(side=tk.LEFT)

        self._make_field_label(model_body, "Base URL").grid(row=1, column=0, sticky="w", pady=(14, 0))
        self.base_url_var = tk.StringVar()
        self.show_sensitive_var = tk.BooleanVar(value=False)
        self.base_url_entry = ttk.Entry(model_body, textvariable=self.base_url_var, style="Glass.TEntry", show="*")
        self.base_url_entry.grid(row=1, column=1, columnspan=3, sticky="ew", pady=(14, 0))
        self._make_check(
            model_body,
            "Show base URL / API key",
            self.show_sensitive_var,
            command=self.toggle_sensitive_fields,
        ).grid(row=1, column=4, sticky="e", padx=(10, 0), pady=(14, 0))

        self._make_field_label(model_body, "Model Name").grid(row=2, column=0, sticky="w", pady=(14, 0))
        self.model_id_var = tk.StringVar()
        self.model_id_entry = ttk.Entry(model_body, textvariable=self.model_id_var, style="Glass.TEntry")
        self.model_id_entry.grid(row=2, column=1, sticky="ew", padx=(0, 10), pady=(14, 0))

        self._make_field_label(model_body, "API Key (optional)").grid(row=2, column=2, sticky="w", pady=(14, 0))
        self.api_key_var = tk.StringVar()
        self.api_key_entry = ttk.Entry(model_body, textvariable=self.api_key_var, show="*", style="Glass.TEntry")
        self.api_key_entry.grid(row=2, column=3, sticky="ew", pady=(14, 0))

        self._make_field_label(model_body, "Timeout(s)").grid(row=3, column=0, sticky="w", pady=(14, 0))
        self.timeout_var = tk.StringVar(value="120")
        self.timeout_entry = ttk.Entry(model_body, textvariable=self.timeout_var, style="Glass.TEntry", width=10)
        self.timeout_entry.grid(row=3, column=1, sticky="w", pady=(14, 0))
        tk.Label(
            model_body,
            text="Use a vision-capable model. Remote pools usually need an API key. Local OpenAI-compatible endpoints can leave API key empty.",
            font=self.font_caption,
            fg=self.ui["muted"],
            bg=self.ui["card"],
        ).grid(row=3, column=2, columnspan=2, sticky="e", pady=(14, 0))

        run_card, run_body = self._create_card(
            dashboard,
            "Run Control",
            "Run and stop the selected workflow. Progress and status update without waiting for the full batch to finish.",
        )
        run_card.grid(row=0, column=1, sticky="nsew", pady=(0, 10))
        self.status_var = tk.StringVar(value="Ready. Identify overlap or blurry pages with the selected model.")
        self._make_pill(run_body, "Overlap / Blurry / Clean / Uncertain", self.ui["accent_soft"], self.ui["ink"]).pack(anchor="w")
        tk.Label(
            run_body,
            textvariable=self.status_var,
            font=self.font_status,
            fg=self.ui["ink_soft"],
            bg=self.ui["card"],
            wraplength=360,
            justify="left",
        ).pack(anchor="w", pady=(12, 14))
        run_buttons = tk.Frame(run_body, bg=self.ui["card"])
        run_buttons.pack(fill=tk.X)
        self.run_btn = ttk.Button(run_buttons, text="Run", style="Accent.TButton", command=self.run_pipeline)
        self.run_btn.pack(side=tk.LEFT)
        self.stop_btn = ttk.Button(run_buttons, text="Stop", style="Danger.TButton", command=self.stop_pipeline)
        self.stop_btn.pack(side=tk.LEFT, padx=(8, 0))
        self.progress = ttk.Progressbar(run_body, mode="determinate", style="Glass.Horizontal.TProgressbar")
        self.progress.pack(fill=tk.X, pady=(16, 10))
        tk.Label(
            run_body,
            text="Logs will explicitly tell you whether a page was identified as overlap, blurry, clean, uncertain, or failed due to a request/model error.",
            font=self.font_caption,
            fg=self.ui["muted"],
            bg=self.ui["card"],
            wraplength=360,
            justify="left",
        ).pack(anchor="w")

        scan_card, scan_body = self._create_card(
            dashboard,
            "Scan Sources And Actions",
            "Choose the source folder, optional replacement folder, and which outputs to generate while the run is active.",
        )
        scan_card.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        scan_body.columnconfigure(1, weight=1)

        self.source_dir_var = tk.StringVar()
        self.replacement_dir_var = tk.StringVar()
        self.csv_path_var = tk.StringVar()
        self.recursive_var = tk.BooleanVar(value=False)
        self.act_csv_var = tk.BooleanVar(value=True)
        self.act_identify_var = tk.BooleanVar(value=True)
        self.act_extract_var = tk.BooleanVar(value=True)
        self.act_replace_var = tk.BooleanVar(value=False)
        self.act_blurry_var = tk.BooleanVar(value=False)
        self.live_output_var = tk.BooleanVar(value=True)
        self.fast_mode_var = tk.BooleanVar(value=False)

        self._make_field_label(scan_body, "Source Directory").grid(row=0, column=0, sticky="w")
        self.source_dir_entry = ttk.Entry(scan_body, textvariable=self.source_dir_var, style="Glass.TEntry")
        self.source_dir_entry.grid(row=0, column=1, sticky="ew", padx=(0, 10))
        ttk.Button(scan_body, text="Browse", style="Glass.TButton", command=self.pick_source_dir).grid(row=0, column=2, sticky="e")

        self._make_field_label(scan_body, "Replacement Directory (optional)").grid(row=1, column=0, sticky="w", pady=(12, 0))
        self.replacement_dir_entry = ttk.Entry(scan_body, textvariable=self.replacement_dir_var, style="Glass.TEntry")
        self.replacement_dir_entry.grid(row=1, column=1, sticky="ew", padx=(0, 10), pady=(12, 0))
        ttk.Button(scan_body, text="Browse", style="Glass.TButton", command=self.pick_replacement_dir).grid(row=1, column=2, sticky="e", pady=(12, 0))

        self._make_field_label(scan_body, "CSV Output Path").grid(row=2, column=0, sticky="w", pady=(12, 0))
        self.csv_path_entry = ttk.Entry(scan_body, textvariable=self.csv_path_var, style="Glass.TEntry")
        self.csv_path_entry.grid(row=2, column=1, sticky="ew", padx=(0, 10), pady=(12, 0))
        ttk.Button(scan_body, text="Pick", style="Glass.TButton", command=self.pick_csv_path).grid(row=2, column=2, sticky="e", pady=(12, 0))

        options_row = tk.Frame(scan_body, bg=self.ui["card"])
        options_row.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(16, 0))
        options_row.columnconfigure(0, weight=1)
        options_row.columnconfigure(1, weight=1)

        mode_panel = self._create_soft_panel(options_row, "Mode")
        mode_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self._make_check(mode_panel, "Recursive scan", self.recursive_var).pack(anchor="w", padx=12)
        self._make_check(mode_panel, "Live output while scanning", self.live_output_var).pack(anchor="w", padx=12)
        self._make_check(mode_panel, "Fast mode (lower DPI)", self.fast_mode_var).pack(anchor="w", padx=12, pady=(0, 10))

        action_panel = self._create_soft_panel(options_row, "Outputs")
        action_panel.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        self._make_check(action_panel, "1) Generate overlap CSV", self.act_csv_var).pack(anchor="w", padx=12)
        self._make_check(action_panel, "2) Export O_*.pdf pages", self.act_identify_var).pack(anchor="w", padx=12)
        self._make_check(action_panel, "3) Export E_*.pdf (remove overlaps)", self.act_extract_var).pack(anchor="w", padx=12)
        self._make_check(action_panel, "4) Replace overlaps -> R_*.pdf", self.act_replace_var).pack(anchor="w", padx=12)
        self._make_check(action_panel, "5) Export B_*.pdf blurry pages", self.act_blurry_var).pack(anchor="w", padx=12, pady=(0, 10))

        prompt_card, prompt_body = self._create_card(
            dashboard,
            "Custom Prompt",
            "Use this only for edge cases. The built-in prompt already defines overlap and blurry behavior.",
        )
        prompt_card.grid(row=2, column=0, sticky="nsew", padx=(0, 10), pady=(0, 10))
        self.prompt_text = ScrolledText(prompt_body, height=9)
        self.prompt_text.pack(fill=tk.BOTH, expand=True)
        self._style_text_widget(self.prompt_text)

        mem_card, mem_body = self._create_card(
            dashboard,
            "Memory And Training",
            "Store corrections and persistent notes so future runs keep the same judgement style for your dataset.",
        )
        mem_card.grid(row=2, column=1, sticky="nsew", pady=(0, 10))
        mem_body.columnconfigure(0, weight=1)
        mem_body.columnconfigure(1, weight=1)
        ttk.Button(mem_body, text="Import Training CSV", style="Glass.TButton", command=self.import_training).grid(row=0, column=0, sticky="ew", padx=(0, 6), pady=(0, 8))
        ttk.Button(mem_body, text="Add Memory Note", style="Glass.TButton", command=self.add_memory_note).grid(row=0, column=1, sticky="ew", pady=(0, 8))
        ttk.Button(mem_body, text="Correct Last Scan Page", style="Glass.TButton", command=self.correct_last_scan_page).grid(row=1, column=0, sticky="ew", padx=(0, 6), pady=(0, 8))
        ttk.Button(mem_body, text="Save Memory", style="Glass.TButton", command=self.save_memory).grid(row=1, column=1, sticky="ew", pady=(0, 8))
        ttk.Button(mem_body, text="Show Memory Stats", style="Glass.TButton", command=self.show_memory_stats).grid(row=2, column=0, columnspan=2, sticky="ew")
        self.memory_info_var = tk.StringVar(value="Memory: 0 notes, 0 overrides")
        tk.Label(
            mem_body,
            textvariable=self.memory_info_var,
            font=self.font_caption,
            fg=self.ui["muted"],
            bg=self.ui["card"],
            wraplength=340,
            justify="left",
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(14, 0))

        log_card, log_body = self._create_card(
            dashboard,
            "Execution Log",
            "Per-page decisions and model failures are written here. This is the first place to check if a request times out or the model returns an uncertain answer.",
        )
        log_card.grid(row=3, column=0, columnspan=2, sticky="nsew")
        log_body.rowconfigure(0, weight=1)
        log_body.columnconfigure(0, weight=1)
        self.log_text = ScrolledText(log_body, height=15)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self._style_text_widget(self.log_text)

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
        corrections = len(self.memory.get("correction_history", []))
        self.memory_info_var.set(f"Memory: {notes} notes, {overrides} overrides, {corrections} corrections")

    def toggle_sensitive_fields(self) -> None:
        show_char = "" if self.show_sensitive_var.get() else "*"
        self.base_url_entry.configure(show=show_char)
        self.api_key_entry.configure(show=show_char)

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
        model = simpledialog.askstring(
            "Add Model",
            "Model Name / Alias (for built-ins use names like GPT-5.4, Kimi-K2.5; for local endpoints use the local model id):",
            parent=self,
        )
        if not model:
            return
        api_key = simpledialog.askstring("Add Model", "API Key (optional; leave blank for local endpoints):", parent=self, show="*") or ""
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

        def worker() -> None:
            self._set_status(f"Testing model connectivity for {profile.name}...")
            self.log(f"Testing model: {profile.name} / {profile.model}")
            client = OpenAICompatibleClient(profile)
            ok, msg = client.quick_test()
            if ok:
                self._set_status(f"Model test passed for {profile.name}.")
                self.log(f"Model test OK: {msg}")
            else:
                self._set_status(f"Model test failed for {profile.name}. Check log.")
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

    def correct_last_scan_page(self) -> None:
        last_scan = self.storage.load_last_scan()
        if not last_scan:
            messagebox.showerror("No Scan Data", "No last_scan.json found. Run a scan first.")
            return
        CorrectionPicker(self, last_scan)

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
        corrections = len(self.memory.get("correction_history", []))
        messagebox.showinfo(
            "Memory Stats",
            f"Global notes: {notes}\nOverrides: {overrides}\nCorrections: {corrections}",
        )

    def stop_pipeline(self) -> None:
        self.cancel_event.set()
        self._set_status("Stop requested. Finishing the current request before shutdown.")
        self.log("Stop requested.")

    def run_pipeline(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showwarning("Busy", "A pipeline is already running.")
            return

        profile = self.current_profile()
        if not profile:
            messagebox.showerror("Error", "No model selected.")
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
        live_output = bool(self.live_output_var.get())
        fast_mode = bool(self.fast_mode_var.get())
        render_dpi = 170 if fast_mode else 220

        self.cancel_event.clear()
        self.progress["value"] = 0
        self._set_status(f"Starting {profile.name} scan...")

        def progress_cb(done: int, total: int) -> None:
            def _set() -> None:
                self.progress["maximum"] = total
                self.progress["value"] = done

            self.after(0, _set)

        def worker() -> None:
            live_csv_fh = None
            try:
                self.log("Starting pipeline...")
                self.log(
                    "Identify overlap/blurry settings: "
                    f"live_output={live_output}, render_dpi={render_dpi}, "
                    f"overlap_csv={self.act_csv_var.get()}, export_O={self.act_identify_var.get()}, "
                    f"export_E={self.act_extract_var.get()}, replace_R={self.act_replace_var.get()}, "
                    f"export_B={self.act_blurry_var.get()}"
                )
                src_pdfs = list_pdfs(source_dir, recursive=recursive)
                if not src_pdfs:
                    self._set_status("No source PDFs found.")
                    self.log("No PDF files found in source directory.")
                    return
                self._set_status(f"Scanning {len(src_pdfs)} source PDFs with {profile.name}...")
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
                    render_dpi=render_dpi,
                )

                live_csv_writer: Optional[csv.DictWriter] = None
                live_o_count = 0
                live_e_count = 0
                live_b_count = 0
                if live_output and self.act_csv_var.get():
                    csv_path.parent.mkdir(parents=True, exist_ok=True)
                    live_csv_fh = csv_path.open("w", newline="", encoding="utf-8")
                    live_csv_writer = csv.DictWriter(live_csv_fh, fieldnames=OVERLAP_CSV_FIELDS)
                    live_csv_writer.writeheader()
                    live_csv_fh.flush()
                    self.log(f"Live CSV started: {csv_path}")

                def on_source_page_result(rec: Dict[str, Any], pdf_path: Path, doc: fitz.Document) -> None:
                    nonlocal live_o_count, live_b_count
                    if rec.get("scope") != "source":
                        return
                    decision = str(rec.get("decision") or "unknown")
                    page_num = int(rec.get("page") or 0)
                    self._set_status(f"{pdf_path.name} p{page_num:03d}: {decision}")
                    if live_output and self.act_csv_var.get() and live_csv_writer and live_csv_fh and rec.get("is_overlap"):
                        live_csv_writer.writerow(overlap_row_for_csv(rec))
                        live_csv_fh.flush()

                    if live_output and self.act_identify_var.get() and rec.get("is_overlap"):
                        if export_single_overlap_page_from_doc(doc, pdf_path, int(rec["page"]), self.log):
                            live_o_count += 1

                    if live_output and self.act_blurry_var.get() and rec.get("is_blurry"):
                        if export_single_blurry_page_from_doc(doc, pdf_path, int(rec["page"]), self.log):
                            live_b_count += 1

                def on_source_file_done(pdf_path: Path, doc: fitz.Document, file_records: List[Dict[str, Any]]) -> None:
                    nonlocal live_e_count
                    if self.cancel_event.is_set():
                        return
                    if live_output and self.act_extract_var.get():
                        if export_extracted_non_overlap_for_file(doc, pdf_path, file_records, self.log):
                            live_e_count += 1

                source_records = engine.scan_pdfs(
                    src_pdfs,
                    scope="source",
                    custom_prompt=custom_prompt,
                    on_page_result=on_source_page_result if live_output else None,
                    on_file_done=on_source_file_done if live_output else None,
                )
                all_records = list(source_records)

                if extra_pdfs and not self.cancel_event.is_set():
                    self._set_status(f"Scanning {len(extra_pdfs)} replacement PDFs...")
                    self.log("Scanning replacement directory for candidate pages...")
                    extra_records = engine.scan_pdfs(extra_pdfs, scope="replacement", custom_prompt=custom_prompt)
                    all_records.extend(extra_records)

                self.storage.save_last_scan(all_records)
                self.log(f"Scan complete. Total pages processed: {len(all_records)}")

                if self.cancel_event.is_set():
                    self._set_status("Stopped before finishing output actions.")
                    self.log("Pipeline stopped before actions.")
                    return

                if self.act_csv_var.get():
                    if live_output:
                        count = len([r for r in source_records if r.get("is_overlap")])
                        self.log(f"Action 1 done (live): CSV saved to {csv_path} (overlaps={count})")
                    else:
                        count = write_overlap_csv(source_records, csv_path)
                        self.log(f"Action 1 done: CSV saved to {csv_path} (overlaps={count})")

                if self.act_identify_var.get():
                    if live_output:
                        self.log(f"Action 2 done (live): exported O_ pages = {live_o_count}")
                    else:
                        cnt = export_overlap_pages(source_records, self.log)
                        self.log(f"Action 2 done: exported O_ pages = {cnt}")

                if self.act_extract_var.get():
                    if live_output:
                        self.log(f"Action 3 done (live): created E_ files = {live_e_count}")
                    else:
                        cnt = export_extracted_non_overlap(source_records, self.log)
                        self.log(f"Action 3 done: created E_ files = {cnt}")

                if self.act_replace_var.get():
                    cnt, rep_csv = replace_overlap_pages(source_records, all_records, self.log)
                    self.log(f"Action 4 done: created R_ files = {cnt}")
                    self.log(f"Replacement report: {rep_csv}")

                if self.act_blurry_var.get():
                    if live_output:
                        self.log(f"Action 5 done (live): exported B_ pages = {live_b_count}")
                    else:
                        cnt = export_blurry_pages(source_records, self.log)
                        self.log(f"Action 5 done: exported B_ pages = {cnt}")

                self._set_status("Pipeline finished. Review the log and generated files.")
                self.log("Pipeline finished.")
            except Exception:
                self._set_status("Pipeline crashed. Check the execution log.")
                self.log("Pipeline crashed:\n" + traceback.format_exc())
            finally:
                if live_csv_fh:
                    try:
                        live_csv_fh.close()
                    except Exception:
                        pass
                self.after(0, lambda: self.progress.configure(value=0))

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
