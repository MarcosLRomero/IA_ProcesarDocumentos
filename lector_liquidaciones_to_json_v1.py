# -*- coding: utf-8 -*-
r"""
lector_liquidaciones_to_json_v1.py

- Lee documentos de múltiples páginas (JPG/PNG/WEBP o PDF).
- Llama a OpenAI y devuelve un archivo de texto con dos columnas: CONCEPTO|TOTAL.
- Modo GUI opcional (--gui): ventana simple con estado, barra, tiempo transcurrido y log.
- Importante: al finalizar OK imprime SOLO la ruta del TXT por stdout (para VB6).
  En error: sale con código != 0 y escribe el mensaje de error en stderr.

Requisitos:
  pip install openai python-dotenv pillow pypdf

Uso:
  python lector_liquidaciones_to_json_v1.py liquidacion.pdf --outdir E:\temp
  python lector_liquidaciones_to_json_v1.py img1.jpg img2.jpg --outdir E:\temp --gui

EXE (PyInstaller):
  pyinstaller --onefile --noconsole lector_liquidaciones_to_json_v1.py
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import io
import json
import os
import re
import sys
import tempfile
import threading
import time
import queue
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI


# ----------------------------
# GUI (Tkinter) opcional
# ----------------------------
try:
    import tkinter as tk
    from tkinter import ttk
except Exception:
    tk = None
    ttk = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    from pypdf import PdfReader, PdfWriter
except Exception:
    PdfReader = None
    PdfWriter = None


class StatusUI:
    """Ventana simple: estado + barra indeterminada + tiempo + log.
    NO escribe en stdout (para no romper VB6).
    """

    def __init__(self, title="Procesando liquidación...", width=560, height=260):
        if tk is None or ttk is None:
            raise RuntimeError("Tkinter no está disponible en este entorno.")

        self.q: "queue.Queue[str]" = queue.Queue()
        self.t0 = time.time()

        self.root = tk.Tk()
        self.root.title(title)
        # Center window on screen
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = max(0, int((sw - width) / 2))
        y = max(0, int((sh - height) / 2))
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.resizable(False, False)

        self.lbl = ttk.Label(self.root, text="Iniciando...", font=("Segoe UI", 10))
        self.lbl.pack(padx=12, pady=(12, 4), anchor="w")

        self.lbl_time = ttk.Label(self.root, text="Tiempo: 00:00", font=("Segoe UI", 9))
        self.lbl_time.pack(padx=12, pady=(0, 6), anchor="w")

        self.pb = ttk.Progressbar(self.root, mode="indeterminate")
        self.pb.pack(fill="x", padx=12, pady=(0, 10))
        self.pb.start(10)

        self.txt = tk.Text(self.root, height=9, wrap="word")
        self.txt.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.txt.configure(state="disabled")

        self._closed = False
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.root.after(100, self._poll)
        self.root.after(200, self._tick_time)

    def _on_close(self):
        # Si cierran la ventana, no matamos el proceso; solo ocultamos.
        self._closed = True
        try:
            self.root.withdraw()
        except Exception:
            pass

    def _tick_time(self):
        if not self._closed:
            secs = int(time.time() - self.t0)
            mm = secs // 60
            ss = secs % 60
            self.lbl_time.configure(text=f"Tiempo: {mm:02d}:{ss:02d}")
            self.root.after(200, self._tick_time)

    def push(self, msg: str):
        """Seguro desde cualquier hilo."""
        try:
            self.q.put_nowait(msg)
        except Exception:
            pass

    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                if msg.startswith("STATUS:"):
                    self.lbl.configure(text=msg.replace("STATUS:", "", 1).strip())
                else:
                    self._append_log(msg)
        except queue.Empty:
            pass

        if not self._closed:
            self.root.after(120, self._poll)

    def _append_log(self, s: str):
        self.txt.configure(state="normal")
        self.txt.insert("end", s + "\n")
        self.txt.see("end")
        self.txt.configure(state="disabled")

    def close(self):
        try:
            self.pb.stop()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass

    def mainloop(self):
        self.root.mainloop()


# ----------------------------
# Utilidades generales
# ----------------------------
def app_dir() -> Path:
    """Carpeta base del .py o del .exe (cuando está 'frozen')."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        # En PyInstaller, el ejecutable real está en sys.executable
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def load_env_near_app() -> None:
    """Carga .env desde la carpeta del script/exe si existe."""
    env_path = app_dir() / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path), override=False)
    else:
        # igual intentamos por si hay .env en cwd
        load_dotenv(override=False)


def safe_basename(file_path: str) -> str:
    name = Path(file_path).stem
    name = re.sub(r"[^a-zA-Z0-9_\-]+", "_", name).strip("_")
    return name or "liquidacion"


def sanitize_json_text(s: str) -> str:
    s = s.strip()
    if s.startswith("\ufeff"):
        s = s.lstrip("\ufeff")
    # remove trailing commas before } or ]
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s


def extract_first_json(text: str) -> dict:
    """Extrae el primer JSON válido del texto (tolerante a basura alrededor)."""
    if not text:
        raise ValueError("Respuesta vacía del modelo.")

    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return json.loads(sanitize_json_text(s))

    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        candidate = s[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return json.loads(sanitize_json_text(candidate))

    raise ValueError("No se pudo extraer JSON de la respuesta.")


def _ensure_object(x: Any) -> dict:
    return x if isinstance(x, dict) else {}


def _ensure_list(x: Any) -> list:
    return x if isinstance(x, list) else []


def _parse_number(raw: Any) -> float:
    if raw is None:
        return 0.0
    s = str(raw).strip()
    if not s:
        return 0.0
    # keep digits, comma, dot, minus
    s = re.sub(r"[^\d,.\-]", "", s)
    if not s:
        return 0.0
    # decide decimal separator
    if "," in s and "." in s:
        if s.rfind(".") > s.rfind(","):
            s = s.replace(",", "")
        else:
            s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return 0.0


def _round2(val: float) -> float:
    try:
        return float(f"{val:.2f}")
    except Exception:
        return 0.0

def _norm_text(s: str) -> str:
    s = s or ""
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s.upper()

def _ensure_keyword(concept: str, keyword: str) -> str:
    t = _norm_text(concept)
    if keyword in t:
        return concept
    return f"{keyword} - {concept}".strip()


def _classify_concept_name(concept: str) -> str:
    """Clasifica con las mismas reglas que el consumidor VB6."""
    t = _norm_text(concept)
    if "RET_GAN" in t:
        return "RET_GAN"
    if "RET_IIBB" in t:
        return "RET_IIBB"
    if "RET_IVA" in t:
        return "RET_IVA"
    if "IVA_CREDITO" in t:
        return "IVA_CREDITO"
    if "BANCO" in t:
        return "BANCO"
    if (
        "ACREDITADO" in t
        or "LIQUIDADO" in t
        or "NETO DE PAGOS" in t
        or "NETO A COBRAR" in t
        or "A DEPOSITAR" in t
    ):
        return "BANCO"
    if "ARANCEL" in t or "COMISION" in t or "CARGO" in t:
        return "GASTO"
    if "IMPUESTO" in t and "IVA" not in t and "GANANCIAS" not in t and "IIBB" not in t and "INGRESOS" not in t:
        return "GASTO"
    if ("GASTO" in t or "COMISION" in t) and "IVA" not in t and "CREDITO" not in t:
        return "GASTO"
    if "IVA" in t and ("ARANCEL" in t or "COMISION" in t or "GASTO" in t):
        return "IVA_CREDITO"
    if ("IVA" in t and "RET" in t) or ("IVA" in t and "PERCEP" in t) or "R.G. 2408" in t or "RG 2408" in t:
        return "RET_IVA"
    if (
        "INGRESOS" in t
        or "IIBB" in t
        or "SIRTAC" in t
        or "ING.BRUTOS" in t
        or "ING. BRUTOS" in t
        or "ING BRUTOS" in t
    ):
        return "RET_IIBB"
    if "GANANCIAS" in t or ("RET" in t and "GAN" in t) or "RG 830" in t:
        return "RET_GAN"
    if ("IVA" in t and "CREDITO" in t) or "CRED.FISC" in t or "CRED FISC" in t or ("IVA" in t and "CRED" in t and "FISC" in t):
        return "IVA_CREDITO"
    if "TARJETA" in t and "IVA" not in t:
        return "TARJETA"
    return "OTROS"


def _ensure_keywords_for_category(concept: str, category: str) -> str:
    if category == "TARJETA":
        return _ensure_keyword(concept, "TARJETA")
    if category == "BANCO":
        return _ensure_keyword(concept, "BANCO")
    if category == "GASTO":
        return _ensure_keyword(concept, "GASTO")
    if category == "IVA_CREDITO":
        return _ensure_keyword(concept, "IVA CREDITO")
    if category == "RET_IVA":
        return _ensure_keyword(concept, "IVA RET")
    if category == "RET_IIBB":
        return _ensure_keyword(concept, "IIBB")
    if category == "RET_GAN":
        return _ensure_keyword(concept, "GANANCIAS")
    return _ensure_keyword(concept, "OTROS")


def _canonical_label_for_category(category: str) -> str:
    mapping = {
        "TARJETA": "TARJETA",
        "BANCO": "BANCO",
        "GASTO": "GASTO",
        "IVA_CREDITO": "IVA_CREDITO",
        "RET_IVA": "RET_IVA",
        "RET_IIBB": "RET_IIBB",
        "RET_GAN": "RET_GAN",
        "OTROS": "OTROS",
    }
    return mapping.get(category, "OTROS")


def _normalize_total_for_category(total: str, category: str) -> str:
    value = _parse_number(total)
    if category == "TARJETA":
        value = abs(value)
    else:
        value = -abs(value)
    value = _round2(value)
    if abs(value) < 0.005:
        value = 0.0
    return f"{value:.2f}"

def _postprocess_output(text: str) -> str:
    lines_in = []
    for raw in text.splitlines():
        ln = raw.strip()
        if not ln:
            continue
        if ln.startswith("```") or ln.startswith("**") or ln == "---":
            continue
        lines_in.append(ln)
    if not lines_in:
        return text

    out: List[str] = []
    in_control = False
    main_lines: List[str] = []

    for ln in lines_in:
        if _norm_text(ln) == "CONTROL_TOTALES_DIARIOS":
            in_control = True
            out.extend(_apply_keywords_to_main(main_lines))
            main_lines = []
            out.append(ln)
            continue

        if not in_control:
            main_lines.append(ln)
        else:
            out.append(ln)

    if main_lines:
        out.extend(_apply_keywords_to_main(main_lines))

    return "\n".join(out) + "\n"

def _apply_keywords_to_main(lines: List[str]) -> List[str]:
    categories = ["TARJETA", "BANCO", "GASTO", "IVA_CREDITO", "RET_IVA", "RET_IIBB", "RET_GAN", "OTROS"]
    sums: Dict[str, float] = {k: 0.0 for k in categories}
    row_idx = 0
    fallback_by_position = {
        1: "TARJETA",
        2: "BANCO",
        3: "GASTO",
        4: "IVA_CREDITO",
        5: "RET_IVA",
        6: "RET_IIBB",
        7: "RET_GAN",
        8: "OTROS",
    }
    for ln in lines:
        if "|" not in ln:
            continue

        concept, total = ln.split("|", 1)
        concept = concept.strip()
        total = total.strip()
        t_concept = _norm_text(concept)
        if t_concept in ("CONCEPTO", "TIPO", "TIPOCONCEPTOIA"):
            continue
        # Evita encabezados como CONCEPTO|TOTAL
        if _norm_text(total) in ("TOTAL", "IMPORTE"):
            continue

        row_idx += 1
        cat = _classify_concept_name(concept)
        # Si el modelo devuelve "OTROS" genérico en las primeras filas,
        # recuperamos la estructura esperada original.
        if cat == "OTROS" and _norm_text(concept) in ("OTRO", "OTROS", "OTHER"):
            cat = fallback_by_position.get(row_idx, "OTROS")
        # Fallback para la línea principal cuando el modelo no incluye la palabra TARJETA.
        if row_idx == 1 and cat == "OTROS":
            cat = "TARJETA"
        sums[cat] += _parse_number(total)

    out: List[str] = []
    for cat in categories:
        concept = _canonical_label_for_category(cat)
        total = _normalize_total_for_category(str(sums[cat]), cat)
        out.append(f"{concept}|{total}")
    return out

def _write_log(log_path: Path, msg: str) -> None:
    try:
        ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass


def _ensure_writable_outdir(preferred: str) -> Path:
    """Devuelve un outdir escribible. Si falla, usa TEMP del sistema."""
    cand = Path(preferred or tempfile.gettempdir())
    try:
        cand.mkdir(parents=True, exist_ok=True)
        test_path = cand / f".__write_test_{os.getpid()}_{int(time.time())}.tmp"
        test_path.write_text("ok", encoding="utf-8")
        test_path.unlink(missing_ok=True)
        return cand
    except Exception:
        fallback = Path(tempfile.gettempdir())
        try:
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback
        except Exception:
            return Path(".")


def _ensure_writable_outdir_with_file_fallback(preferred: str, first_input_file: str) -> Path:
    """Prioriza outdir solicitado; si no se puede, usa carpeta del archivo fuente; último recurso TEMP."""
    # 1) Intentar outdir solicitado (si viene por parámetro)
    if preferred and preferred.strip():
        cand = Path(preferred.strip())
        try:
            cand.mkdir(parents=True, exist_ok=True)
            test_path = cand / f".__write_test_{os.getpid()}_{int(time.time())}.tmp"
            test_path.write_text("ok", encoding="utf-8")
            test_path.unlink(missing_ok=True)
            return cand
        except Exception:
            pass

    # 2) Fallback: carpeta del archivo de entrada
    try:
        src_dir = Path(first_input_file).resolve().parent
        src_dir.mkdir(parents=True, exist_ok=True)
        test_path = src_dir / f".__write_test_{os.getpid()}_{int(time.time())}.tmp"
        test_path.write_text("ok", encoding="utf-8")
        test_path.unlink(missing_ok=True)
        return src_dir
    except Exception:
        pass

    # 3) Último recurso: TEMP
    return _ensure_writable_outdir("")


def _ensure_outdir_preferred_or_fail(preferred: str) -> Path:
    cand = Path((preferred or "").strip())
    if not str(cand):
        raise SystemExit("ERROR: outdir solicitado vacío.")
    try:
        cand.mkdir(parents=True, exist_ok=True)
        test_path = cand / f".__write_test_{os.getpid()}_{int(time.time())}.tmp"
        test_path.write_text("ok", encoding="utf-8")
        test_path.unlink(missing_ok=True)
        return cand
    except Exception as e:
        raise SystemExit(f"ERROR: No se puede escribir en outdir solicitado: {cand} ({e})")


# ----------------------------
# Prompt
# ----------------------------
DEFAULT_PROMPT = r"""
Vas a analizar un PDF correspondiente a UNA liquidación de tarjeta de crédito o débito.
El documento puede tener múltiples páginas.

Tu objetivo es DEVOLVER un archivo de texto con DOS columnas:
CONCEPTO|TOTAL

Reglas obligatorias:

- La PRIMER línea debe ser el TOTAL PRESENTADO de la tarjeta (importe POSITIVO).
- Todas las demás líneas deben ser importes NEGATIVOS.
- La suma total de todos los importes debe ser 0.

Los conceptos a devolver son:

1. Nombre de la tarjeta - TOTAL PRESENTADO
2. Nombre del banco - TOTAL LIQUIDADO / ACREDITADO
3. Gastos de tarjeta / comisiones
4. IVA Crédito Fiscal (solo IVA del gasto)
5. Retenciones / Percepciones de IVA
6. Retenciones / Percepciones de Ingresos Brutos
7. Retenciones / Percepciones de Ganancias
8. Otros conceptos (si existen)

ACLARACIONES IMPORTANTES:

- El IVA Crédito Fiscal corresponde EXCLUSIVAMENTE
  al IVA aplicado sobre COMISIONES / GASTOS de tarjeta.
- NUNCA usar retenciones o percepciones de IVA
  como IVA Crédito Fiscal.
- Si un concepto no se identifica claramente,
  devolver su total como 0.

Formato de salida OBLIGATORIO (sin texto adicional):

CONCEPTO|TOTAL

IMPORTANTE:
- El concepto de la PRIMER línea debe incluir la palabra TARJETA.
- El concepto de la línea del banco debe incluir la palabra BANCO.
- El concepto de gastos debe incluir la palabra GASTO o COMISION.
- El concepto de IVA Crédito Fiscal debe incluir las palabras IVA y CREDITO.
- El concepto de retenciones de IVA debe incluir las palabras IVA y RET.
- El concepto de retenciones de IIBB debe incluir IIBB o INGRESOS.
- El concepto de retenciones de Ganancias debe incluir GANANCIAS.
- El concepto de otros debe incluir la palabra OTROS.

CONTROL ADICIONAL (OPCIONAL):

Algunas liquidaciones contienen, al final de cada día,
un resumen con totales diarios como por ejemplo:
- Ventas con descuento contado
- Arancel
- IVA Crédito Fiscal
- Retenciones / Percepciones
- Importe Neto de Pagos

Si el PDF contiene estos totales diarios:

- Sumá los importes de TODOS los días.
- Agregá una sección adicional al final del archivo
  llamada EXACTAMENTE:

CONTROL_TOTALES_DIARIOS

- En esa sección, devolvé dos columnas:
  CONCEPTO|TOTAL

- NO modifiques ni recalcules el resumen principal.
- NO inventes conceptos que no existan.
- Si el PDF NO contiene totales diarios,
  NO agregues esta sección.
"""


def read_prompt(prompt_file: Optional[str]) -> str:
    if prompt_file:
        p = Path(prompt_file)
        if p.exists():
            return p.read_text(encoding="utf-8", errors="replace")
    return DEFAULT_PROMPT


# ----------------------------
# Conversión de archivos a bloques para OpenAI
# ----------------------------
def file_to_content_block(file_path: str) -> Dict[str, Any]:
    ext = Path(file_path).suffix.lower()
    data = Path(file_path).read_bytes()

    if ext in (".jpg", ".jpeg", ".png", ".webp"):
        b64 = base64.b64encode(data).decode("utf-8")
        if ext in (".jpg", ".jpeg"):
            mime = "image/jpeg"
        elif ext == ".png":
            mime = "image/png"
        else:
            mime = "image/webp"
        return {"type": "input_image", "image_url": f"data:{mime};base64,{b64}"}

    if ext == ".pdf":
        b64 = base64.b64encode(data).decode("utf-8")
        return {
            "type": "input_file",
            "filename": Path(file_path).name,
            "file_data": f"data:application/pdf;base64,{b64}",
        }

    raise ValueError(f"Tipo no soportado: {ext}. Usá JPG/PNG/WEBP o PDF.")


def _count_pdf_pages(file_path: str) -> int:
    if PdfReader is None:
        return 1
    try:
        return max(1, len(PdfReader(file_path).pages))
    except Exception:
        return 1


def _pdf_to_chunked_blocks(file_path: str, pages_per_chunk: int) -> List[Dict[str, Any]]:
    pages_per_chunk = int(pages_per_chunk or 0)
    if pages_per_chunk <= 0:
        return [file_to_content_block(file_path)]

    if PdfReader is None or PdfWriter is None:
        raise SystemExit("ERROR: Para dividir PDFs grandes necesitás instalar pypdf: pip install pypdf")

    reader = PdfReader(file_path)
    total = len(reader.pages)
    if total <= pages_per_chunk:
        return [file_to_content_block(file_path)]

    blocks: List[Dict[str, Any]] = []
    stem = Path(file_path).stem
    for start in range(0, total, pages_per_chunk):
        end = min(total, start + pages_per_chunk)
        writer = PdfWriter()
        for i in range(start, end):
            writer.add_page(reader.pages[i])

        buf = io.BytesIO()
        writer.write(buf)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        blocks.append(
            {
                "type": "input_file",
                "filename": f"{stem}_p{start + 1:03d}-{end:03d}.pdf",
                "file_data": f"data:application/pdf;base64,{b64}",
            }
        )

    return blocks


def _is_request_too_large_error(exc: Exception) -> bool:
    """Detecta errores típicos de límite de tamaño/tokens para reintentar en chunks."""
    msg = str(exc or "").lower()
    return (
        "request too large" in msg
        or "tokens per min" in msg
        or ("rate_limit_exceeded" in msg and "requested" in msg)
    )


def file_to_content_blocks(file_path: str, tiles: int = 1, pdf_chunk_pages: int = 0) -> List[Dict[str, Any]]:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return _pdf_to_chunked_blocks(file_path, pdf_chunk_pages)

    if tiles <= 1 or ext == ".pdf":
        return [file_to_content_block(file_path)]

    if Image is None:
        raise SystemExit("ERROR: Para --tile necesitás instalar Pillow: pip install pillow")

    if ext not in (".jpg", ".jpeg", ".png", ".webp"):
        return [file_to_content_block(file_path)]

    img = Image.open(file_path).convert("RGB")
    w, h = img.size
    tiles = max(1, min(int(tiles), 6))
    slice_h = (h + tiles - 1) // tiles
    overlap = min(60, max(20, slice_h // 10))

    blocks: List[Dict[str, Any]] = []
    for i in range(tiles):
        top = i * slice_h
        bottom = min(h, (i + 1) * slice_h)
        if i > 0:
            top = max(0, top - overlap)
        if i < tiles - 1:
            bottom = min(h, bottom + overlap)

        crop = img.crop((0, top, w, bottom))
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        blocks.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"})

    return blocks


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        add_help=True,
        description="Lector de liquidaciones -> TXT (multipágina). Usa OPENAI_API_KEY (env o .env junto al exe/script).",
    )
    parser.add_argument("files", nargs="*", help="Archivos de entrada (imágenes/PDF) en orden de páginas")
    parser.add_argument("--outdir", default="", help="Carpeta de salida. Default: TEMP del sistema")
    parser.add_argument("--prompt-file", default="", help="Archivo .txt con prompt personalizado")
    parser.add_argument("--model", default="gpt-4o-mini", help="Modelo a usar (default: gpt-4o-mini)")
    parser.add_argument("--gui", action="store_true", help="Muestra ventana de progreso (no altera stdout)")
    parser.add_argument(
        "--per-page",
        action="store_true",
        help="Procesa cada archivo/página por separado y luego unifica (mejora extracción en docs largos).",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-ajusta parámetros (tile y per-page) según cantidad de páginas.",
    )
    parser.add_argument(
        "--tile",
        type=int,
        default=1,
        help="Divide cada página en N franjas horizontales (solo imágenes). Requiere Pillow.",
    )
    parser.add_argument(
        "--pdf-chunk-pages",
        type=int,
        default=0,
        help="Divide PDFs en bloques de N páginas para documentos grandes. 0 = no dividir.",
    )
    args = parser.parse_args()

    ui = None
    if args.gui:
        try:
            ui = StatusUI()
            ui.push("STATUS:Inicializando...")
        except Exception:
            ui = None

    def log(msg: str):
        if ui:
            ui.push(msg)

    def status(msg: str):
        if ui:
            ui.push(f"STATUS:{msg}")

    result = {"out_path": None, "error": None, "log_path": None}

    def worker():
        try:
            status("Cargando .env / variables...")
            load_env_near_app()

            api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
            if not api_key:
                raise SystemExit(
                    "ERROR: No está configurada OPENAI_API_KEY. "
                    "Creá un .env junto al exe/script con OPENAI_API_KEY=... o definila como variable de entorno."
                )

            if not args.files:
                raise SystemExit("ERROR: Debés pasar al menos 1 archivo por parámetro.")
            if len(args.files) > 100:
                raise SystemExit("ERROR: Máximo 100 archivos de entrada.")

            if args.tile < 1 or args.tile > 6:
                raise SystemExit("ERROR: --tile debe ser un entero entre 1 y 6.")
            if args.pdf_chunk_pages < 0:
                raise SystemExit("ERROR: --pdf-chunk-pages no puede ser negativo.")

            status("Validando archivos...")
            for f in args.files:
                if not Path(f).exists():
                    raise SystemExit(f"ERROR: No existe el archivo: {f}")

            # Auto-ajuste según páginas reales (si se puede leer el PDF).
            if args.auto:
                effective_pages = 0
                for f in args.files:
                    if Path(f).suffix.lower() == ".pdf":
                        effective_pages += _count_pdf_pages(f)
                    else:
                        effective_pages += 1

                if effective_pages <= 2:
                    args.tile = 3
                    args.per_page = False
                    if args.pdf_chunk_pages == 0:
                        args.pdf_chunk_pages = 0
                elif effective_pages <= 8:
                    args.tile = 4
                    args.per_page = True
                    if args.pdf_chunk_pages == 0:
                        args.pdf_chunk_pages = 4
                else:
                    args.tile = 5
                    args.per_page = True
                    if args.pdf_chunk_pages == 0:
                        args.pdf_chunk_pages = 5

            status("Preparando salida...")
            base = safe_basename(args.files[0])
            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            requested_outdir = args.outdir.strip()
            if requested_outdir:
                outdir = _ensure_outdir_preferred_or_fail(requested_outdir)
            else:
                outdir = _ensure_writable_outdir_with_file_fallback("", args.files[0])
            log_path = Path(outdir) / f"{base}_{ts}_log.txt"
            result["log_path"] = str(log_path)
            _write_log(log_path, f"Inicio proceso. Archivos: {', '.join(args.files)}")

            status("Cargando prompt...")
            prompt = read_prompt(args.prompt_file.strip() or None)
            if "concepto|total" not in prompt.lower():
                prompt = "Respondé solo con texto en formato CONCEPTO|TOTAL.\n" + prompt
            _write_log(
                log_path,
                f"Modelo: {args.model} | per-page: {args.per_page} | tile: {args.tile} | pdf-chunk-pages: {args.pdf_chunk_pages}",
            )

            status("Armando contenido...")
            log(
                f"Modelo: {args.model} | per-page: {args.per_page} | tile: {args.tile} | pdf-chunk-pages: {args.pdf_chunk_pages}"
            )
            content = [{"type": "input_text", "text": prompt}]
            total_files = len(args.files)
            for i, f in enumerate(args.files, start=1):
                status(f"Adjuntando página {i}/{total_files}...")
                log(f"Archivo: {f}")
                content.extend(file_to_content_blocks(f, args.tile, args.pdf_chunk_pages))

            status("Analizando con Inteligencia Artificial...")
            log("Motor IA: Activo")
            client = OpenAI(api_key=api_key)

            def call_model(content_blocks: List[Dict[str, Any]]) -> str:
                resp = client.responses.create(
                    model=args.model,
                    max_output_tokens=4000,
                    input=[{"role": "user", "content": content_blocks}],
                )

                out_text = ""
                try:
                    out_text = resp.output[0].content[0].text
                except Exception:
                    parts = []
                    for item in getattr(resp, "output", []) or []:
                        for c in getattr(item, "content", []) or []:
                            t = getattr(c, "text", None)
                            if t:
                                parts.append(t)
                    out_text = "\n".join(parts)

                if not out_text.strip():
                    raise SystemExit("ERROR: Respuesta vacía del modelo.")

                return out_text.strip()

            def build_units(force_pdf_page_split: bool = False) -> List[tuple[str, List[Dict[str, Any]]]]:
                units: List[tuple[str, List[Dict[str, Any]]]] = []
                for f in args.files:
                    ext = Path(f).suffix.lower()
                    if ext == ".pdf" and (force_pdf_page_split or args.per_page):
                        chunk = 1 if force_pdf_page_split else (args.pdf_chunk_pages if args.pdf_chunk_pages > 0 else 1)
                        pdf_blocks = _pdf_to_chunked_blocks(f, chunk)
                        for b in pdf_blocks:
                            units.append((f, [b]))
                    else:
                        units.append((f, file_to_content_blocks(f, args.tile, args.pdf_chunk_pages)))
                return units

            def run_units(units: List[tuple[str, List[Dict[str, Any]]]], status_label: str) -> str:
                page_results: List[str] = []
                total_units = len(units)
                t_units_start = time.time()
                for i, (src, blocks) in enumerate(units, start=1):
                    if i > 1:
                        elapsed = time.time() - t_units_start
                        avg = elapsed / (i - 1)
                        remaining = avg * (total_units - i + 1)
                        mm = int(remaining // 60)
                        ss = int(remaining % 60)
                        status(f"{status_label} {i}/{total_units}... (ETA ~{mm:02d}:{ss:02d})")
                    else:
                        status(f"{status_label} {i}/{total_units}...")
                    log(f"Unidad {i}/{total_units}: {src}")
                    unit_content = [{"type": "input_text", "text": prompt}]
                    unit_content.extend(blocks)
                    page_results.append(call_model(unit_content))
                return "\n".join([t for t in page_results if t.strip()])

            units = build_units(force_pdf_page_split=False)
            if args.per_page and len(units) > 1:
                page_results: List[str] = []
                data = run_units(units, "IA por página/bloque")
            else:
                try:
                    data = call_model(content)
                except Exception as e:
                    if not _is_request_too_large_error(e):
                        raise

                    _write_log(log_path, f"Reintento automático por tamaño/tokens: {e!r}")
                    log("Documento grande detectado. Reintentando automáticamente por páginas...")
                    status("Documento grande: reintentando por páginas...")
                    retry_units = build_units(force_pdf_page_split=True)
                    data = run_units(retry_units, "Reintento por página")

            data = _postprocess_output(str(data))

            status("Guardando TXT...")
            out_path = Path(outdir) / f"{base}_{ts}.txt"
            try:
                out_path.write_text(str(data).strip() + "\n", encoding="utf-8")
            except Exception:
                if requested_outdir:
                    raise SystemExit(f"ERROR: No se pudo guardar el TXT en outdir solicitado: {requested_outdir}")
                # fallback: carpeta del archivo de entrada; último recurso TEMP
                outdir = _ensure_writable_outdir_with_file_fallback("", args.files[0])
                out_path = Path(outdir) / f"{base}_{ts}.txt"
                out_path.write_text(str(data).strip() + "\n", encoding="utf-8")
            _write_log(log_path, f"Salida generada: {out_path}")

            result["out_path"] = str(out_path)
            status("Listo")
            log(f"Generado: {out_path}")

        except SystemExit as e:
            result["error"] = str(e)
            if result.get("log_path"):
                _write_log(Path(result["log_path"]), f"ERROR: {result['error']}")
        except Exception as e:
            result["error"] = f"ERROR: {e!r}"
            if result.get("log_path"):
                _write_log(Path(result["log_path"]), f"ERROR: {result['error']}")

        if ui:
            if result["error"]:
                ui.push("STATUS:Error")
                ui.push(result["error"])
                # No cerrar automáticamente: dejar que el usuario cierre la ventana
                return
            time.sleep(0.8)
            ui.close()

    if ui:
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        ui.mainloop()
    else:
        worker()

    if result["error"]:
        print(result["error"], file=sys.stderr)
        raise SystemExit(1)

    print(result["out_path"])


if __name__ == "__main__":
    main()
