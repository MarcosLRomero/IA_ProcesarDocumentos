# -*- coding: utf-8 -*-
r"""
lector_facturas_to_json_v5.py

- Lee 1 a 5 páginas (JPG/PNG/WEBP o PDF).
- Llama a OpenAI y devuelve JSON normalizado.
- Modo GUI opcional (--gui): ventana simple con estado, barra, tiempo transcurrido y log.
- Importante: al finalizar OK imprime SOLO la ruta del JSON por stdout (para VB6).
  En error: sale con código != 0 y escribe el mensaje de error en stderr.

Requisitos:
  pip install openai python-dotenv

Uso:
  python lector_facturas_to_json_v5.py factura.pdf --outdir E:\temp
  python lector_facturas_to_json_v5.py fac1.jpg fac2.jpg --outdir E:\temp --gui

EXE (PyInstaller):
  pyinstaller --onefile --noconsole lector_facturas_to_json_v5.py
"""

from __future__ import annotations

import argparse
import io
import base64
import datetime as dt
import json
import os
import re
import sys
import tempfile
import threading
import time
import queue
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


class StatusUI:
    """Ventana simple: estado + barra indeterminada + tiempo + log.
    NO escribe en stdout (para no romper VB6).
    """

    def __init__(self, title="Procesando factura...", width=560, height=260):
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
    return name or "factura"

def sanitize_json_text(s: str) -> str:
    s = s.strip()
    if s.startswith("﻿"):
        s = s.lstrip("﻿")
    # remove trailing commas before } or ]
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s



def extract_first_json(text: str) -> dict:
    """Extrae el primer JSON v?lido del texto (tolerante a basura alrededor)."""
    if not text:
        raise ValueError("Respuesta vac?a del modelo.")

    # ya es JSON puro
    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return json.loads(sanitize_json_text(s))

    # buscar bloque entre llaves (simple y efectivo)
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


# ----------------------------
# Esquema esperado
# ----------------------------
CAB_KEYS = [
    "Proveedor",
    "CUIT",
    "Domicilio",
    "CondicionIVA",
    "TipoComprobante",
    "Letra",
    "PuntoVenta",
    "Numero",
    "Fecha",
    "Vencimiento",
    "Moneda",
    "CAE",
    "VtoCAE",
    "Observaciones",
]

ROW_KEYS = [
    "Cantidad",
    "Codigo_Articulo",
    "Descripcion",
    "UD",
    "Importe_Lista",
    "% Dto1",
    "% Dto2",
    "Importe_Neto",
    "IVA",
    "Impuestos internos",
    "Bl/Pq",
    "Moneda",
    "Total",
    "AuxNroLote",
    "AuxNroSerie",
]

TOTALES_KEYS = [
    "Neto gravado",
    "Neto no gravado",
    "Exento",
    "IVA 21%",
    "IVA 10.5%",
    "IVA 27%",
    "Otros",
    "Percepcion IIBB",
    "Percepcion Ganancias",
    "Impuestos internos",
    "Otros impuestos",
    "Total",
    "Total final",
    "Moneda",
]

META_KEYS = [
    "comprobante_raw",
    "moneda_detectada",
    "observaciones",
    "totales_raw",
    "orden_columnas",
]


def normalize_schema(data: dict) -> dict:
    data = _ensure_object(data)

    cab = _ensure_object(data.get("CAB"))
    rows = _ensure_list(data.get("ROWS"))
    tot = _ensure_object(data.get("TOTALES"))
    meta = _ensure_object(data.get("meta"))

    data["CAB"] = cab
    data["ROWS"] = rows
    data["TOTALES"] = tot
    data["meta"] = meta

    for k in CAB_KEYS:
        cab.setdefault(k, "")

    norm_rows = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        for k in ROW_KEYS:
            r.setdefault(k, "")
        norm_rows.append(r)
    data["ROWS"] = norm_rows

    for k in TOTALES_KEYS:
        tot.setdefault(k, "")

    otros = tot.get("Otros")
    if not isinstance(otros, list):
        tot["Otros"] = [{"Etiqueta": "", "Importe_Neto": ""}]
    else:
        norm_otros = []
        for it in otros:
            if not isinstance(it, dict):
                continue
            it.setdefault("Etiqueta", "")
            it.setdefault("Importe_Neto", "")
            norm_otros.append(it)
        tot["Otros"] = norm_otros or [{"Etiqueta": "", "Importe_Neto": ""}]

    for k in META_KEYS:
        if k == "orden_columnas":
            v = meta.get(k)
            if not isinstance(v, list):
                meta[k] = []
        else:
            meta.setdefault(k, "")

    return data


def infer_orden_columnas(data: dict) -> None:
    """Completa meta.orden_columnas si viene vacío.
    Intenta tomar "Detalle: ..." desde meta.comprobante_raw u observaciones.
    """
    try:
        meta = _ensure_object(data.get("meta"))
        data["meta"] = meta

        oc = meta.get("orden_columnas")
        if isinstance(oc, list) and oc:
            return

        rows = data.get("ROWS") or []
        row_keys: List[str] = []
        for r in rows:
            if isinstance(r, dict) and any(str(v).strip() for v in r.values()):
                row_keys = list(r.keys())
                break
        if not row_keys and rows and isinstance(rows[0], dict):
            row_keys = list(rows[0].keys())

        if not row_keys:
            meta["orden_columnas"] = []
            return

        key_lookup = {k.lower(): k for k in row_keys}

        def pick(*cands):
            for c in cands:
                kk = c.lower()
                if kk in key_lookup:
                    return key_lookup[kk]
            return None

        def norm(s: str) -> str:
            s = s.lower().strip()
            s = (
                s.replace("á", "a")
                .replace("é", "e")
                .replace("í", "i")
                .replace("ó", "o")
                .replace("ú", "u")
                .replace("ñ", "n")
            )
            s = re.sub(r"\s+", " ", s)
            return s

        raw = str(meta.get("comprobante_raw") or "")
        header_part = ""
        m = re.search(r"(?i)\bdetalle\s*:\s*([^\n\r]+)", raw)
        if m:
            header_part = m.group(1).strip()
        if not header_part:
            obs = str(meta.get("observaciones") or "")
            m2 = re.search(r"(?i)\bdetalle\s*:\s*([^\n\r]+)", obs)
            if m2:
                header_part = m2.group(1).strip()

        ordered: List[str] = []
        dto_seen = 0

        if header_part:
            parts = re.split(r"[,\|;]+", header_part)
            tokens = [p.strip() for p in parts if p.strip()]

            for t in tokens:
                nt = norm(t)
                k = None

                if "cant" in nt or "cantidad" in nt:
                    k = pick("Cantidad")
                elif "artic" in nt or "cod" in nt or "codigo" in nt or "producto" in nt:
                    k = pick("Codigo_Articulo")
                elif "descripcion" in nt or nt == "desc":
                    k = pick("Descripcion")
                elif nt in ("ud", "unidad", "u."):
                    k = pick("UD")
                elif "dto" in nt or "descuento" in nt or "%" in nt:
                    dto_seen += 1
                    k = pick("% Dto1" if dto_seen == 1 else "% Dto2")
                elif "lista" in nt:
                    k = pick("Importe_Lista")
                elif "neto" in nt or "precio" in nt:
                    k = pick("Importe_Neto", "Importe_Lista")
                elif "iva" in nt:
                    k = pick("IVA")
                elif "impuestos internos" in nt or "imp internos" in nt:
                    k = pick("Impuestos internos")
                elif "bl/pq" in nt or "bulto" in nt:
                    k = pick("Bl/Pq")
                elif "moneda" in nt:
                    k = pick("Moneda")
                elif "total" in nt:
                    k = pick("Total")
                elif "lote" in nt:
                    k = pick("AuxNroLote")
                elif "serie" in nt:
                    k = pick("AuxNroSerie")

                if k and k not in ordered:
                    ordered.append(k)

        if not ordered:
            common = [
                "Cantidad",
                "Codigo_Articulo",
                "Descripcion",
                "Importe_Lista",
                "% Dto1",
                "% Dto2",
                "Importe_Neto",
                "Total",
            ]
            for k in common:
                if k in row_keys and k not in ordered:
                    ordered.append(k)

        meta["orden_columnas"] = ordered if ordered else []
    except Exception:
        data.setdefault("meta", {})
        if not isinstance(data["meta"].get("orden_columnas"), list):
            data["meta"]["orden_columnas"] = []


def _is_empty_value(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip() == ""
    if isinstance(v, (list, tuple, dict)):
        return len(v) == 0
    return False

def _parse_number(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    # keep digits, comma, dot, minus
    s = re.sub(r"[^\d,.\-]", "", s)
    if not s:
        return None
    # decide decimal separator
    if "," in s and "." in s:
        # decide by last separator (decimal usually last)
        if s.rfind(".") > s.rfind(","):
            # dot decimal, comma thousands
            s = s.replace(",", "")
        else:
            # comma decimal, dot thousands
            s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None

def _extract_int(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    m = re.search(r"\d+", str(raw))
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None

def _format_number_ar(val: float, decimals: int = 3) -> str:
    s = f"{val:,.{decimals}f}"
    # python uses comma as thousands and dot as decimal -> swap
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return s

def adjust_importe_lista_for_bultos(data: dict) -> None:
    """Proveedor CAFES LA VIRGINIA: Importe_Lista = Total / (Cantidad * Bl/Pq) when mismatch.
    This guards against OCR picking UNIT BRUTO instead of U.NETO.
    """
    try:
        cab = _ensure_object(data.get("CAB"))
        proveedor = str(cab.get("Nombre") or "").upper()
        if "LA VIRGINIA" not in proveedor:
            return

        rows = _ensure_list(data.get("ROWS"))
        for r in rows:
            if not isinstance(r, dict):
                continue
            total = _parse_number(r.get("Total"))
            cant = _parse_number(r.get("Cantidad"))
            blpq = _extract_int(r.get("Bl/Pq")) or 0
            if not total or not cant or blpq <= 0:
                continue

            implied = total / (cant * blpq)
            current = _parse_number(r.get("Importe_Lista"))
            # overwrite if empty or far from implied (>2%)
            if current is None or abs(current - implied) / implied > 0.02:
                r["Importe_Lista"] = _format_number_ar(implied, decimals=3)
    except Exception:
        return

def _parse_expected_items(meta: dict) -> Optional[int]:
    for k in ("comprobante_raw", "observaciones", "totales_raw"):
        txt = str(meta.get(k) or "")
        m = re.search(r"(?i)cantidad\s+de\s+items\s*[:\-]?\s*(\d+)", txt)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return None

def validate_totals_integrity(data: dict, tolerance: float = 0.03) -> None:
    """Validate sum(ROWS.Total) against Neto gravado or Total.
    Adds warning into meta.observaciones if mismatch exceeds tolerance.
    """
    try:
        tot = _ensure_object(data.get("TOTALES"))
        meta = _ensure_object(data.get("meta"))
        rows = _ensure_list(data.get("ROWS"))

        row_sum = 0.0
        rows_count = 0
        for r in rows:
            if not isinstance(r, dict):
                continue
            v = _parse_number(r.get("Total"))
            if v is None:
                continue
            row_sum += v
            rows_count += 1

        if rows_count == 0:
            return

        target = _parse_number(tot.get("Neto gravado"))
        if target is None:
            target = _parse_number(tot.get("Total"))
        if target is None or target == 0:
            return

        diff = abs(row_sum - target) / target
        if diff > tolerance:
            msg = (
                f"ADVERTENCIA: suma de ROWS.Total ({_format_number_ar(row_sum, 2)}) "
                f"no coincide con Neto/Total ({_format_number_ar(target, 2)}). "
                f"Desvío {diff*100:.2f}%."
            )
            obs = str(meta.get("observaciones") or "")
            meta["observaciones"] = (obs + " | " if obs else "") + msg

        exp = _parse_expected_items(meta)
        if exp is not None and rows_count < exp:
            msg = f"ADVERTENCIA: filas detectadas {rows_count} < cantidad de items {exp}."
            obs = str(meta.get("observaciones") or "")
            meta["observaciones"] = (obs + " | " if obs else "") + msg

        data["meta"] = meta
    except Exception:
        return

def merge_data_keep_best(datas: List[dict]) -> dict:
    """Merge multiple page-level results into a single invoice.
    - CAB: keep first non-empty per key
    - ROWS: concat
    - TOTALES: prefer last non-empty per key
    - meta: keep first non-empty, except totales_raw (prefer last) and orden_columnas (first non-empty list)
    """
    if not datas:
        return {}

    out = {"CAB": {}, "ROWS": [], "TOTALES": {}, "meta": {}}

    for d in datas:
        cab = _ensure_object(d.get("CAB"))
        rows = _ensure_list(d.get("ROWS"))
        tot = _ensure_object(d.get("TOTALES"))
        meta = _ensure_object(d.get("meta"))

        # CAB: keep first non-empty
        for k, v in cab.items():
            if _is_empty_value(out["CAB"].get(k)) and not _is_empty_value(v):
                out["CAB"][k] = v
            elif k not in out["CAB"]:
                out["CAB"][k] = out["CAB"].get(k, v)

        # ROWS: concat
        out["ROWS"].extend(rows)

        # TOTALES: prefer last non-empty
        for k, v in tot.items():
            if not _is_empty_value(v):
                out["TOTALES"][k] = v
            elif k not in out["TOTALES"]:
                out["TOTALES"][k] = v

        # meta: first non-empty, except totales_raw (last), orden_columnas (first non-empty list)
        if "orden_columnas" in meta:
            oc = meta.get("orden_columnas")
            if isinstance(oc, list) and oc and not out["meta"].get("orden_columnas"):
                out["meta"]["orden_columnas"] = oc
        for k, v in meta.items():
            if k == "orden_columnas":
                continue
            if k == "totales_raw":
                if not _is_empty_value(v):
                    out["meta"][k] = v
                elif k not in out["meta"]:
                    out["meta"][k] = v
                continue
            if _is_empty_value(out["meta"].get(k)) and not _is_empty_value(v):
                out["meta"][k] = v
            elif k not in out["meta"]:
                out["meta"][k] = v

    return out



def dedupe_rows(rows: List[dict]) -> List[dict]:
    seen = set()
    out: List[dict] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        key = (
            str(r.get("Codigo_Articulo", "")).strip(),
            str(r.get("Descripcion", "")).strip(),
            str(r.get("Cantidad", "")).strip(),
            str(r.get("Importe_Neto", "")).strip(),
            str(r.get("Total", "")).strip(),
        )
        if not any(key):
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


# ----------------------------
# Prompt
# ----------------------------
DEFAULT_PROMPT = r"""
Vas a analizar 1 a 5 páginas de una factura / comprobante de compra.
Respondé **SOLO** con JSON válido (sin texto adicional).

El JSON debe tener ESTE formato fijo (NO elimines claves):

{
  "CAB": {
    "Proveedor": "",
    "CUIT": "",
    "Domicilio": "",
    "CondicionIVA": "",
    "TipoComprobante": "",
    "Letra": "",
    "PuntoVenta": "",
    "Numero": "",
    "Fecha": "",
    "Vencimiento": "",
    "Moneda": "",
    "CAE": "",
    "VtoCAE": "",
    "Observaciones": ""
  },
  "ROWS": [
    {
      "Cantidad": "",
      "Codigo_Articulo": "",
      "Descripcion": "",
      "UD": "",
      "Importe_Lista": "",
      "% Dto1": "",
      "% Dto2": "",
      "Importe_Neto": "",
      "IVA": "",
      "Impuestos internos": "",
      "Bl/Pq": "",
      "Moneda": "",
      "Total": "",
      "AuxNroLote": "",
      "AuxNroSerie": ""
    }
  ],
  "TOTALES": {
    "Neto gravado": "",
    "Neto no gravado": "",
    "Exento": "",
    "IVA 21%": "",
    "IVA 10.5%": "",
    "IVA 27%": "",
    "Otros": [{"Etiqueta":"","Importe_Neto":""}],
    "Percepcion IIBB": "",
    "Percepcion Ganancias": "",
    "Impuestos internos": "",
    "Otros impuestos": "",
    "Total": "",
    "Total final": "",
    "Moneda": ""
  },
  "meta": {
    "comprobante_raw": "",
    "moneda_detectada": "",
    "observaciones": "",
    "totales_raw": "",
    "orden_columnas": []
  }
}

REGLAS IMPORTANTES:
- NO inventes datos.
- Si algo no se ve o no es seguro, dejalo "".
- Respetá formato de números y fechas tal como aparece.
- Si hay varias páginas, unificá en un solo JSON final.
- Respondé SOLO JSON.
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



def file_to_content_blocks(file_path: str, tiles: int = 1) -> List[Dict[str, Any]]:
    ext = Path(file_path).suffix.lower()
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
        description="Lector de facturas -> JSON (1 a 5 páginas). Usa OPENAI_API_KEY (env o .env junto al exe/script).",
    )
    parser.add_argument("files", nargs="*", help="1 a 5 archivos (imágenes/PDF) en orden de páginas")
    parser.add_argument("--outdir", default="", help="Carpeta de salida. Default: TEMP del sistema")
    parser.add_argument("--prompt-file", default="", help="Archivo .txt con prompt personalizado")
    parser.add_argument("--model", default="gpt-4.1-mini", help="Modelo a usar (default: gpt-4.1-mini)")
    parser.add_argument("--gui", action="store_true", help="Muestra ventana de progreso (no altera stdout)")
    parser.add_argument(
        "--per-page",
        action="store_true",
        help="Procesa cada archivo/pagina por separado y luego unifica filas (mejora extraccion en tablas largas).",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-ajusta parametros (tile y per-page) segun cantidad de paginas.",
    )
    parser.add_argument(
        "--tile",
        type=int,
        default=1,
        help="Divide cada pagina en N franjas horizontales (solo imagenes). Requiere Pillow.",
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

    result = {"out_path": None, "error": None}

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
                raise SystemExit("ERROR: Debés pasar 1 a 5 archivos por parámetro.")
            if len(args.files) > 5:
                raise SystemExit("ERROR: Máximo 5 archivos.")

            if args.tile < 1 or args.tile > 6:
                raise SystemExit("ERROR: --tile debe ser un entero entre 1 y 6.")

            # Auto-ajuste simple segun cantidad de paginas
            if args.auto:
                n = len(args.files)
                if n <= 1:
                    args.tile = 3
                    args.per_page = False
                elif n <= 3:
                    args.tile = 4
                    args.per_page = True
                else:
                    args.tile = 5
                    args.per_page = True

            status("Validando archivos...")
            for f in args.files:
                if not Path(f).exists():
                    raise SystemExit(f"ERROR: No existe el archivo: {f}")

            status("Preparando salida...")
            outdir = args.outdir.strip() or tempfile.gettempdir()
            Path(outdir).mkdir(parents=True, exist_ok=True)

            status("Cargando prompt...")
            prompt = read_prompt(args.prompt_file.strip() or None)

            if 'json' not in prompt.lower():
                prompt = 'Responde solo con json.\n' + prompt

            status("Armando contenido...")
            log(f"Modelo: {args.model} | per-page: {args.per_page} | tile: {args.tile}")
            content = [{"type": "input_text", "text": prompt}]
            total_files = len(args.files)
            for i, f in enumerate(args.files, start=1):
                status(f"Adjuntando página {i}/{total_files}...")
                log(f"Archivo: {f}")
                content.extend(file_to_content_blocks(f, args.tile))

            status("Analizando con Inteligencia Artificial...")
            log("Motor IA: Activo")
            client = OpenAI(api_key=api_key)

            def call_model(content_blocks: List[Dict[str, Any]]) -> dict:
                resp = client.responses.create(
                    model=args.model,
                    max_output_tokens=16000,  # tablas largas
                    input=[{"role": "user", "content": content_blocks}],
                    text={"format": {"type": "json_object"}},
                )

                out_text = ""
                try:
                    out_text = resp.output[0].content[0].text
                except Exception:
                    # fallback: juntar textos si vinieron en partes
                    parts = []
                    for item in getattr(resp, "output", []) or []:
                        for c in getattr(item, "content", []) or []:
                            t = getattr(c, "text", None)
                            if t:
                                parts.append(t)
                    out_text = "\n".join(parts)

                try:
                    data = extract_first_json(out_text)
                except Exception as e:
                    raw_path = Path(outdir) / f"{safe_basename(args.files[0])}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_raw.txt"
                    raw_path.write_text(out_text, encoding="utf-8", errors="replace")
                    raise SystemExit(f"ERROR: No se pudo parsear JSON. Se guardo la respuesta cruda en: {raw_path}") from e

                data = normalize_schema(data)
                infer_orden_columnas(data)
                return data

            if args.per_page and len(args.files) > 1:
                page_results: List[dict] = []
                total_files = len(args.files)
                t_pages_start = time.time()
                for i, f in enumerate(args.files, start=1):
                    # ETA aproximado basado en promedio por página procesada
                    if i > 1:
                        elapsed = time.time() - t_pages_start
                        avg = elapsed / (i - 1)
                        remaining = avg * (total_files - i + 1)
                        mm = int(remaining // 60)
                        ss = int(remaining % 60)
                        status(f"IA por pagina {i}/{total_files}... (ETA ~{mm:02d}:{ss:02d})")
                    else:
                        status(f"IA por pagina {i}/{total_files}...")
                    log(f"Archivo: {f}")
                    page_content = [{"type": "input_text", "text": prompt}]
                    page_content.extend(file_to_content_blocks(f, args.tile))
                    page_results.append(call_model(page_content))
                data = merge_data_keep_best(page_results)
            else:
                data = call_model(content)

            data["ROWS"] = dedupe_rows(_ensure_list(data.get("ROWS")))
            adjust_importe_lista_for_bultos(data)
            validate_totals_integrity(data, tolerance=0.03)

            status("Guardando JSON...")
            base = safe_basename(args.files[0])
            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = Path(outdir) / f"{base}_{ts}.json"
            out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

            result["out_path"] = str(out_path)
            status("Listo ✅")
            log(f"Generado: {out_path}")

        except SystemExit as e:
            result["error"] = str(e)
        except Exception as e:
            result["error"] = f"ERROR: {e!r}"

        if ui:
            # que se llegue a ver el “Listo” o el error
            if result["error"]:
                ui.push("STATUS:Error ❌")
                ui.push(result["error"])
            time.sleep(0.8)
            ui.close()

    # Ejecutar con GUI (thread) o directo
    if ui:
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        ui.mainloop()
    else:
        worker()

    # Contrato VB6: OK -> stdout solo ruta. Error -> stderr y exit != 0
    if result["error"]:
        print(result["error"], file=sys.stderr)
        raise SystemExit(1)

    print(result["out_path"])  # SOLO la ruta


if __name__ == "__main__":
    main()
