# -*- coding: utf-8 -*-
r"""
lector_liquidaciones_to_json_v1.py

- Lee 1 a 5 páginas (JPG/PNG/WEBP o PDF).
- Llama a OpenAI y devuelve un archivo de texto con dos columnas: CONCEPTO|TOTAL.
- Modo GUI opcional (--gui): ventana simple con estado, barra, tiempo transcurrido y log.
- Importante: al finalizar OK imprime SOLO la ruta del JSON por stdout (para VB6).
  En error: sale con código != 0 y escribe el mensaje de error en stderr.

Requisitos:
  pip install openai python-dotenv

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

def _postprocess_output(text: str) -> str:
    lines_in = [ln.strip() for ln in text.splitlines() if ln.strip()]
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
    rules = [
        ("TARJETA", ["TARJETA"]),
        ("BANCO", ["BANCO"]),
        ("GASTOS", ["GASTO", "COMISION"]),
        ("IVA_CREDITO", ["IVA", "CREDITO"]),
        ("RET_IVA", ["IVA", "RET"]),
        ("RET_IIBB", ["INGRESOS", "IIBB"]),
        ("RET_GAN", ["GANANCIAS"]),
        ("OTROS", ["OTROS"]),
    ]

    out: List[str] = []
    for idx, ln in enumerate(lines):
        if "|" not in ln:
            out.append(ln)
            continue

        concept, total = ln.split("|", 1)
        concept = concept.strip()
        total = total.strip()

        if idx < len(rules):
            _, must = rules[idx]
            t = _norm_text(concept)
            if not all(k in t for k in must):
                if idx == 0:
                    concept = _ensure_keyword(concept, "TARJETA")
                elif idx == 1:
                    concept = _ensure_keyword(concept, "BANCO")
                elif idx == 2:
                    concept = _ensure_keyword(concept, "GASTO")
                elif idx == 3:
                    concept = _ensure_keyword(concept, "IVA CREDITO")
                elif idx == 4:
                    concept = _ensure_keyword(concept, "RET IVA")
                elif idx == 5:
                    concept = _ensure_keyword(concept, "IIBB")
                elif idx == 6:
                    concept = _ensure_keyword(concept, "GANANCIAS")
                elif idx == 7:
                    concept = _ensure_keyword(concept, "OTROS")

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
        description="Lector de liquidaciones -> JSON (1 a 5 páginas). Usa OPENAI_API_KEY (env o .env junto al exe/script).",
    )
    parser.add_argument("files", nargs="*", help="1 a 5 archivos (imágenes/PDF) en orden de páginas")
    parser.add_argument("--outdir", default="", help="Carpeta de salida. Default: TEMP del sistema")
    parser.add_argument("--prompt-file", default="", help="Archivo .txt con prompt personalizado")
    parser.add_argument("--model", default="gpt-4.1", help="Modelo a usar (default: gpt-4.1)")
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
                raise SystemExit("ERROR: Debés pasar 1 a 5 archivos por parámetro.")
            if len(args.files) > 5:
                raise SystemExit("ERROR: Máximo 5 archivos.")

            if args.tile < 1 or args.tile > 6:
                raise SystemExit("ERROR: --tile debe ser un entero entre 1 y 6.")

            # Auto-ajuste simple según cantidad de páginas
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
            base = safe_basename(args.files[0])
            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            outdir = _ensure_writable_outdir(args.outdir.strip())
            log_path = Path(outdir) / f"{base}_{ts}_log.txt"
            result["log_path"] = str(log_path)
            _write_log(log_path, f"Inicio proceso. Archivos: {', '.join(args.files)}")

            status("Cargando prompt...")
            prompt = read_prompt(args.prompt_file.strip() or None)
            if "concepto|total" not in prompt.lower():
                prompt = "Respondé solo con texto en formato CONCEPTO|TOTAL.\n" + prompt
            _write_log(log_path, f"Modelo: {args.model} | per-page: {args.per_page} | tile: {args.tile}")

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

            if args.per_page and len(args.files) > 1:
                page_results: List[str] = []
                total_files = len(args.files)
                t_pages_start = time.time()
                for i, f in enumerate(args.files, start=1):
                    if i > 1:
                        elapsed = time.time() - t_pages_start
                        avg = elapsed / (i - 1)
                        remaining = avg * (total_files - i + 1)
                        mm = int(remaining // 60)
                        ss = int(remaining % 60)
                        status(f"IA por página {i}/{total_files}... (ETA ~{mm:02d}:{ss:02d})")
                    else:
                        status(f"IA por página {i}/{total_files}...")
                    log(f"Archivo: {f}")
                    page_content = [{"type": "input_text", "text": prompt}]
                    page_content.extend(file_to_content_blocks(f, args.tile))
                    page_results.append(call_model(page_content))
                # unificar: concatenar salidas con separación de línea
                data = "\n".join([t for t in page_results if t.strip()])
            else:
                data = call_model(content)

            data = _postprocess_output(str(data))

            status("Guardando TXT...")
            out_path = Path(outdir) / f"{base}_{ts}.txt"
            try:
                out_path.write_text(str(data).strip() + "\n", encoding="utf-8")
            except Exception:
                # fallback a TEMP si falla el outdir
                outdir = _ensure_writable_outdir("")
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
