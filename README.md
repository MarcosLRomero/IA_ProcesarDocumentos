# IA_ProcesarDocumentos

Scripts para leer documentos (imágenes o PDF) y devolver JSON normalizado usando OpenAI.

## Requisitos
- Python 3.10+ (recomendado 3.11)
- API key en `OPENAI_API_KEY` (env o `.env` junto al script)

## Instalación
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Variables de entorno
Crear un archivo `.env` en la carpeta del proyecto con:
```
OPENAI_API_KEY=tu_api_key
```

## Uso básico (facturas)
```powershell
python .\lector_facturas_to_json_v5.py factura.pdf --outdir E:\temp
```

## Varias páginas (imágenes)
```powershell
python .\lector_facturas_to_json_v5.py fac1.jpg fac2.jpg --outdir E:\temp
```

## Prompt personalizado
```powershell
python .\lector_facturas_to_json_v5.py fac1.jpg fac2.jpg --prompt-file E:\DocProcesar\Prompt_211010026.txt --outdir E:\DocProcesar
```

## GUI (ventana de progreso)
```powershell
python .\lector_facturas_to_json_v5.py fac1.jpg --outdir E:\temp --gui
```

## Modo por página (mejora tablas largas)
```powershell
python .\lector_facturas_to_json_v5.py fac1.jpg fac2.jpg --outdir E:\temp --per-page
```

## Auto-ajuste (tile + per-page)
Auto-ajusta parámetros según cantidad de páginas:
- 1 página: `tile=3`, `per-page` OFF
- 2-3 páginas: `tile=4`, `per-page` ON
- 4+ páginas: `tile=5`, `per-page` ON
```powershell
python .\lector_facturas_to_json_v5.py fac1.jpg fac2.jpg --outdir E:\temp --auto
```

## Tileado por franjas (mejor OCR en tablas largas)
Requiere Pillow. Divide cada imagen en N franjas horizontales y unifica los resultados.
```powershell
python .\lector_facturas_to_json_v5.py fac1.jpg fac2.jpg --outdir E:\temp --per-page --tile 3
```

## Modelo
```powershell
python .\lector_facturas_to_json_v5.py fac1.jpg --model gpt-4.1 --outdir E:\temp
```
****************************************************************************************************
Para liquidaciones, el modelo por defecto es `gpt-4.1`. Podés cambiarlo con `--model`.
****************************************************************************************************
## Uso básico (liquidaciones de tarjetas)
Genera un archivo de texto con dos columnas: `CONCEPTO|TOTAL`.
```powershell
python .\lector_liquidaciones_to_json_v1.py liquidacion.pdf --outdir E:\temp
```

## Varias páginas (liquidaciones)
```powershell
python .\lector_liquidaciones_to_json_v1.py img1.jpg img2.jpg --outdir E:\temp
```

## Prompt personalizado (liquidaciones)
```powershell
python .\lector_liquidaciones_to_json_v1.py liquidacion.pdf --prompt-file E:\DocProcesar\Prompt_Liq.txt --outdir E:\DocProcesar
```

## GUI (liquidaciones)
```powershell
python .\lector_liquidaciones_to_json_v1.py liquidacion.pdf --outdir E:\temp --gui
```

## Modo por página (liquidaciones)
```powershell
python .\lector_liquidaciones_to_json_v1.py img1.jpg img2.jpg --outdir E:\temp --per-page
```

## Auto-ajuste (liquidaciones)
```powershell
python .\lector_liquidaciones_to_json_v1.py img1.jpg img2.jpg --outdir E:\temp --auto
```

## Notas
- `--tile` solo aplica a imágenes (JPG/PNG/WEBP). Para PDF se ignora.
- Máximo 5 archivos por ejecución.
- Para liquidaciones, la salida es `.txt` con formato `CONCEPTO|TOTAL`.
- Se valida integridad básica: suma de `ROWS.Total` vs `TOTALES.Neto gravado` (o `TOTALES.Total`). Si el desvío supera 3%, se agrega una advertencia en `meta.observaciones`.
- Si en el texto aparece "Cantidad de items: N" y se detectan menos filas, se agrega una advertencia en `meta.observaciones`.

## Troubleshooting
### Error: `No module named 'openai'`
Activá el entorno y reinstalá dependencias:
```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Error PyInstaller en Server 2012: `Failed to load Python DLL ... python311.dll`
Server 2012 no soporta Python 3.11. Rebuild con Python 3.10 x64 y usá `--onedir`.


## Empaquetado (opcional)
Con PyInstaller:
```powershell
pyinstaller --onefile --noconsole lector_facturas_to_json_v5.py
pyinstaller --onefile --noconsole lector_liquidaciones_to_json_v1.py
```
