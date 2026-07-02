# ui_tab_new_entry.py
# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — New Entry: manual form + live AI breakdown + CSV/Excel bulk upload.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import time
from datetime import date

import pandas as pd
import streamlit as st

from ai_engine import predict, pick_best_attribute
from config import ATTRIBUTES, CSV_TEMPLATE_COLUMNS, PIPELINE_STAGES
from database import add_tender, recalculate_all_probabilities
from ui_score_breakdown import render_breakdown


def render(df_master: pd.DataFrame, staff_list: list[str]) -> None:
    st.subheader("Add New Tender")
    tab_manual, tab_upload = st.tabs(["Manual Entry", "Bulk Upload (CSV / Excel)"])

    with tab_manual:
        _render_manual(df_master, staff_list)

    with tab_upload:
        _render_upload(df_master, staff_list)


# ── Manual entry ──────────────────────────────────────────────────────────────

# session_state keys used to pre-fill the form from a scanned PDF
_SS = {
    "proj":   "ne_proj",
    "cl":     "ne_cl",
    "val":    "ne_val",
    "start":  "ne_start",
    "dat":    "ne_dat",
    "method": "ne_method",
    "brand":  "ne_brand",
    "model":  "ne_model",
}


def _extract_pdf_fields(pdf_bytes: bytes) -> dict:
    """
    Pull text from every page of the PDF via pdfplumber, then apply
    regex heuristics to find common tender-document fields.
    Returns a dict with whichever keys could be found.
    """
    import io, re
    try:
        import pdfplumber
    except ImportError:
        return {}

    text = ""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
    except Exception:
        return {}

    # Fallback to OCR if extracted text is missing or extremely short (e.g. scanned PDF)
    if len(text.strip()) < 50:
        try:
            import numpy as np
            import pypdfium2 as pdfium
            import easyocr

            st.toast("📷 Scanned PDF detected. Running OCR...")
            
            pdf_doc = pdfium.PdfDocument(io.BytesIO(pdf_bytes))
            max_pages = min(len(pdf_doc), 3)
            
            reader = easyocr.Reader(['en', 'ms'], verbose=False)
            ocr_text = ""
            for i in range(max_pages):
                page = pdf_doc.get_page(i)
                pil_img = page.render(scale=2).to_pil()
                img_np = np.array(pil_img)
                results = reader.readtext(img_np)
                page_text = "\n".join([r[1] for r in results])
                ocr_text += page_text + "\n"
                
            if ocr_text.strip():
                text = ocr_text
        except Exception:
            pass

    if not text.strip():
        return {}

    return _parse_ocr_text(text)


def _parse_ocr_text(text: str) -> dict:
    """
    Shared field-extraction logic used by both _extract_pdf_fields and
    _extract_image_fields. Applies regex heuristics to raw OCR/PDF text
    and returns a dict with whichever tender fields could be found.
    """
    import re
    found: dict = {}

    lines = [l.strip() for l in text.split('\n') if l.strip()]

    # ── Project / Tender Title ────────────────────────────────────────────────
    # Keywords that can appear as labels. "tajuk" has no position restriction
    # because it appears as a column header anywhere in a table header row.
    title_keywords = [
        "tajuk projek", "tajuk", "tender", "project", "projek", "title",
        "bidding title", "sebutharga", "keterangan tawaran",
        "keterangan", "perihal", "butiran",
    ]
    # Keywords that are position-restricted to near the start of the line (≤15 chars)
    _POS_RESTRICTED = {"tender", "project", "projek", "title", "sebutharga",
                       "keterangan", "perihal", "butiran"}
    # Words that disqualify a "next line" as the actual title
    _TITLE_BLOCKLIST = [
        "agensi", "client", "budget", "nilai", "amount", "tutup",
        "deadline", "tarikh", "kod", "bidang", "cidb", "mof",
        "rm", "jumlah", "keputusan", "no perolehan",
    ]

    for i, line in enumerate(lines):
        line_lower = line.lower()
        matched_kw = None
        for kw in title_keywords:
            kw_pos = line_lower.find(kw)
            if kw_pos == -1:
                continue
            # Position-restricted keywords must appear near the start of the line
            if kw in _POS_RESTRICTED and kw_pos > 15:
                continue

            pat_same_line = r'\b' + re.escape(kw) + r'\b[\s]*[:\-\/\\|]+[\s]*([^\n]{5,120})$'
            pat_next_line = r'\b' + re.escape(kw) + r'\b[\s]*[:\-\/\\|]*$'

            m_same = re.search(pat_same_line, line_lower)
            m_next = re.search(pat_next_line, line_lower)

            if m_same:
                start_idx = m_same.start(1)
                val = line[start_idx:].strip()
                is_another_label = re.search(r'[:\-\/\\|]+[\s]*$', val) or any(val.lower().strip(': \t/\\').strip() == k for k in title_keywords)
                if not is_another_label and len(val) >= 5:
                    title_parts = [val]
                    idx = i
                    line_count = 1
                    while idx + 1 < len(lines) and line_count < 3:
                        candidate = lines[idx + 1]
                        if any(k in candidate.lower() for k in _TITLE_BLOCKLIST):
                            break
                        if re.search(r'\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}', candidate):
                            break  # stop at a date line
                        title_parts.append(candidate)
                        idx += 1
                        line_count += 1
                    found["proj"] = " ".join(title_parts)
                    matched_kw = kw
                    break
                elif is_another_label:
                    m_next = True

            if m_next and i + 1 < len(lines):
                next_line = lines[i + 1]
                if any(next_line.lower().strip(': \t/\\').strip() == k for k in title_keywords) and i + 2 < len(lines):
                    next_line = lines[i + 2]
                if not any(k in next_line.lower() for k in _TITLE_BLOCKLIST):
                    title_parts = [next_line]
                    try:
                        idx = lines.index(next_line)
                    except ValueError:
                        idx = i + 1
                    line_count = 1
                    while idx + 1 < len(lines) and line_count < 3:
                        candidate = lines[idx + 1]
                        if any(k in candidate.lower() for k in _TITLE_BLOCKLIST):
                            break
                        if re.search(r'\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}', candidate):
                            break
                        title_parts.append(candidate)
                        idx += 1
                        line_count += 1
                    found["proj"] = " ".join(title_parts)
                    matched_kw = kw
                    break
        if matched_kw:
            break

    # Reference-number row fallback: EasyOCR often reads an entire table row as
    # one line, e.g.: "1 29/06/2026 ... NETSS202600269 SEBUT HARGA KERJA-KERJA... CIDB ..."
    # Strategy: find the tender reference number, then take the text immediately
    # after it as the title (stop before Kod Bidang / CIDB / MOF markers).
    if "proj" not in found:
        _REF_PAT = r'\b([A-Z]{2,}[A-Z0-9]{4,}\d{4,})\b'  # e.g. NETSS202600269
        _COL_STOP = re.compile(
            r'\b(cidb|mof|kod\s+bidang|kod\s+petender|jumlah|keputusan|'
            r'atas\s+tallan|klik\s+untuk|tiada|g1[-\s]|1\/2|2\/2)\b',
            re.IGNORECASE,
        )
        for line in lines:
            m_ref = re.search(_REF_PAT, line)
            if not m_ref:
                continue
            after = line[m_ref.end():].strip().lstrip(',; ')
            # Trim anything from the first column-stop keyword onwards
            m_stop = _COL_STOP.search(after)
            if m_stop and m_stop.start() > 5:
                after = after[:m_stop.start()].strip()
            # Must be a meaningful length and not just a date / number
            if len(after) >= 10 and not re.match(r'^[\d\s\/\-\.,:]+$', after):
                found["proj"] = after
                break

    # ── Client / Institution ──────────────────────────────────────────────────
    client_keywords = ["agensi", "institution", "institusi", "client", "pelanggan", "kepada", "to", "jabatan", "kementerian", "universiti", "university", "hospital", "majlis", "suruhanjaya", "perbadanan", "lembaga", "pejabat"]
    for i, line in enumerate(lines):
        line_lower = line.lower()
        matched_kw = None
        for kw in client_keywords:
            kw_pos = line_lower.find(kw)
            if kw_pos == -1 or kw_pos > 15:
                continue
                
            pat_same_line = r'\b' + re.escape(kw) + r'\b[\s]*[:\-\/\\|]+[\s]*([^\n]{3,100})$'
            pat_next_line = r'\b' + re.escape(kw) + r'\b[\s]*[:\-\/\\|]*$'
            
            m_same = re.search(pat_same_line, line_lower)
            m_next = re.search(pat_next_line, line_lower)
            
            if m_same:
                start_idx = m_same.start(1)
                val = line[start_idx:].strip()
                is_another_label = re.search(r'[:\-\/\\|]+[\s]*$', val) or any(val.lower().strip(': \t/\\').strip() == k for k in client_keywords)
                if not is_another_label and len(val) >= 3:
                    found["cl"] = val
                    matched_kw = kw
                    break
                elif is_another_label:
                    m_next = True
                    
            if m_next and i + 1 < len(lines):
                next_line = lines[i + 1]
                if any(next_line.lower().strip(': \t/\\').strip() == k for k in client_keywords) and i + 2 < len(lines):
                    next_line = lines[i + 2]
                
                if not any(k in next_line.lower() for k in ["tajuk", "projek", "title", "budget", "nilai", "amount", "tutup", "deadline"]):
                    found["cl"] = next_line
                    matched_kw = kw
                    break
        if matched_kw:
            break

    # Client letterhead fallback: if no client keyword matched, check the first 5 lines for organization keywords
    if "cl" not in found:
        org_keywords = ["jabatan", "kementerian", "universiti", "university", "hospital", "majlis", "suruhanjaya", "perbadanan", "lembaga", "pejabat"]
        for line in lines[:5]:
            line_lower = line.lower()
            if any(ok in line_lower for ok in org_keywords) and not any(k in line_lower for k in ["kenyataan", "tawaran", "sebutharga", "pelawa"]):
                found["cl"] = line
                break

    # Institution-from-title fallback: if still no client, scan the project title
    # for known institution keywords or common Malaysian university/hospital abbreviations.
    if "cl" not in found and "proj" in found:
        _INST_KW = [
            "hospital", "jabatan", "kementerian", "universiti", "university",
            "majlis", "suruhanjaya", "perbadanan", "lembaga", "pejabat",
            "pusat", "institut", "kolej",
        ]
        _INST_ABBREV = re.compile(
            r'\b(UKM|HUKM|UM|UTM|UPM|USM|UITM|UniMAP|UNIMAS|UMS|UPSI|'
            r'MOH|KKM|JKR|JPS|JPN|JPJ|AADK|LHDN|EPF|KWSP|SOCSO|PERKESO|'
            r'DBKL|MBPJ|MBSJ|MBKT|MBI|MBJB|MPPG|TNB|TM|PETRONAS|FELDA|FELCRA|'
            r'RISDA|MARA|TEKUN|SME|PUNB|MIDA|MITI|MCMC|SKMM|BNM|SC|BURSA)\b'
        )
        title = found["proj"]
        # First: look for an institution keyword phrase inside the title
        for kw in _INST_KW:
            m = re.search(r'(?:' + kw + r')[^,\n]{0,60}', title, re.IGNORECASE)
            if m:
                candidate = m.group(0).strip().rstrip('.,;')
                if len(candidate) >= 5:
                    found["cl"] = candidate
                    break
        # Second: look for a known abbreviation if still no client
        if "cl" not in found:
            m = _INST_ABBREV.search(title)
            if m:
                found["cl"] = m.group(0)

    # ── Budget / Value ────────────────────────────────────────────────────────
    budget_keywords = ["jumlah harga", "harga indikatif", "anggaran", "budget", "nilai", "harga", "jumlah", "amount", "value"]
    found_val = False
    
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if any(k in line_lower for k in budget_keywords):
            m_same = re.search(r'(?:RM\s*)?([\d,\.]+)', line[line_lower.find(budget_keywords[0]):], re.IGNORECASE)
            if m_same:
                try:
                    val = float(m_same.group(1).replace(",", ""))
                    if val > 10:
                        found["val"] = val
                        found_val = True
                        break
                except ValueError:
                    pass
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                m_next = re.search(r'(?:RM\s*)?([\d,\.]+)', next_line, re.IGNORECASE)
                if m_next:
                    try:
                        val = float(m_next.group(1).replace(",", ""))
                        if val > 10:
                            found["val"] = val
                            found_val = True
                            break
                    except ValueError:
                        pass
                        
    if not found_val:
        for pat in [
            r"(?:Jumlah|Amount|Nilai|Value|Harga|Budget|Anggaran)[^\n]{0,20}RM\s*([\d,\.]+)",
            r"RM\s*([\d,\.]+)",
        ]:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                try:
                    found["val"] = float(m.group(1).replace(",", ""))
                    found_val = True
                    break
                except ValueError:
                    pass

    # ── Date parsing helpers ──────────────────────────────────────────────────
    # Map Malay month names → English so pd.to_datetime can parse them
    _MY_MONTHS = {
        "januari": "January", "februari": "February", "mac": "March",
        "april": "April", "mei": "May", "jun": "June",
        "julai": "July", "ogos": "August", "september": "September",
        "oktober": "October", "november": "November", "disember": "December",
    }

    def _normalise_month(s: str) -> str:
        """Replace Malay month names with English equivalents (case-insensitive)."""
        for my, en in _MY_MONTHS.items():
            s = re.sub(r'\b' + my + r'\b', en, s, flags=re.IGNORECASE)
        return s

    def _try_parse_date(raw: str):
        """
        Return a date object or None.
        Strips any trailing time component (e.g. '02/07/2026 17:00 PM' -> '02/07/2026')
        and handles Malay month names before parsing.
        """
        raw = raw.strip()
        # Strip trailing time: HH:MM or HH:MM AM/PM
        raw = re.sub(r'\s+\d{1,2}:\d{2}(?:\s*[APap][Mm])?\s*$', '', raw).strip()
        raw_en = _normalise_month(raw)
        try:
            return pd.to_datetime(raw_en, dayfirst=True).date()
        except Exception:
            return None

    # ── Deadline / Due Date ───────────────────────────────────────────────────
    _DEADLINE_KW = (
        r"(?:Tarikh\s+[Tt]utup\s+[Tt]awaran|Tarikh\s+[Tt]awaran\s+[Tt]utup"
        r"|Tarikh\s+[Tt]utup|Tarikh\s+[Ll]aku"
        r"|Tarikh\s+[Tt]amat|Tarikh\s+[Aa]khir|Tarikh\s+[Hh]antar"
        r"|Deadline|Due\s+Date|Closing\s+Date|Submission\s+Deadline"
        r"|Tarikh\s+[Pp]erolehan\s+[Dd]itutup|Tutup\s+Tawaran)"
    )
    # Capture date; optional time after is non-captured so _try_parse_date strips it
    _DATE_NUM  = r"(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}(?:\s+\d{1,2}:\d{2}(?:\s*[APap][Mm])?)?)"
    _DATE_WORD = r"(\d{1,2}\s+\w+\s+\d{4}(?:\s+\d{1,2}:\d{2}(?:\s*[APap][Mm])?)?)"

    for pat in [
        _DEADLINE_KW + r"[\s:]+\s*" + _DATE_NUM,
        _DEADLINE_KW + r"[\s:]+\s*" + _DATE_WORD,
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            parsed = _try_parse_date(m.group(1))
            if parsed:
                found["dat"] = parsed
                break

    # Multi-line OCR fallback: keyword on one line, date on the next
    if "dat" not in found:
        for i, line in enumerate(lines):
            if re.search(_DEADLINE_KW, line, re.IGNORECASE) and i + 1 < len(lines):
                candidate = lines[i + 1]
                m = re.search(r"(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}(?:\s+\d{1,2}:\d{2}(?:\s*[APap][Mm])?)?)", candidate)
                if m:
                    parsed = _try_parse_date(m.group(1))
                    if parsed:
                        found["dat"] = parsed
                        break

    # ── Start Date ────────────────────────────────────────────────────────────
    _START_KW = (
        r"(?:Tarikh\s+[Mm]ula\s+[Tt]awaran|Tarikh\s+[Tt]awaran\s+[Dd]ibuka"
        r"|Tarikh\s+[Mm]ula|Tarikh\s+[Ii]klan|Tarikh\s+[Bb]uka"
        r"|Tarikh\s+[Dd]ibuka|Start\s+Date|Starting\s+Date|Opening\s+Date"
        r"|Tarikh\s+[Dd]iiklankan)"
    )
    for pat in [
        _START_KW + r"[\s:]+\s*" + _DATE_NUM,
        _START_KW + r"[\s:]+\s*" + _DATE_WORD,
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            parsed = _try_parse_date(m.group(1))
            if parsed:
                found["start"] = parsed
                break

    # Multi-line OCR fallback for start date
    if "start" not in found:
        for i, line in enumerate(lines):
            if re.search(_START_KW, line, re.IGNORECASE) and i + 1 < len(lines):
                candidate = lines[i + 1]
                m = re.search(r"(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}(?:\s+\d{1,2}:\d{2}(?:\s*[APap][Mm])?)?)", candidate)
                if m:
                    parsed = _try_parse_date(m.group(1))
                    if parsed:
                        found["start"] = parsed
                        break

    # ── Tabular / portal fallback ─────────────────────────────────────────────
    # For portal screenshots (e.g. ePerolehan) where dates sit in table columns
    # without keyword labels, collect ALL dates in the text and infer:
    #   earliest date = start/iklan date
    #   latest date   = closing/tutup date
    # Lines containing TARIKH/MASA or lawatan tapak are skipped so site-visit
    # dates don't corrupt the start/closing assignment.
    if "dat" not in found or "start" not in found:
        _SKIP_KW = r"(?:tarikh.?masa|lawatan\s+tapak|site\s+visit)"
        all_dates = []
        for line in lines:
            if re.search(_SKIP_KW, line, re.IGNORECASE):
                continue
            for m in re.finditer(
                r"\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{4})\b",
                line,
            ):
                parsed = _try_parse_date(m.group(1))
                if parsed:
                    all_dates.append(parsed)
        all_dates = sorted(set(all_dates))

        if len(all_dates) >= 2:
            if "start" not in found:
                found["start"] = all_dates[0]   # earliest = iklan / start date
            if "dat" not in found:
                found["dat"] = all_dates[-1]     # latest   = tutup / closing date
        elif len(all_dates) == 1:
            if "dat" not in found:
                found["dat"] = all_dates[0]

    # ── Submission Method ─────────────────────────────────────────────────────
    text_lower = text.lower()
    if "online" in text_lower or "eperolehan" in text_lower or "e-perolehan" in text_lower:
        found["method"] = "Online Bidding"
    elif "email" in text_lower or "e-mail" in text_lower:
        found["method"] = "Email"
    elif "courier" in text_lower or "pos" in text_lower:
        found["method"] = "Hardcopy by Courier"
    elif "hand" in text_lower or "tangan" in text_lower or "kaunter" in text_lower:
        found["method"] = "Hardcopy by Hand"

    return found


def _extract_image_fields(img_bytes: bytes) -> dict:
    """
    Enhanced image OCR pipeline for PNG/JPG/TIFF/BMP/WEBP uploads.

    Improvements over the basic version
    ─────────────────────────────────────
    1. Pre-processing
       • Upscale images narrower than 1 400 px (OCR needs ~300 DPI equivalent).
       • Boost contrast (1.6×) and sharpness (2.0×) so faint/blurry text is
         readable.
       • Convert to greyscale for the sharpening pass, then back to RGB so
         EasyOCR still has colour context.

    2. OCR with confidence gating
       • Runs EasyOCR with detail=1 (returns bounding box + confidence score).
       • Drops any token with confidence < 0.25 (reduces garbage characters).

    3. Position-aware text reconstruction
       • Groups tokens into logical rows based on vertical proximity
         (adaptive threshold = image_height ÷ 60, minimum 12 px).
       • Within each row, tokens are sorted left→right by their X coordinate.
       • Rows are emitted top→bottom as separate lines.
       This means a 3-column table becomes readable left-to-right text instead
       of a random dump of column headers followed by values.

    4. Raw-text debug expander
       • Shows the reconstructed text inside a collapsed expander so users can
         verify what OCR saw if fields are not filled correctly.
    """
    try:
        import io
        import numpy as np
        from PIL import Image, ImageEnhance, ImageFilter
        import easyocr
    except ImportError:
        st.warning("⚠️ Missing libraries for image OCR. Install: pillow easyocr numpy")
        return {}

    try:
        # ── 1. Load & pre-process ─────────────────────────────────────────────
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h = img.size

        # Upscale if too small (OCR accuracy drops significantly below ~150 DPI)
        if w < 1400:
            scale = max(2, 1400 // max(w, 1))
            img = img.resize((w * scale, h * scale), Image.LANCZOS)
            w, h = img.size

        # Greyscale sharpening pass → back to RGB
        grey = img.convert("L")
        grey = ImageEnhance.Contrast(grey).enhance(1.6)
        grey = grey.filter(ImageFilter.SHARPEN)
        grey = ImageEnhance.Sharpness(grey).enhance(2.0)
        img = grey.convert("RGB")

        img_np = np.array(img)

        # ── 2. OCR ────────────────────────────────────────────────────────────
        with st.spinner("📷 Running OCR on image…"):
            reader = easyocr.Reader(['en', 'ms'], verbose=False)
            results = reader.readtext(img_np, detail=1, paragraph=False)

        # Drop low-confidence tokens (garbage, partial characters)
        results = [(bbox, txt, conf) for bbox, txt, conf in results if conf >= 0.25]

        if not results:
            st.warning("⚠️ OCR found no readable text in this image.")
            return {}

        # ── 3. Position-aware text reconstruction ─────────────────────────────
        # bbox format: [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
        # Sort tokens top→bottom by their top-left Y coordinate
        results.sort(key=lambda r: r[0][0][1])

        # Adaptive row-grouping threshold (tokens within this many px share a row)
        line_thresh = max(12, h // 60)

        rows: list[list] = []
        current_row = [results[0]]
        for r in results[1:]:
            y_top = r[0][0][1]
            row_y  = current_row[0][0][0][1]
            if abs(y_top - row_y) <= line_thresh:
                current_row.append(r)
            else:
                rows.append(current_row)
                current_row = [r]
        rows.append(current_row)

        # Within each row sort left→right by X
        for row in rows:
            row.sort(key=lambda r: r[0][0][0])

        # Build final text (two spaces between tokens in same row → easier regex)
        lines = ["  ".join(txt for _, txt, _ in row) for row in rows]
        text  = "\n".join(lines)

    except Exception as e:
        st.warning(f"⚠️ Image OCR failed: {e}")
        return {}

    if not text.strip():
        st.warning("⚠️ OCR produced no text from this image.")
        return {}

    # ── 4. Debug expander ────────────────────────────────────────────────────
    with st.expander("🔍 OCR raw text (click to verify)", expanded=False):
        st.code(text, language=None)

    return _parse_ocr_text(text)



def _render_manual(df_master: pd.DataFrame, staff_list: list[str]) -> None:
    if not staff_list:
        st.warning("⚠️ No staff members found. Add at least one staff member in the sidebar before creating a tender.")
        return

    # ── File uploader: PDF or image ─────────────────────────────────────────────
    st.markdown("##### 📎 Upload Tender Document *(optional — auto-fills the form)*")
    st.caption("📄 PDF  ·  🖼️ Image (JPG, PNG, BMP, TIFF, WEBP)  — scanned documents are auto-OCR’d")
    s_file = st.file_uploader(
        "Drop a PDF or image to auto-extract fields",
        type=["pdf", "jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"],
        label_visibility="collapsed",
        key="ne_pdf_uploader",
    )

    if s_file is not None:
        file_bytes = s_file.read()
        sig = f"{s_file.name}_{len(file_bytes)}"
        if st.session_state.get("ne_last_pdf_sig") != sig:
            is_image = s_file.type.startswith("image/") or s_file.name.lower().split(".")[-1] in (
                "jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"
            )
            if is_image:
                spinner_msg = "📷 Running OCR on image…"
            else:
                spinner_msg = "🔍 Reading PDF and extracting fields…"
            with st.spinner(spinner_msg):
                if is_image:
                    extracted = _extract_image_fields(file_bytes)
                else:
                    extracted = _extract_pdf_fields(file_bytes)
            if extracted:
                for key, ss_key in _SS.items():
                    if key in extracted:
                        st.session_state[ss_key] = extracted[key]
                st.session_state["ne_last_pdf_sig"] = sig
                st.session_state["ne_pdf_bytes"] = file_bytes
                st.session_state["ne_pdf_name"]  = s_file.name
                filled = ", ".join(extracted.keys())
                st.success(f"✅ Auto-filled: **{filled}**. Review and adjust below.")
            else:
                st.info("ℹ️ Could not extract fields automatically — please fill the form manually.")
                st.session_state["ne_last_pdf_sig"] = sig
                st.session_state["ne_pdf_bytes"] = file_bytes
                st.session_state["ne_pdf_name"]  = s_file.name
    else:
        # Uploader cleared — wipe pre-fill cache
        for ss_key in _SS.values():
            st.session_state.pop(ss_key, None)
        st.session_state.pop("ne_last_pdf_sig", None)
        st.session_state.pop("ne_pdf_bytes", None)
        st.session_state.pop("ne_pdf_name",  None)

    st.markdown("---")

    with st.container(border=True):
        col_form, col_ai = st.columns([3, 2], gap="large")

        with col_form:
            st.markdown("#### Tender Details")
            c1, c2 = st.columns(2)

            # ── Pre-fill helpers ──────────────────────────────────────────────
            def _str(key, default=""):
                return st.session_state.get(_SS[key], default)
            def _date(key, default=None):
                v = st.session_state.get(_SS[key])
                return v if isinstance(v, date) else (default or date.today())
            def _num(key, default=0):
                return float(st.session_state.get(_SS[key], default))

            METHODS = ["Online Bidding", "Email", "Hardcopy by Hand", "Hardcopy by Courier"]
            def _method_idx():
                m = st.session_state.get(_SS["method"], "")
                return METHODS.index(m) if m in METHODS else 0

            s_proj   = c1.text_input("Project Name *",  value=_str("proj"))
            s_cl     = c1.text_input("Client Name *",   value=_str("cl"))
            s_start  = c1.date_input("Start Date",      value=_date("start"))
            s_dat    = c1.date_input("Deadline *",      value=_date("dat"))
            s_stat   = c1.selectbox("Initial Status",   PIPELINE_STAGES, index=0)
            s_method = c1.selectbox("Submission Method", METHODS, index=_method_idx())

            s_stf   = c2.selectbox("Assigned Lead *", staff_list)
            s_val   = c2.number_input("Estimated Budget (RM) *", min_value=0, step=1000, value=int(_num("val")))
            s_brand = c2.text_input("Product Brand",  value=_str("brand"))
            s_model = c2.text_input("Product Model",  value=_str("model"))

            # ── Key Driver: Auto (AI) or Manual (user) ────────────────────────────
            driver_mode = c2.radio(
                "Key Driver Selection",
                ["🤖 Auto (AI)", "✏️ Manual"],
                horizontal=True,
                help="Auto lets the AI pick the best attribute; Manual lets you override.",
            )

            s_fac    = ATTRIBUTES[0]
            best_res = None

            if driver_mode == "✏️ Manual":
                s_fac = c2.selectbox("Key Driver *", ATTRIBUTES)
                if s_cl and s_val > 0:
                    best_res = predict(s_val, s_cl, s_fac, s_stf, df_master, deadline=s_dat)
                    c2.caption(f"AI score for this driver: **{best_res.probability}%** · Confidence: {best_res.confidence_level}")
                else:
                    c2.caption("Fill in Client & Budget to preview AI score.")
            else:
                if s_cl and s_val > 0:
                    s_fac, best_res = pick_best_attribute(
                        s_val, s_cl, s_stf, df_master, ATTRIBUTES, deadline=s_dat
                    )
                    c2.info(f"🤖 AI selected Key Driver: **{s_fac}** · Confidence: {best_res.confidence_level}")
                else:
                    c2.info("🤖 AI will choose Key Driver once fields are filled.")

            submitted = st.button("➕ Add Tender", type="primary", use_container_width=True)
            if submitted:
                if not (s_proj and s_cl and s_val > 0):
                    st.error("Please fill in all required fields (marked with *).")
                else:
                    if not best_res:
                        best_res = predict(s_val, s_cl, s_fac, s_stf, df_master, deadline=s_dat)

                    # ── Save uploaded PDF ─────────────────────────────────────
                    saved_pdf_path = ""
                    pdf_bytes_to_save = st.session_state.get("ne_pdf_bytes")
                    pdf_name_to_save  = st.session_state.get("ne_pdf_name", "submission.pdf")
                    if pdf_bytes_to_save:
                        upload_dir = os.path.join(
                            os.path.dirname(os.path.abspath(__file__)), "uploads"
                        )
                        os.makedirs(upload_dir, exist_ok=True)
                        safe_name = pdf_name_to_save.replace(" ", "_")
                        dest = os.path.join(upload_dir, safe_name)
                        with open(dest, "wb") as f:
                            f.write(pdf_bytes_to_save)
                        saved_pdf_path = dest

                    add_tender(s_proj, s_cl, s_val, best_res.probability,
                               s_stat, s_fac, s_stf, s_dat, s_start,
                               s_method, s_brand, s_model, saved_pdf_path)
                    if best_res.probability >= 70:
                        st.success(f"✅ Tender added! 🔥 High potential (**{best_res.probability}%**). Focus your team's effort here.")
                    elif best_res.probability < 40:
                        st.warning(f"✅ Tender added, but ⚠️ low probability (**{best_res.probability}%**). Consider deprioritising.")
                    else:
                        st.success(f"✅ Tender added with a predicted win probability of **{best_res.probability}%**.")
                    # Clear pre-fill cache after successful save
                    for ss_key in list(_SS.values()) + ["ne_last_pdf_sig", "ne_pdf_bytes", "ne_pdf_name"]:
                        st.session_state.pop(ss_key, None)
                    time.sleep(1.5)
                    st.rerun()

        # ── Live AI preview panel ─────────────────────────────────────────────
        with col_ai:
            st.markdown("#### 🤖 Live AI Prediction")
            if s_proj and s_cl and s_val > 0 and best_res:
                render_breakdown(best_res, compact=True)
            else:
                st.info("Fill in the form to see the AI score breakdown.")
            st.caption(
                "⚠️ AI can make mistakes — predictions are based on historical patterns "
                "and should be used as a guide only."
            )

# ── Bulk upload ───────────────────────────────────────────────────────────────

_COLUMN_ALIASES: dict[str, list[str]] = {
    "project_name":   ["project_name",   "Project Name",  "Project",
                       "Bidding_Title",   "Bidding Title"],
    "client_name":    ["client_name",    "Client Name",   "Client",
                       "Institution",    "Institutions/University"],
    "value":          ["value",          "Value",         "Budget", "Est. Value",
                       "Amount_Value",   "Amount Value"],
    "bid_amount":     ["bid_amount",     "Bid Amount",    "My Bid Amount", "Bid",
                       "Amount_Value",   "Amount Value"],
    "primary_factor": ["primary_factor", "Primary Factor","Key Driver",
                       "Key Driver / Strategy", "Product_Model", "Product Model"],
    "assignee":       ["assignee",       "Assignee",      "Lead", "Staff",
                       "SalesPerson",    "Sales Person"],
    "deadline":       ["deadline",       "Deadline",      "Date",
                       "Due_Date",       "Due Date"],
    "status":         ["status",         "Status",        "Success"],
    "pdf_path":        ["pdf_path",        "PDF Path",      "PDF"],
}

# Gaia Sebutharga CSV: row 0 is a title, row 1 is the real header.
# These are the positional column names after reading with header=1.
_GAIA_COL_MAP = {
    "No":                        None,
    "Starting Date":             "starting_date",
    "Due Date":                  "deadline",
    "Institutions/University":   "client_name",
    "Reference No":              None,
    "Bidding Title":             "project_name",
    "SalesPerson":               "assignee",
    "Product Brand":             "product_brand",
    "Product Model":             "product_model",
    "Amount Value":              "value",
    "Submitted Date":            None,
    "Submission Method":         "submission_method",
    "Success":                   "status",
    "Winning Company":           None,
    "Remark":                    None,
}


def _is_gaia_format(df: pd.DataFrame) -> bool:
    """Return True when the first column looks like the Sebutharga title row."""
    first_col = str(df.columns[0])
    return "Tender Sebutharga" in first_col or "Sebutharga" in first_col


def _parse_gaia_csv(raw_bytes: bytes) -> pd.DataFrame:
    """
    Re-read the CSV with header on row 1, map Gaia columns to system schema,
    parse Amount Value, filter to Won/Lost only.
    """
    import io, re
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(raw_bytes), encoding=enc,
                             header=1, on_bad_lines="skip")
            break
        except UnicodeDecodeError:
            continue

    # Rename to system schema using _GAIA_COL_MAP
    rename = {old: new for old, new in _GAIA_COL_MAP.items()
              if old in df.columns and new is not None}
    df = df.rename(columns=rename)

    # Parse Amount Value → numeric float
    def _parse_rm(val):
        if pd.isna(val):
            return 0.0
        return float(re.sub(r"[^\d.]", "", str(val)) or 0)

    if "value" in df.columns:
        df["value"] = df["value"].apply(_parse_rm)

    # Map Success column → pipeline status; Untracked for blank outcomes
    if "status" in df.columns:
        def _map(v):
            if pd.isna(v):
                return "Untracked"
            s = str(v).strip().lower()
            return "Won" if s == "yes" else ("Lost" if s == "no" else "Untracked")
        df["status"] = df["status"].apply(_map)

    return df



def _render_upload(df_master: pd.DataFrame, staff_list: list[str]) -> None:
    st.markdown(
        "Upload a CSV or Excel file that matches the template. "
        "Any column not provided will use a safe default."
    )

    uploaded = st.file_uploader("Choose file", type=["csv", "xlsx", "xls"])
    if not uploaded:
        return

    st.markdown("---")
    if st.button("⬆️ Import Tenders", type="primary"):
        try:
            if uploaded.name.endswith(".csv"):
                # Try UTF-8 first; fall back to cp1252 for Windows-generated files
                raw_bytes = uploaded.read()
                for enc in ("utf-8", "cp1252", "latin-1"):
                    try:
                        import io
                        raw = pd.read_csv(io.BytesIO(raw_bytes), encoding=enc, on_bad_lines="skip")
                        break
                    except (UnicodeDecodeError, Exception):
                        continue
                else:
                    st.error("Could not decode the CSV file. Please save it as UTF-8 and try again.")
                    return

                # ── Detect Gaia Sebutharga format and re-parse properly ────────
                if _is_gaia_format(raw):
                    data = _parse_gaia_csv(raw_bytes)
                    st.info(
                        f"📋 Gaia Sebutharga format detected — "
                        f"importing **{len(data)}** closed (Won/Lost) tenders."
                    )
                else:
                    data = _normalise_columns(raw)
            else:
                raw  = pd.read_excel(uploaded)
                data = _normalise_columns(raw)

            if "project_name" not in data.columns or "client_name" not in data.columns:
                st.error(
                    f"Missing required columns. "
                    f"Found in file: {list(data.columns)}"
                )
                return

            errors: list[str] = []
            count = 0

            for i, row in data.iterrows():
                try:
                    p_name   = str(row["project_name"]).strip()
                    c_name   = str(row["client_name"]).strip()
                    val      = float(row.get("value", 0))
                    raw_fac  = row.get("primary_factor", ATTRIBUTES[0])
                    fac      = raw_fac if raw_fac in ATTRIBUTES else ATTRIBUTES[0]
                    raw_stat = row.get("status", "Qualified Lead")
                    status   = raw_stat if raw_stat in PIPELINE_STAGES else "Qualified Lead"
                    assignee = str(row.get("assignee", staff_list[0] if staff_list else "Unassigned")).strip().title()
                    s_date   = row.get("starting_date", None)
                    s_method = str(row.get("submission_method", "") or "")
                    p_brand  = str(row.get("product_brand", "") or "")
                    p_model  = str(row.get("product_model", "") or "")
                    if assignee and assignee not in staff_list:
                        from database import add_staff
                        add_staff(assignee, "Imported")
                        staff_list.append(assignee)

                    try:
                        deadline = pd.to_datetime(row.get("deadline", date.today())).date()
                    except Exception:
                        deadline = date.today()

                    try:
                        starting_date = pd.to_datetime(s_date).date() if s_date and str(s_date) not in ("nan", "None", "") else None
                    except Exception:
                        starting_date = None

                    if not p_name or not c_name or val <= 0:
                        errors.append(f"Row {i + 2}: skipped — missing name or zero value.")
                        continue

                    best_fac, best_res = pick_best_attribute(
                        val, c_name, assignee, df_master, ATTRIBUTES, deadline=deadline
                    )
                    fac = best_fac
                    result = best_res

                    inserted = add_tender(p_name, c_name, val, result.probability,
                               status, fac, assignee, deadline,
                               starting_date, s_method, p_brand, p_model)
                    if inserted:
                        count += 1
                    else:
                        errors.append(f"Row {i + 2}: skipped — duplicate entry already exists.")

                except Exception as row_err:
                    errors.append(f"Row {i + 2}: {row_err}")

            if count:
                with st.spinner("Recalculating AI scores against full dataset..."):
                    recalculate_all_probabilities(predict)
                st.success(f"✅ Successfully imported **{count}** tender(s) and recalculated all AI scores.")
            if errors:
                with st.expander(f"⚠️ {len(errors)} row(s) had issues"):
                    for e in errors:
                        st.write(e)

            time.sleep(1)
            st.rerun()

        except Exception as exc:
            st.error(f"Import failed: {exc}")


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map: dict[str, str] = {}
    for target, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                col_map[alias] = target
                break
    return df.rename(columns=col_map)
