import warnings
warnings.filterwarnings("ignore")

import re
import json
from pathlib import Path
import pdfplumber

STAMP_LINE_RE = re.compile(
    r"(Инв\.\s*№|Взам\.?\s*инв|Подп\.\s*и\s*дата|Изм\.\s*Лист|№\s*докум|"
    r"\S*15-00\.00\.000\s*\S*|ÐÝ|ÏÝ|3260|0623|28\.12\.24|42\.21\.82|"
    r"Лит\.\s*Лист|ООО\s*«АРМ»|Разраб\.|Трифонов|Дурандин|Н\.контр\.|Справ\.\s*№|Перв\.\s*примен)",
    re.IGNORECASE
)

REVERSED_GARBAGE_RE = re.compile(
    r"\b[а-яё\-]*(?:ротом|ывотроб|ыроткудер|аметсис|анзомрот|говелур|яинелварпу|"
    r"йымеялварпу|тсом|яаксечилвардиг|иксид|ыниш|яинежбансорткелэ|"
    r"ерагаб|вератаб|роталумукка|ротялунукка|овтсйортсу|едпозург|здопотург|огурд)[а-яё\-]*\b",
    re.IGNORECASE
)

ALLOWED_CHARS_RE = re.compile(
    r"[^А-Яа-яЁёA-Za-z0-9\s.,;:!?\-–—()\[\]{}\"'«»/\\+*=%<>°±×÷№@#&|_~•]"
)

AUTO_SEC_RE = re.compile(r"^(РАЗДЕЛ\s+\d+\s+.*)$", re.IGNORECASE)
AUTO_SUBSEC_RE = re.compile(r"^(\d{1,2}\.\d(?:\.\s*\d)?\.?\s+[А-ЯЁA-Z].*)$")

ELEC_SEC_RE = re.compile(
    r"^([1-8])\s+(ТЕХНИКА БЕЗОПАСНОСТИ|ОПИСАНИЕ|ЭКСПЛУАТАЦИЯ|"
    r"ОСМОТРЫ И ТЕХНИЧЕСКОЕ ОБСЛУЖИВАНИЕ|РАСХОДНЫЕ МАТЕРИАЛЫ.*|"
    r"КОНТРОЛЬНЫЙ ЛИСТ.*|ТИПОВЫЕ НЕИСПРАВНОСТИ.*|ТЕХНИЧЕСКИЕ ХАРАКТЕРИСТИКИ.*)$"
)
ELEC_SUBSEC_RE = re.compile(r"^([1-8]\.\d{1,2})\s+([А-ЯЁA-Z].*)$")

GENERIC_ROMAN_SEC_RE = re.compile(r"^([IІVХXШП]{1,4}\.\s+[А-ЯЁA-Z][А-ЯЁA-Z\s,\-]{3,60})$")
GENERIC_CAPS_SEC_RE = re.compile(r"^([А-ЯЁ]{4,}[А-ЯЁ\s,\-()]{4,55})$")

MODEL_SCAN_RE = re.compile(r"\b(П[ДГЭ])\s*(\d{1,2}[.,]\d)\b", re.IGNORECASE)
INTL_FORKLIFT_RE = re.compile(r"\b(CPCD|CPD|CPQD|FD|FB|FE)\s*[-()]?\s*(\d{2}[A-ZА-Я]?)\b", re.IGNORECASE)

REGISTRY_PATH = "data/models_registry.json"


def normalize_latin_suffix(code: str) -> str:
    cyr_to_lat = str.maketrans({"А": "A", "В": "B", "С": "C", "Е": "E", "М": "M", "Т": "T", "О": "O"})
    return code.translate(cyr_to_lat)


def clean_special_symbols(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\S*15-00\.00\.000\s*\S*", "", text)
    text = text.replace("ÐÝ", "").replace("ÏÝ", "")
    # убираем задвоенные буквы жирного шрифта из сканов типа ООССТТООРРООЖЖННОО -> ОСТОРОЖНО
    text = re.sub(r"ООССТТООРРООЖЖННОО", "ОСТОРОЖНО", text)
    text = re.sub(r"ВВННИИММААННИИЕЕ", "ВНИМАНИЕ", text)
    text = re.sub(r"йымеялва[а-яё\s]*УМ", "Управляемый мост", text, flags=re.IGNORECASE)
    text = re.sub(r"тсом\s*йищудев", "Ведущий мост", text, flags=re.IGNORECASE)
    text = REVERSED_GARBAGE_RE.sub("", text)
    text = ALLOWED_CHARS_RE.sub("•", text)
    text = re.sub(r"(?:•\s*){2,}", "• ", text)
    return text


def dehyphenate_cell(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"([а-яёА-ЯЁ]{2,})-\s*\n\s*([а-яё]{2,})", r"\1\2", text)
    text = text.replace("\n", " ")
    text = re.sub(r"([а-яёА-ЯЁ]{2,})-\s+([а-яё]{2,})", r"\1\2", text)
    text = re.sub(r"\bПрове\s*ве\s*рить\b", "Проверить", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_elec_to_system_by_row(row_num_str: str) -> str:
    m = re.match(r"^(\d{1,2})$", row_num_str.strip())
    if not m:
        return ""
    n = int(m.group(1))
    if 1 <= n <= 3:
        return "Бортовые мотор-редукторы"
    if 4 <= n <= 8:
        return "Тормозная система"
    if 9 <= n <= 11:
        return "Система рулевого управления"
    if 12 <= n <= 16:
        return "Управляемый мост"
    if 17 <= n <= 22:
        return "Гидравлическая система"
    if 23 <= n <= 25:
        return "Шины и диски"
    if 26 <= n <= 30:
        return "Система электроснабжения"
    if 31 <= n <= 35:
        return "Аккумуляторная батарея"
    if 36 <= n <= 43:
        return "Грузоподъемное устройство"
    if 44 <= n <= 46:
        return "Другое"
    return ""


def sort_model_key(code: str):
    p = code[:2].upper()
    order = {"ПД": 1, "ПГ": 2, "ПЭ": 3}.get(p, 9)
    try:
        t = float(code[2:].replace(",", "."))
    except ValueError:
        t = 0.0
    return (order, t, code)


def extract_models_from_pdf_pages(pdf) -> list[str]:
    found_codes = set()
    
    for i in range(min(len(pdf.pages), 5)):
        txt = pdf.pages[i].extract_text() or ""
        for match in MODEL_SCAN_RE.finditer(txt):
            prefix = match.group(1).upper()
            tonnage = match.group(2).replace(".", ",")
            if tonnage in ("1,5", "1,8", "2,0", "2,5", "3,0", "3,5", "4,0", "5,0", "6,0", "7,0", "8,0", "10,0"):
                found_codes.add(f"{prefix}{tonnage}")

    if found_codes:
        return sorted(found_codes, key=sort_model_key)

    first_txt = (pdf.pages[0].extract_text() or "").strip() if len(pdf.pages) > 0 else ""
    cover_brand = ""

    if "GROS" in first_txt.upper():
        cover_brand = "GROS"
        found_codes.add("GROS 1.0-3.0т (электро)")
    else:
        skip_words = ("РУКОВОДСТВО", "ЗАВОД", "ОБЩЕСТВО", "ЭКСПЛУАТАЦ", "АВТОМОБИЛ", "ПОГРУЗЧИК", "SSS PLUS", "SS5PLUS", "ВОДИТЕЛЯ")
        for line in first_txt.split("\n"):
            s = line.strip()
            if not (3 <= len(s) <= 35):
                continue
            if any(w in s.upper() for w in skip_words):
                continue
            if re.search(r"[A-ZА-ЯЁ]{2,}\s*[A-ZА-ЯЁ0-9\-]{2,}", s):
                found_codes.add(s)

    for page in pdf.pages:
        p_txt = page.extract_text() or ""
        for m in INTL_FORKLIFT_RE.finditer(p_txt):
            series = normalize_latin_suffix(m.group(1).upper())
            num = normalize_latin_suffix(m.group(2).upper())
            label = f"{cover_brand} {series}{num}".strip() if cover_brand else f"{series}{num}"
            found_codes.add(label)
        for m_grp in re.finditer(r"\b(CPD)\s*\(\s*(\d{2}[A-ZА-Я])\s+(\d{2}[A-ZА-Я])\s*\)", p_txt, re.IGNORECASE):
            s_pref = normalize_latin_suffix(m_grp.group(1).upper())
            for raw_n in (m_grp.group(2).upper(), m_grp.group(3).upper()):
                n_part = normalize_latin_suffix(raw_n)
                label = f"{cover_brand} {s_pref}{n_part}".strip() if cover_brand else f"{s_pref}{n_part}"
                found_codes.add(label)

    return sorted(found_codes, key=sort_model_key)


def format_model_label(code: str) -> str:
    if code == "ПД5,0":
        return "ПД5,0 (дизель ММЗ Д-243)"
    if code.startswith("ПД"):
        return f"{code} (дизель)"
    if code.startswith("ПГ"):
        return f"{code} (газ/бензин)"
    if code.startswith("ПЭ"):
        return f"{code} (электро)"
    return code


def detect_doc_info(filename: str, first_page_text: str, extracted_models: list[str]) -> dict:
    name_low = filename.lower()
    first_low = first_page_text.lower()
    models_str = "; ".join(extracted_models) if extracted_models else "уточняется по тексту документа"

    is_silant_pe = (
        "пэ15" in name_low
        or "п315" in name_low
        or any(m.startswith("ПЭ") for m in extracted_models)
    )
    is_silant_pd_pg = (
        any(m.startswith(("ПД", "ПГ")) for m in extracted_models)
        or ("автопогрузчик" in name_low and "чзса" in first_low)
    )

    if is_silant_pe:
        return {
            "doc_id": f"doc_{Path(filename).stem}",
            "doc_title": "Руководство по эксплуатации электропогрузчиков ПЭ1,5; ПЭ1,8; ПЭ2,0; ПЭ2,5 (ПЭ15-00.00.000 РЭ)",
            "family": "электропогрузчики СИЛАНТ (ПЭ)",
            "base_models": models_str,
            "model_codes_str": "; ".join(extracted_models),
            "doc_type": "elec"
        }
    elif is_silant_pd_pg:
        return {
            "doc_id": f"doc_{Path(filename).stem}",
            "doc_title": "Руководство по эксплуатации автопогрузчиков вилочных универсальных ПД1,5–3,5; ПГ1,5–3,5; ПД5,0",
            "family": "автопогрузчики СИЛАНТ с ДВС (дизельные ПД и газобензиновые ПГ)",
            "base_models": models_str,
            "model_codes_str": "; ".join(extracted_models),
            "doc_type": "auto"
        }
    else:
        title_model = models_str if extracted_models else Path(filename).stem
        return {
            "doc_id": f"doc_{Path(filename).stem}",
            "doc_title": f"Руководство по эксплуатации: {title_model}",
            "family": f"дополнительная техника ({title_model})",
            "base_models": models_str,
            "model_codes_str": "; ".join(extracted_models),
            "doc_type": "generic"
        }


def is_stamp_or_fake_table(table: list) -> bool:
    if not table or len(table) < 2:
        return True
    flat_text = " ".join(str(cell) for row in table if row for cell in row if cell)
    if "Изм." in flat_text and " докум" in flat_text:
        return True
    if ("Инв. №" in flat_text or "15-00.00.000" in flat_text) and len(table) <= 6:
        return True

    meaningful_cols_per_row = []
    for row in table:
        if not row:
            continue
        cnt = sum(1 for c in row if c and len(str(c).strip()) > 1)
        meaningful_cols_per_row.append(cnt)

    if not meaningful_cols_per_row or max(meaningful_cols_per_row) < 2:
        return True

    return False


def format_table_to_md(table: list, is_elec_to_sheet: bool = False, auto_to_mode: str = "", is_gsm_section: bool = False) -> str:
    if is_stamp_or_fake_table(table):
        return ""
    
    cleaned_rows = []
    for row in table:
        if not row:
            continue
        cells = []
        for cell in row:
            if cell is None:
                cells.append("")
            else:
                c_text = dehyphenate_cell(clean_special_symbols(str(cell)))
                cells.append(c_text)
        
        row_str = " ".join(cells).strip()
        if not row_str or STAMP_LINE_RE.search(row_str):
            continue

        if is_elec_to_sheet and len(cells) >= 3:
            sys_name = get_elec_to_system_by_row(cells[0])
            if sys_name:
                cells[1] = sys_name

        cleaned_rows.append(cells)

    if len(cleaned_rows) < 2:
        return ""

    if is_elec_to_sheet:
        data_rows = [r for r in cleaned_rows if re.match(r"^\d{1,2}$", r[0].strip())]
        if data_rows:
            header = ["№", "Система", "Вид работы", "ЕТО (50 м/ч)", "ТО-0 (100* м/ч)", "ТО-1 (250 м/ч)", "ТО-2 (1000 м/ч)", "ТО-3 (2000 м/ч)", "ТО-4 (4000 м/ч)"]
            md_lines = [
                "| " + " | ".join(header) + " |",
                "| " + " | ".join(["---"] * len(header)) + " |"
            ]
            for r in data_rows:
                row_fixed = (r + [""] * len(header))[:len(header)]
                md_lines.append("| " + " | ".join(row_fixed) + " |")
            return "\n".join(md_lines)

    if auto_to_mode in ("pd50", "base") and max(len(r) for r in cleaned_rows) >= 8:
        if auto_to_mode == "pd50":
            header = [
                "Узел / Операция", "Ежедневно", "Каждые 50 ч", "Первые 50 ч", "Первые 100 ч",
                "ТО-1 (125 ч)", "2ТО-1 (250 ч)", "ТО-2 (500 ч)", "750 ч", "ТО-3 (1000 ч)",
                "1500 ч", "2000 ч (1 год)", "2ТО-3 (4000 ч)"
            ]
        else:
            header = [
                "Узел / Операция", "Ежедневно", "Каждые 50 ч", "Первые 50 ч", "Первые 100 ч",
                "ТО-1 (100 ч)", "2ТО-1 (200 ч)", "ТО-2 (400 ч)", "800 ч", "ТО-3 (1000 ч)",
                "1400 ч", "2000 ч (1 год)", "2ТО-3 (4000 ч)"
            ]

        data_rows = []
        for r in cleaned_rows:
            first_col = r[0].lower()
            row_joined = " ".join(r).lower()
            if any(k in first_col for k in ("операции", "перед", "началом", "работы")):
                continue
            if "сервисное то" in row_joined or ("то-1" in row_joined and "то-2" in row_joined):
                continue
            if "каждые" in row_joined and ("200" in row_joined or "250" in row_joined):
                continue
            data_rows.append(r)

        if data_rows:
            md_lines = [
                "| " + " | ".join(header) + " |",
                "| " + " | ".join(["---"] * len(header)) + " |"
            ]
            for r in data_rows:
                row_fixed = (r + [""] * len(header))[:len(header)]
                md_lines.append("| " + " | ".join(row_fixed) + " |")
            return "\n".join(md_lines)

    max_cols = max(len(r) for r in cleaned_rows)
    padded_rows = [r + [""] * (max_cols - len(r)) for r in cleaned_rows]

    title_prefix = ""
    first_non_empty = [c for c in padded_rows[0] if c]
    if len(padded_rows) >= 3 and len(first_non_empty) == 1 and len(first_non_empty[0]) > 18:
        title_prefix = f"**{first_non_empty[0]}**\n\n"
        padded_rows = padded_rows[1:]

    first_row_low = " ".join(padded_rows[0]).lower()
    if is_gsm_section and max_cols == 4 and "система" not in first_row_low and "наименование" not in first_row_low and "температура" not in first_row_low:
        header = ["Система", "Рабочая жидкость", "Объем, л", "Примечание / ДВС"]
        start_idx = 0
    else:
        header = padded_rows[0]
        start_idx = 1
        non_empty_in_first = sum(1 for c in header if c)
        if len(padded_rows) >= 3 and non_empty_in_first <= max_cols // 2:
            second = padded_rows[1]
            merged_header = []
            for c1, c2 in zip(header, second):
                comb = f"{c1} {c2}".strip()
                merged_header.append(comb if comb else "-")
            header = merged_header
            start_idx = 2

    header = [h if h else f"Кол. {idx+1}" for idx, h in enumerate(header)]

    md_lines = []
    md_lines.append("| " + " | ".join(header) + " |")
    md_lines.append("| " + " | ".join(["---"] * max_cols) + " |")
    
    for row in padded_rows[start_idx:]:
        md_lines.append("| " + " | ".join(row[:max_cols]) + " |")

    return title_prefix + "\n".join(md_lines)


def extract_non_table_text(page) -> tuple[str, list]:
    found_tables = page.find_tables() or []
    valid_bboxes = []
    raw_tables = []

    for ft in found_tables:
        t_data = ft.extract()
        if not is_stamp_or_fake_table(t_data):
            valid_bboxes.append(ft.bbox)
            raw_tables.append(t_data)

    if not valid_bboxes:
        return (page.extract_text() or ""), []

    def is_outside_tables(obj):
        if obj.get("object_type") not in ("char", "word"):
            return True
        cx = (obj.get("x0", 0) + obj.get("x1", 0)) / 2.0
        cy = (obj.get("top", 0) + obj.get("bottom", 0)) / 2.0
        for (x0, top, x1, bottom) in valid_bboxes:
            if (x0 - 2) <= cx <= (x1 + 2) and (top - 2) <= cy <= (bottom + 2):
                return False
        return True

    filtered_page = page.filter(is_outside_tables)
    outside_text = filtered_page.extract_text() or ""
    return outside_text, raw_tables


def clean_text_and_track_sections(raw_text: str, doc_type: str, state: dict, page_num: int) -> str:
    if not raw_text:
        return ""

    raw_text = clean_special_symbols(raw_text)

    if doc_type == "elec" and page_num in (2, 3):
        state["section"] = "Оглавление"
        state["subsection"] = ""

    if doc_type == "auto" and "РАЗДЕЛ 10" in state.get("section", ""):
        if "Для погрузчиков ПД5,0" in raw_text or ("125" in raw_text and "250" in raw_text and "750" in raw_text):
            state["to_table_target"] = "pd50"
        elif "ПД1,5/ПГ1,5" in raw_text or ("200" in raw_text and "400" in raw_text and "1400" in raw_text):
            state["to_table_target"] = "base"

    clean_lines = []
    for line in raw_text.split("\n"):
        s = re.sub(r"\s+", " ", line).strip()
        if not s:
            continue
        
        if STAMP_LINE_RE.search(s):
            continue
        if s in ("SSS PLUS", "SS5PLUS"):
            continue
        if re.match(r"^(Лист\s*)?\d{1,3}$", s, re.IGNORECASE):
            continue

        if doc_type == "auto":
            m_sec = AUTO_SEC_RE.match(s)
            if m_sec:
                state["section"] = m_sec.group(1)
                state["subsection"] = ""
                if "РАЗДЕЛ 10" not in state["section"]:
                    state["to_table_target"] = ""
            else:
                m_sub = AUTO_SUBSEC_RE.match(s)
                if m_sub:
                    state["subsection"] = m_sub.group(1)
                elif s.startswith("Каждые ") and "моточасов" in s.lower():
                    state["subsection"] = s
                elif state["section"].startswith("РАЗДЕЛ 7") and len(s) < 45 and s[0].isupper() and not s.endswith("."):
                    state["subsection"] = s

        elif doc_type == "elec" and page_num > 3:
            m_sec = ELEC_SEC_RE.match(s)
            if m_sec:
                state["section"] = f"Раздел {m_sec.group(1)}. {m_sec.group(2)}"
                state["subsection"] = ""
            else:
                m_sub = ELEC_SUBSEC_RE.match(s)
                if m_sub:
                    state["subsection"] = f"Пункт {m_sub.group(1)} {m_sub.group(2)}"
                elif "4.3" in state["subsection"] and s in (
                    "Тяговая аккумуляторная батарея", "Свинцово-кислотная батарея",
                    "Литий-ионная батарея", "Колеса и шины", "Замена колес ведущего моста",
                    "Замена колес управляемого моста", "Гидравлическая система",
                    "Долив гидравлической жидкости и замена фильтров", "Ходовая система",
                    "Грузоподъемное устройство", "Проверка состояния цепей",
                    "Проверка состояния вил", "Тормозная система", "Электрооборудование",
                    "Контроллеры управления электромоторами"
                ):
                    state["subsection"] = f"Пункт 4.3 ({s})"

        elif doc_type == "generic":
            m_rom = GENERIC_ROMAN_SEC_RE.match(s)
            if m_rom:
                state["section"] = m_rom.group(1)
                state["subsection"] = ""
            else:
                m_num = re.match(r"^(\d{1,2}\.\s+[А-ЯЁA-Z][А-ЯЁа-яёA-Za-z0-9\s,\-()]{4,75})$", s)
                if m_num:
                    state["subsection"] = m_num.group(1)
                else:
                    m_paren = re.match(r"^(\(\d{1,2}\)\.?\s+[А-ЯЁA-Z][А-ЯЁа-яёA-Za-z0-9\s,\-()]{4,65})$", s)
                    if m_paren:
                        state["subsection"] = m_paren.group(1)
                    else:
                        m_caps = GENERIC_CAPS_SEC_RE.match(s)
                        # исключаем слова-предупреждения из заголовков разделов
                        if m_caps and not any(w in s.upper() for w in ("ВНИМАНИЕ", "ПРЕДУПРЕЖД", "ПРИМЕЧАН", "ОПАСНО", "ОСТОРОЖН", "ПРОИЗВОДИТЕЛЬ")):
                            state["subsection"] = m_caps.group(1)

        clean_lines.append(s)

    return "\n".join(clean_lines)


def detect_specific_models(page_text: str, doc_info: dict, state: dict) -> str:
    if doc_info["doc_type"] in ("elec", "generic"):
        return doc_info["base_models"]

    sec = state.get("section", "")
    if "РАЗДЕЛ 11" in sec:
        return "Газобензиновые автопогрузчики ПГ1,5; ПГ1,8; ПГ2,0; ПГ2,5; ПГ3,0; ПГ3,5"
    
    if "РАЗДЕЛ 10" in sec:
        if state.get("to_table_target") == "pd50":
            return "Только дизельный автопогрузчик ПД5,0 (интервалы ТО: 125/250/500/750/1000/1500 м/ч)"
        else:
            return "Автопогрузчики ПД1,5-3,5 и ПГ1,5-3,5 (интервалы ТО: 100/200/400/800/1000/1400 м/ч)"

    if "РАЗДЕЛ 9" in sec:
        found = []
        if "ПД 1,5" in page_text or "ПД1,5" in page_text:
            found.append("ПД1,5/ПГ1,5, ПД1,8/ПГ1,8 (Табл. 6)")
        if "ПД2,0" in page_text:
            found.append("ПД2,0/ПГ2,0, ПД2,5/ПГ2,5 (Табл. 7)")
        if "ПД3,0" in page_text:
            found.append("ПД3,0/ПГ3,0, ПД3,5/ПГ3,5 (Табл. 8)")
        if "ПД5,0" in page_text or "Д-243" in page_text:
            found.append("ПД5,0 с ДВС ММЗ Д-243 (отдельная таблица заправочных объемов)")
        if found:
            return "; ".join(found)

    return doc_info["base_models"]


def parse_single_pdf(pdf_path: Path) -> tuple[list, list[str]]:
    chunks = []
    print(f"\nЧитаем файл: {pdf_path.name} ...")

    with pdfplumber.open(pdf_path) as pdf:
        first_text = pdf.pages[0].extract_text() or ""
        extracted_models = extract_models_from_pdf_pages(pdf)
        doc_info = detect_doc_info(pdf_path.name, first_text, extracted_models)
        doc_type = doc_info["doc_type"]
        
        state = {
            "section": "Введение и общие сведения",
            "subsection": "",
            "to_table_target": ""
        }
        prev_page_tail = ""

        for idx, page in enumerate(pdf.pages):
            page_num = idx + 1
            subsec_at_start = state["subsection"]

            full_raw_text = page.extract_text() or ""
            outside_raw_text, raw_tables = extract_non_table_text(page)
            cleaned_outside_text = clean_text_and_track_sections(outside_raw_text, doc_type, state, page_num)
            _ = clean_text_and_track_sections(full_raw_text, doc_type, state, page_num)

            is_elec_to = (doc_type == "elec") and "КОНТРОЛЬНЫЙ ЛИСТ" in state.get("section", "").upper()
            auto_to_mode = state.get("to_table_target", "") if (doc_type == "auto" and "РАЗДЕЛ 10" in state.get("section", "")) else ""
            is_gsm = (doc_type == "auto") and "РАЗДЕЛ 9" in state.get("section", "")

            md_tables = []
            for t in raw_tables:
                md_t = format_table_to_md(t, is_elec_to_sheet=is_elec_to, auto_to_mode=auto_to_mode, is_gsm_section=is_gsm)
                if md_t:
                    md_tables.append(md_t)

            graphics_note = ""
            if "РАЗДЕЛ 15" in state["section"] or "Нагрузочные диаграммы" in full_raw_text:
                graphics_note = (
                    "\n[ПРИМЕЧАНИЕ К ГРАФИКЕ]: На этой странице приведены графические нагрузочные диаграммы "
                    "(зависимость грузоподъемности в кг от расстояния до центра тяжести груза 500-1200 мм "
                    "и высоты подъема 4000-6500 мм). Точные кривые необходимо смотреть непосредственно на графике."
                )
            elif "Схема гидравлическая" in full_raw_text or "РАЗДЕЛ 13 Электрооборудование" in state["section"]:
                graphics_note = (
                    "\n[ПРИМЕЧАНИЕ К ГРАФИКЕ]: На этой странице расположена принципиальная схема "
                    "(гидравлическая / электрическая) с условными обозначениями элементов."
                )

            if not cleaned_outside_text and not md_tables:
                continue

            models_scope = detect_specific_models(full_raw_text, doc_info, state)
            full_section = state["section"]
            if state["subsection"]:
                if subsec_at_start and subsec_at_start != state["subsection"] and md_tables:
                    full_section += f" -> {subsec_at_start} / {state['subsection']}"
                else:
                    full_section += f" -> {state['subsection']}"

            header_block = (
                f"ДОКУМЕНТ: {doc_info['doc_title']}\n"
                f"СЕМЕЙСТВО ТЕХНИКИ: {doc_info['family']}\n"
                f"МОДЕЛИ: {models_scope}\n"
                f"РАЗДЕЛ И ПУНКТ: {full_section}\n"
                f"СТРАНИЦА (ЛИСТ): {page_num}\n"
                f"----------------------------------------\n"
            )

            parts = []
            # если на странице есть таблица, которая продолжает пункт с конца прошлой страницы (как таблица объемов гидробака на стр. 68 GROS),
            # прикрепляем последние строки предыдущей страницы, чтобы таблица не отрывалась от своего заголовка
            if md_tables and prev_page_tail:
                parts.append(f"[Начало пункта на стр. {page_num - 1}]: {prev_page_tail}")
            if md_tables:
                parts.append("\n\n".join(md_tables))
            if cleaned_outside_text:
                parts.append(cleaned_outside_text)
            if graphics_note:
                parts.append(graphics_note)

            # запоминаем последние 3 строки текста текущей страницы для связки со следующей страницей
            if cleaned_outside_text:
                tail_lines = [ln.strip() for ln in cleaned_outside_text.split("\n") if ln.strip()][-3:]
                prev_page_tail = " ".join(tail_lines)
            else:
                prev_page_tail = ""

            body_block = "\n\n".join(parts)
            full_chunk_text = header_block + body_block

            chunks.append({
                "chunk_id": f"{doc_info['doc_id']}_p{page_num}",
                "text": full_chunk_text,
                "metadata": {
                    "doc_id": str(doc_info["doc_id"]),
                    "doc_title": str(doc_info["doc_title"]),
                    "family": str(doc_info["family"]),
                    "models": str(models_scope),
                    "model_codes_str": str(doc_info["model_codes_str"]),
                    "section": str(full_section),
                    "page": int(page_num),
                    "source_file": str(pdf_path.name)
                }
            })

    if not chunks:
        print(f"[ВНИМАНИЕ] В файле {pdf_path.name} не найден текстовый слой (возможно, это скан-копия без OCR)!")
    else:
        print(f"[ОК] Извлечено {len(chunks)} чанков | Найдены модели: {', '.join(extracted_models)}")
    return chunks, extracted_models


def update_models_registry(chunks: list):
    existing_pdf_names = {p.name for p in Path("data/manuals").glob("*.pdf")}
    all_codes = set()
    files_map = {}

    for ch in chunks:
        meta = ch.get("metadata", {})
        src = meta.get("source_file", "")
        if existing_pdf_names and src not in existing_pdf_names:
            continue

        codes_str = meta.get("model_codes_str", "")
        if isinstance(codes_str, list):
            codes_str = "; ".join(str(x) for x in codes_str)

        if codes_str:
            for part in str(codes_str).split(";"):
                c = part.strip()
                if c:
                    all_codes.add(c)
                    files_map.setdefault(src, set()).add(c)
        else:
            raw_m = str(meta.get("models", ""))
            silant_matches = list(MODEL_SCAN_RE.finditer(raw_m))
            if silant_matches:
                for m in silant_matches:
                    c = f"{m.group(1).upper()}{m.group(2).replace('.', ',')}"
                    if c[2:] in ("1,5", "1,8", "2,0", "2,5", "3,0", "3,5", "4,0", "5,0", "6,0", "7,0", "8,0", "10,0"):
                        all_codes.add(c)
                        files_map.setdefault(src, set()).add(c)
            else:
                for part in raw_m.split(";"):
                    p_clean = part.strip()
                    if p_clean and len(p_clean) <= 35 and not any(w in p_clean.lower() for w in ("интервалы", "табл.", "автопогрузчик", "уточняется")):
                        all_codes.add(p_clean)
                        files_map.setdefault(src, set()).add(p_clean)

    sorted_codes = sorted(all_codes, key=sort_model_key)
    ui_labels = [format_model_label(c) for c in sorted_codes]

    registry_data = {
        "model_codes": sorted_codes,
        "ui_options": ["Не выбрана"] + ui_labels,
        "files_models": {k: sorted(list(v), key=sort_model_key) for k, v in files_map.items()}
    }

    reg_path = Path(REGISTRY_PATH)
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(reg_path, "w", encoding="utf-8") as f:
        json.dump(registry_data, f, ensure_ascii=False, indent=2)
    return registry_data


def load_models_registry() -> dict:
    reg_path = Path(REGISTRY_PATH)
    if reg_path.exists():
        with open(reg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    chunks_path = Path("data/parsed_chunks.json")
    if chunks_path.exists():
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        return update_models_registry(chunks)

    return {"model_codes": [], "ui_options": ["Не выбрана"], "files_models": {}}


def parse_all_manuals(manuals_dir: str = "data/manuals", output_json: str = "data/parsed_chunks.json") -> list:
    folder = Path(manuals_dir)
    pdf_files = sorted(folder.glob("*.pdf"))
    
    if not pdf_files:
        print(f"[ОШИБКА] В папке {manuals_dir} нет PDF-файлов!")
        update_models_registry([])
        return []

    all_chunks = []
    for pdf_file in pdf_files:
        doc_chunks, _ = parse_single_pdf(pdf_file)
        all_chunks.extend(doc_chunks)

    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    reg = update_models_registry(all_chunks)
    print(f"\n=== ГОТОВО! Всего чанков: {len(all_chunks)} | Моделей в реестре: {len(reg['model_codes'])} ===")
    return all_chunks


if __name__ == "__main__":
    parse_all_manuals()