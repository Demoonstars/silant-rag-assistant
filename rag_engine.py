import warnings
warnings.filterwarnings("ignore")

import os
import re
import json
import time
from pathlib import Path
import requests
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from indexer import get_chroma_instance, sanitize_metadata
from parser import clean_special_symbols, MODEL_SCAN_RE, INTL_FORKLIFT_RE, load_models_registry

load_dotenv(override=True)

# Словарь коротких аббревиатур для расширения поискового запроса в BM25 и ChromaDB
SHORT_QUERY_MAP = {
    "то": "Периодичность и регламент технического обслуживания (ЕТО, ТО-1, ТО-2, ТО-3, контрольный лист техобслуживания)",
    "то1": "Регламент и перечень работ первого технического обслуживания ТО-1 (контрольный лист техобслуживания)",
    "то-1": "Регламент и перечень работ первого технического обслуживания ТО-1 (контрольный лист техобслуживания)",
    "то2": "Регламент и перечень работ второго технического обслуживания ТО-2 (контрольный лист техобслуживания)",
    "то-2": "Регламент и перечень работ второго технического обслуживания ТО-2 (контрольный лист техобслуживания)",
    "то3": "Регламент и перечень работ технического обслуживания ТО-3 (контрольный лист техобслуживания)",
    "то-3": "Регламент и перечень работ технического обслуживания ТО-3 (контрольный лист техобслуживания)",
    "ето": "Ежедневный осмотр перед началом работы и ежедневное техническое обслуживание (ЕТО)",
    "масло": "Заправочные объемы, рабочие жидкости, точки смазки и рекомендованное масло",
    "гсм": "Расходные материалы, масла, смазки и рабочие жидкости (заправочные объемы)",
    "акб": "Обслуживание и зарядка аккумуляторной батареи (АКБ), плотность электролита, требования безопасности",
    "шины": "Давление воздуха в шинах, размер шин и момент затяжки колесных гаек",
    "колеса": "Замена колес ведущего и управляемого моста, момент затяжки гаек и давление в шинах",
    "ошибки": "Коды ошибок контроллеров и типовые неисправности (контрольные лампы)"
}

# Системная инструкция для генерации ответа строго по заводской документации ООО «ЧЗСА»
SYSTEM_PROMPT = """Ты - технический RAG-ассистент сервисной службы завода «СИЛАНТ» (ООО «ЧЗСА»).
Твои пользователи - механики сервисной службы и операторы вилочных погрузчиков. От точности твоих ответов зависит безопасность людей и сохранность техники.

ЖЕСТКИЕ ПРАВИЛА РАБОТЫ:

1. ТОЛЬКО ДОКУМЕНТАЦИЯ ИЗ КОНТЕКСТА:
Отвечай исключительно на основе предоставленного ниже КОНТЕКСТА из руководств по эксплуатации. Категорически запрещено использовать внешние знания, домысливать значения, моменты затяжки, марки масел или порядок действий. Не смешивай предупреждения от посторонних узлов, если они не относятся к заданному вопросу.

2. ПРИВЯЗКА К МОДЕЛИ И УЧЕТ РАЗЛИЧИЙ:
В базовой поставке два семейства погрузчиков СИЛАНТ:
- Автопогрузчики СИЛАНТ с ДВС: дизельные (ПД1,5; ПД1,8; ПД2,0; ПД2,5; ПД3,0; ПД3,5 и тяжелая модель ПД5,0 с ДВС ММЗ Д-243) и газобензиновые (ПГ1,5; ПГ1,8; ПГ2,0; ПГ2,5; ПГ3,0; ПГ3,5).
- Электропогрузчики СИЛАНТ: ПЭ1,5; ПЭ1,8; ПЭ2,0; ПЭ2,5.

Обязательно учитывай при чтении таблиц КОНТЕКСТА:
- Для основных автопогрузчиков (ПД1,5–3,5 и ПГ1,5–3,5) интервалы ТО: 100 / 200 (ТО-1) / 400 (ТО-2) / 800 / 1000 (ТО-3) / 1400 моточасов.
- Для тяжелой модели ПД5,0 (ДВС ММЗ Д-243) своя сетка ТО: 125 (ТО-1) / 250 (2ТО-1) / 500 (ТО-2) / 750 / 1000 (ТО-3) / 1500 моточасов (Раздел 6, 7 и Таблица 10 на стр. 34–38), и своя таблица заправочных объемов в Разделе 9 на стр. 27–28 (моторное масло ДВС ММЗ Д-243 SAE 10W40/15W40 API CF — 13,2 л; трансмиссия Dextron III — 14,5 л; ведущий мост — 9,2 л; гидросистема МГЕ 46В — 65 л / 55 л в баке; ОЖ-40 — 14 л).
- Для электропогрузчиков ПЭ1,5–2,5 регламент ТО указан в Разделе 4 (п. 4.3) и Разделе 6 «Контрольный лист ТО» (стр. 55–56): ЕТО — каждые 50 м/ч, ТО-0 — после первых 100 м/ч, ТО-1 — каждые 250 м/ч, ТО-2 — каждые 1000 м/ч, ТО-3 — каждые 2000 м/ч, ТО-4 — каждые 4000 м/ч.

3. СОХРАНЕНИЕ ПРОЦЕДУР И ПРЕДУПРЕЖДЕНИЙ БЕЗОПАСНОСТИ:
- Если в КОНТЕКСТЕ описана пошаговая процедура (например, замена колес ведущего моста на стр. 20–21, пуск двигателя, обслуживание ГБО на стр. 42–43, зарядка АКБ на стр. 17 и 37), приводи все шаги полностью, сохраняя последовательность.
- Обязательно включай в ответ все относящиеся к этой операции блоки «ВНИМАНИЕ», «ОПАСНО», «ОСТОРОЖНО», «ЗАПРЕЩАЕТСЯ», «ВАЖНО», а также требования к грузоподъемному оборудованию (например, грузоподъемность домкрата не менее 2/3 массы погрузчика, деревянные балки), СИЗ, вентиляции (6–8 кратный обмен воздуха, не ниже +15 °C) и допуску персонала.

4. ПОВЕДЕНИЕ ПРИ ОТСУТСТВИИ ОТВЕТА (ЧЕСТНЫЙ ОТКАЗ):
Никогда не выводи в ответе служебные слова вроде «Ситуация А», «Ситуация Б» или «Ситуация В».

Если информации нет в руководствах вообще:
1) Прямо напиши в первой строке: «В предоставленных руководствах по эксплуатации такой информации нет.»
2) Добавь подзаголовок «Что найдено в документации по смежной теме:» и кратко перечисли близкие разделы из КОНТЕКСТА с указанием документа, раздела и страницы.
3) Предложи обратиться в сервисную службу завода и выведи контакты:
   - Изготовитель: ООО «ЧЕБОКСАРСКИЙ ЗАВОД СИЛОВЫХ АГРЕГАТОВ» (ООО «ЧЗСА»)
   - Телефон: +7 (835) 220-42-43
   - E-mail: mail@chzsa.ru
   - Адрес: 428028, Чувашская Республика, г. Чебоксары, пр-т Тракторостроителей, 109 (Источник: РЭ Автопогрузчиков, Раздел 18, стр. 58).

Если ответ в базе есть, но ТОЛЬКО для другой модели погрузчика (например, спрашивают про газовый редуктор или ЭМК газа на дизельном погрузчике ПД3,0):
1) Прямо укажи, что для запрошенной дизельной машины таких узлов и данных нет, и назови модели, для которых это описано в руководстве (газобензиновые погрузчики серии ПГ1,5–ПГ3,5, Раздел 11, стр. 40–44).
2) Добавь предупреждение: «ВНИМАНИЕ: переносить эти данные и регламенты на запрошенную модель нельзя!»
3) Укажи ссылку на раздел, страницу и выведи контакты сервисной службы ООО «ЧЗСА».

Если вопрос выходит за рамки документации погрузчика (например, разборка/ремонт самого двигателя Kubota или ММЗ Д-243, моменты затяжки болтов ГБЦ, регулировка клапанов ДВС, внутреннее устройство сторонней АКБ):
1) Прямо напиши в первой строке: «В предоставленных руководствах по эксплуатации погрузчиков такой информации нет.» и поясни, что внутренний ремонт и регулировка ДВС описаны в отдельном документе изготовителя агрегата — «Инструкции по эксплуатации на двигатель внутреннего сгорания» (согласно Введению РЭ Автопогрузчиков, стр. 2), которая в текущую базу не входит.
2) Добавь блок «Что найдено в документации по смежной теме:» (например, моменты затяжки колесных гаек в Таблице 2 Раздела 4 на стр. 10 и проверка тепловых зазоров в Разделе 7 на стр. 18).
3) Выведи контакты сервисной службы ООО «ЧЗСА».

5. ФОРМАТ ПРЕДМЕТНОГО ОТВЕТА:
Используй четкую структуру:
- **Модель техники:** точное название модели или семейства.
- **Точный ответ и регламент:** конкретные марки масел/жидкостей, точные заправочные объемы в литрах, интервалы в моточасах или полный пошаговый порядок работ.
- **Предупреждения и безопасность:** все запреты («ЗАПРЕЩАЕТСЯ», «ВНИМАНИЕ», «ОСТОРОЖНО», «ВАЖНО») по теме вопроса.
- **Где проверить в первоисточнике:** официальное название руководства, номер и название раздела/пункта, номер таблицы или рисунка и точный номер страницы (листа).
"""


class SilantRAG:
    """Ядро RAG-ассистента: гибридный поиск с учетом структуры таблиц и генерация ответа."""

    def __init__(self, top_k: int = 14):
        load_dotenv(override=True)
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        self.deepseek_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        self.gemini_key = os.getenv("GEMINI_API_KEY", "").strip()

        self.max_chunks = 6
        print(" -> Подключаем гибридный поиск (ChromaDB + BM25)...", flush=True)
        self.vectorstore = get_chroma_instance()
        self.all_chunks_raw = []
        self._reload_chunks()

        if self.openrouter_key:
            print(" -> Основной движок ИИ: OpenRouter API (работает без VPN)", flush=True)
        elif self.deepseek_key:
            print(" -> Основной движок ИИ: DeepSeek API", flush=True)
        elif self.gemini_key:
            print(" -> Основной движок ИИ: Google Gemini API", flush=True)

    def _reload_chunks(self):
        """Читает актуальную базу чанков из JSON, фильтруя только подключенные PDF."""
        p = Path("data/parsed_chunks.json")
        existing_pdfs = {f.name for f in Path("data/manuals").glob("*.pdf")}
        self.all_chunks_raw = []
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    items = json.load(f)
                for it in items:
                    src = it.get("metadata", {}).get("source_file", "")
                    if not existing_pdfs or src in existing_pdfs:
                        self.all_chunks_raw.append(it)
            except Exception:
                pass

    def _expand_short_query(self, q: str, active_code: str = "") -> str:
        """Расширяет запрос синонимами из документации для лексического и векторного поиска."""
        clean_q = re.sub(r"[?!.,;:]", "", q.strip().lower())
        if clean_q in SHORT_QUERY_MAP:
            return SHORT_QUERY_MAP[clean_q]

        expanded = q
        if re.search(r"\b(1\s*-?\s*то|то\s*-?\s*1|перв\w+\s+то)\b", clean_q):
            expanded += " (техническое обслуживание ТО-1 периодичность контрольный лист каждые 200 250 моточасов)"
        elif re.search(r"\b(2\s*-?\s*то|то\s*-?\s*2|втор\w+\s+то)\b", clean_q):
            expanded += " (техническое обслуживание ТО-2 периодичность контрольный лист каждые 400 500 1000 моточасов)"
        elif re.search(r"\b(3\s*-?\s*то|то\s*-?\s*3|трет\w+\s+то)\b", clean_q):
            expanded += " (техническое обслуживание ТО-3 периодичность контрольный лист каждые 1000 2000 моточасов)"
        elif re.search(r"\bто\b", clean_q) or "периодичност" in clean_q:
            expanded += " (техническое обслуживание техобслуживание регламент периодичность контрольный лист моточасов)"

        if "масл" in clean_q or "объем" in clean_q or "жидкост" in clean_q:
            expanded += " (РАЗДЕЛ 9 Расходные материалы масла смазки жидкости заправочные объемы рабочие жидкости)"
            if "gros" in active_code.lower() or "cpd" in active_code.lower():
                expanded += " (точки смазки рекомендованное масло проверка гидравлического масла масляный бак CPD 10B 15B 20B 25B 30B)"
        if "колес" in clean_q or "домкрат" in clean_q or "шин" in clean_q:
            expanded += " (Замена колес ведущего моста Таблица 2 момент затяжки гаек давление в шинах грузоподъемность домкрата 2/3 массы деревянные балки)"
        if "гбц" in clean_q or "головк" in clean_q or "клапан" in clean_q:
            expanded += " (Введение инструкция по эксплуатации на двигатель внутреннего сгорания Таблица 2 момент затяжки колесных гаек Раздел 18 ЧЗСА)"
        if re.search(r"\bакб\b", clean_q) or "батаре" in clean_q or "зарядк" in clean_q:
            expanded += " (Техника безопасности во время обслуживания аккумуляторной батареи приточно-вытяжная вентиляция электролит)"

        return expanded

    def _match_known_model_in_text(self, text: str) -> str:
        """Проверяет, упоминается ли в тексте вопроса конкретная модель техники."""
        m_silant = MODEL_SCAN_RE.search(text)
        if m_silant:
            return f"{m_silant.group(1).upper()}{m_silant.group(2).replace('.', ',')}"

        m_intl = INTL_FORKLIFT_RE.search(text)
        if m_intl:
            return f"{m_intl.group(1).upper()}{m_intl.group(2).upper()}"

        reg = load_models_registry()
        for code in reg.get("model_codes", []):
            if code.lower() in text.lower():
                return code
        return ""

    def _detect_active_model_code(self, question: str, selected_model: str, chat_history: list = None) -> tuple[str, str, bool]:
        """Определяет активную модель погрузчика из текста вопроса, ответа на уточнение или селектора UI."""
        explicit_in_q = self._match_known_model_in_text(question)
        if explicit_in_q:
            if chat_history and len(chat_history) >= 2 and len(question.strip()) <= 25:
                last_bot_msg = chat_history[-1].get("content", "").lower()
                prev_user_q = chat_history[-2].get("content", "")
                if "уточните" in last_bot_msg and "модель" in last_bot_msg:
                    return explicit_in_q, f"{prev_user_q} (для модели {explicit_in_q})", True
            return explicit_in_q, question, False

        if selected_model and selected_model != "Не выбрана":
            m_sel = MODEL_SCAN_RE.search(selected_model)
            if m_sel:
                return f"{m_sel.group(1).upper()}{m_sel.group(2).replace('.', ',')}", question, False
            return selected_model.strip(), question, False

        return "", question, False

    def _needs_model_clarification(self, question: str, active_code: str) -> bool:
        """Проверяет необходимость уточнения модели до выдачи ответа (согласно п. 6 ТЗ)."""
        if active_code:
            return False
        q_low = question.lower()
        if any(fam in q_low for fam in ("электропогрузчик", "газобензин", "дизельн")):
            return False
        if any(gen in q_low for gen in ("ламп", "буксиров", "хранен", "транспортиров", "обкатк")):
            return False

        model_dependent_keywords = (
            "то", "то-1", "то-2", "то-3", "то1", "то2", "то3", "моточас", "периодичност",
            "интервал", "масл", "объем", "литр", "заправ", "антифриз", "тосол",
            "жидкост", "давлен", "шин", "момент затяжк", "гайк", "колес"
        )
        return any(re.search(rf"\b{kw}", q_low) for kw in model_dependent_keywords)

    def _is_chunk_compatible_with_model(self, meta: dict, model_code: str) -> bool:
        """Фильтрует чанки документации по принадлежности к выбранной модели или семейству."""
        if not model_code:
            return True

        family = str(meta.get("family", "")).lower()
        models_meta = str(meta.get("models", ""))
        codes_str = str(meta.get("model_codes_str", ""))

        if model_code.startswith("ПЭ"):
            return "электропогрузчики силант" in family or ("электропогрузчик" in family and "дополнительная" not in family)

        if model_code.startswith(("ПД", "ПГ")):
            if "электропогрузчик" in family or "дополнительная техника" in family:
                return False
            if model_code != "ПД5,0" and "Только дизельный автопогрузчик ПД5,0" in models_meta:
                return False
            if model_code == "ПД5,0" and "интервалы ТО: 100/200/400" in models_meta:
                return False
            return True

        return (
            model_code.lower() in models_meta.lower()
            or model_code.lower() in codes_str.lower()
            or model_code.lower() in family
        )

    def _get_priority_pages_for_query(self, question: str, active_code: str) -> list[tuple[str, int]]:
        """Возвращает приоритетные страницы для процедур, расположенных на разворотах нескольких страниц."""
        q_low = question.lower()
        priority = []

        is_pe = active_code.startswith("ПЭ")
        is_pd_pg = active_code.startswith(("ПД", "ПГ")) or not active_code

        if is_pd_pg:
            auto_doc_prefix = "РЭ Автопогрузчики"
            if any(w in q_low for w in ("колес", "домкрат", "момент затяжк", "шин")):
                priority.extend([(auto_doc_prefix, 10), (auto_doc_prefix, 20), (auto_doc_prefix, 21)])
            if any(w in q_low for w in ("масл", "объем", "трансмисс", "охлажд", "антифриз", "тосол", "жидкост")):
                if active_code == "ПД5,0":
                    priority.extend([(auto_doc_prefix, 27), (auto_doc_prefix, 28)])
                elif active_code in ("ПД1,5", "ПГ1,5", "ПД1,8", "ПГ1,8"):
                    priority.extend([(auto_doc_prefix, 25), (auto_doc_prefix, 26)])
                elif active_code in ("ПД2,0", "ПГ2,0", "ПД2,5", "ПГ2,5"):
                    priority.extend([(auto_doc_prefix, 26), (auto_doc_prefix, 28)])
                else:
                    priority.extend([(auto_doc_prefix, 27), (auto_doc_prefix, 28)])
            if any(w in q_low for w in ("газ", "редуктор", "конденсат", "эмк", "баллон")) or active_code.startswith("ПГ"):
                priority.extend([(auto_doc_prefix, 41), (auto_doc_prefix, 42), (auto_doc_prefix, 43)])
            if any(w in q_low for w in ("то-1", "то-2", "то-3", "моточас", "периодичност")):
                if active_code == "ПД5,0":
                    priority.extend([(auto_doc_prefix, 18), (auto_doc_prefix, 34)])
                else:
                    priority.extend([(auto_doc_prefix, 18), (auto_doc_prefix, 29)])
            if any(w in q_low for w in ("гбц", "головк", "клапан", "поршн", "коленвал")):
                priority.extend([(auto_doc_prefix, 2), (auto_doc_prefix, 10), (auto_doc_prefix, 18), (auto_doc_prefix, 58)])

        if is_pe:
            pe_doc_prefix = "ПЭ15"
            if any(w in q_low for w in ("периодичност", "регламент", "технического обслуживания", "то-1", "то-2", "моточас")):
                priority.extend([(pe_doc_prefix, 55), (pe_doc_prefix, 56), (pe_doc_prefix, 35)])
            if any(w in q_low for w in ("акб", "батаре", "заряд", "электролит")):
                priority.extend([(pe_doc_prefix, 17), (pe_doc_prefix, 37)])
            if any(w in q_low for w in ("ошибк", "напряжен", "момент", "контроллер", "неисправн")):
                priority.extend([(pe_doc_prefix, 58), (pe_doc_prefix, 57)])

        return priority

    def _retrieve_best_chunks(self, question: str, search_query: str, active_code: str) -> list[dict]:
        """Собирает до 6 наиболее точных страниц с гарантированным захватом парных страниц процедур."""
        self._reload_chunks()
        if not self.all_chunks_raw:
            return []

        if active_code:
            compatible_raw = [
                it for it in self.all_chunks_raw
                if self._is_chunk_compatible_with_model(it.get("metadata", {}), active_code)
            ]
        else:
            silant_only = [
                it for it in self.all_chunks_raw
                if "дополнительная техника" not in str(it.get("metadata", {}).get("family", "")).lower()
            ]
            compatible_raw = silant_only if silant_only else self.all_chunks_raw

        pool_raw = compatible_raw if compatible_raw else self.all_chunks_raw

        by_file_page = {}
        for it in self.all_chunks_raw:
            m = it.get("metadata", {})
            by_file_page[(m.get("source_file", ""), int(m.get("page", 0)))] = it

        merged = []
        seen_keys = set()

        def add_chunk_item(raw_item: dict):
            if len(merged) >= self.max_chunks:
                return
            meta = sanitize_metadata(raw_item.get("metadata", {}))
            key = f"{meta.get('source_file')}_{meta.get('page')}"
            if key not in seen_keys:
                seen_keys.add(key)
                merged.append({
                    "doc_title": meta.get("doc_title", ""),
                    "section": meta.get("section", ""),
                    "models": meta.get("models", ""),
                    "page": meta.get("page", ""),
                    "text": clean_special_symbols(raw_item.get("text", "")),
                    "meta": meta
                })

        priority_targets = self._get_priority_pages_for_query(question, active_code)
        for file_sub, p_num in priority_targets:
            for (src_f, page_n), raw_it in by_file_page.items():
                if file_sub.lower() in src_f.lower() and page_n == p_num:
                    add_chunk_item(raw_it)

        pool_docs = [
            Document(page_content=clean_special_symbols(it["text"]), metadata=sanitize_metadata(it.get("metadata", {})))
            for it in pool_raw
        ]

        bm25 = BM25Retriever.from_documents(pool_docs)
        bm25.k = 6
        bm25_docs = bm25.invoke(search_query)

        vec_docs = []
        try:
            raw_vec = self.vectorstore.similarity_search(search_query, k=25)
            for d in raw_vec:
                if active_code:
                    if self._is_chunk_compatible_with_model(d.metadata, active_code):
                        vec_docs.append(d)
                else:
                    if "дополнительная техника" not in str(d.metadata.get("family", "")).lower():
                        vec_docs.append(d)
                if len(vec_docs) >= 6:
                    break
        except Exception:
            pass

        max_len = max(len(bm25_docs), len(vec_docs))
        for i in range(max_len):
            for candidate_list in (bm25_docs, vec_docs):
                if i < len(candidate_list):
                    doc = candidate_list[i]
                    meta = doc.metadata
                    raw_match = by_file_page.get((meta.get("source_file", ""), int(meta.get("page", 0))))
                    if raw_match:
                        add_chunk_item(raw_match)
                    if len(merged) >= self.max_chunks:
                        return merged

        return merged[:self.max_chunks]

    def _try_openrouter(self, api_key: str, user_prompt: str) -> tuple[bool, str]:
        """
        Отправляет запрос через OpenRouter без жесткого обрезания max_tokens и с минимальным
        режимом reasoning, чтобы модели не тратили лимит токенов на внутренние рассуждения.
        """
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "SILANT Service Assistant"
        }

        # Проверенные быстрые модели OpenRouter + автоматический роутер openrouter/free
        models_sequence = [
            "google/gemma-4-26b-a4b-it:free",
            "baidu/ernie-5.5-flash-preview:free",
            "nvidia/nemotron-3-ultra-550b-a55b:free",
            "cohere/north-mini-code:free",
            "openrouter/free",
            "openrouter/free",
            "openrouter/free"
        ]

        last_err = ""
        for idx, m_name in enumerate(models_sequence):
            payload = {
                "model": m_name,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.0,
                "reasoning": {"effort": "low"}
            }
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=(6, 30))
                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices", [])
                    if choices:
                        msg_obj = choices[0].get("message", {})
                        txt = msg_obj.get("content") or ""
                        if txt and txt.strip():
                            clean_txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL).strip()
                            used_model = data.get("model", m_name)
                            print(f" -> Ответ получен через OpenRouter ({used_model})", flush=True)
                            return True, clean_txt
                last_err = f"Код {resp.status_code} ({m_name})"
                time.sleep(1.5)
            except Exception as e:
                last_err = str(e)
                time.sleep(1.5)
                continue

        return False, last_err

    def _try_deepseek_direct(self, api_key: str, user_prompt: str) -> tuple[bool, str]:
        url = "https://api.deepseek.com/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.0,
            "stream": False
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=(5, 15))
            if resp.status_code == 200:
                choices = resp.json().get("choices", [])
                if choices:
                    return True, choices[0].get("message", {}).get("content", "").strip()
            return False, f"DeepSeek ({resp.status_code})"
        except Exception as e:
            return False, str(e)

    def _try_gemini_http(self, api_key: str, user_prompt: str) -> tuple[bool, str]:
        headers = {"Content-Type": "application/json"}
        payload = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": 0.0}
        }
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent?key={api_key}"
        proxy_url = os.getenv("GEMINI_PROXY", "").strip()
        proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None

        try:
            resp = requests.post(url, headers=headers, json=payload, proxies=proxies, timeout=(5, 14))
            if resp.status_code == 200:
                candidates = resp.json().get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    return True, "\n".join(p.get("text", "") for p in parts if not p.get("thought", False)).strip()
            return False, f"Gemini ({resp.status_code})"
        except Exception as e:
            return False, str(e)

    def _generate_answer(self, user_prompt: str) -> str:
        load_dotenv(override=True)
        or_key = os.getenv("OPENROUTER_API_KEY", "").strip() or self.openrouter_key
        ds_key = os.getenv("DEEPSEEK_API_KEY", "").strip() or self.deepseek_key
        gm_key = os.getenv("GEMINI_API_KEY", "").strip() or self.gemini_key

        errors = []
        if or_key:
            ok, res = self._try_openrouter(or_key, user_prompt)
            if ok:
                return res
            errors.append(f"OpenRouter: {res}")

        if ds_key:
            ok, res = self._try_deepseek_direct(ds_key, user_prompt)
            if ok:
                return res
            errors.append(res)

        if gm_key:
            ok, res = self._try_gemini_http(gm_key, user_prompt)
            if ok:
                return res
            errors.append(res)

        return f"⚠️ Сервер языковой модели временно занят. Пожалуйста, повторите вопрос через пару секунд. ({' | '.join(errors)})"

    def ask(self, question: str, selected_model: str = "Не выбрана", chat_history: list = None) -> dict:
        t0 = time.time()
        active_code, effective_question, need_history = self._detect_active_model_code(question, selected_model, chat_history)

        if self._needs_model_clarification(effective_question, active_code):
            clarification_text = (
                "Пожалуйста, **уточните модель погрузчика** (например: дизельный **ПД1,5–ПД3,5**, тяжелый **ПД5,0**, "
                "газобензиновый **ПГ1,5–ПГ3,5** или электропогрузчик **ПЭ1,5–ПЭ2,5**).\n\n"
                "Регламенты технического обслуживания, интервалы в моточасах, заправочные объемы ГСМ и моменты затяжки "
                "в заводских руководствах ООО «ЧЗСА» различаются в зависимости от модификации техники. "
                "Укажите модель в меню слева или напишите её в следующем сообщении, и я выдам точный регламент."
            )
            return {
                "answer": clarification_text,
                "sources": [],
                "elapsed_sec": round(time.time() - t0, 2)
            }

        expanded_q = self._expand_short_query(effective_question, active_code)
        search_query = f"{active_code}: {expanded_q}" if active_code else expanded_q

        final_candidates = self._retrieve_best_chunks(effective_question, search_query, active_code)

        context_parts = []
        sources_list = []
        for idx, item in enumerate(final_candidates, 1):
            context_parts.append(f"=== ФРАГМЕНТ ДОКУМЕНТАЦИИ #{idx} ===\n{item['text']}")
            sources_list.append({
                "doc_title": item["doc_title"],
                "section": item["section"],
                "models": item["models"],
                "page": item["page"],
                "text": item["text"]
            })

        context_str = "\n\n".join(context_parts)

        history_str = ""
        if chat_history and need_history:
            recent = chat_history[-2:]
            lines = []
            for msg in recent:
                role = "Пользователь" if msg["role"] == "user" else "Ассистент"
                lines.append(f"{role}: {msg['content']}")
            history_str = "КОНТЕКСТ ПРЕДЫДУЩЕГО СООБЩЕНИЯ:\n" + "\n".join(lines) + "\n\n"

        effective_model_label = active_code if active_code else "Не выбрана (вопрос по общему разделу)"

        user_prompt = (
            f"{history_str}"
            f"ВЫБРАННАЯ В ИНТЕРФЕЙСЕ МОДЕЛЬ ТЕХНИКИ: {effective_model_label}\n"
            f"ВОПРОС ПОЛЬЗОВАТЕЛЯ: {effective_question}\n\n"
            f"КОНТЕКСТ ИЗ РУКОВОДСТВ ПО ЭКСПЛУАТАЦИИ:\n{context_str}\n\n"
            f"Сформируй полный и точный ответ строго по системным правилам на основе предоставленных фрагментов."
        )

        answer_text = self._generate_answer(user_prompt)
        elapsed = round(time.time() - t0, 1)

        return {
            "answer": answer_text,
            "sources": sources_list,
            "elapsed_sec": elapsed
        }