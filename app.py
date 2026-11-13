import warnings
warnings.filterwarnings("ignore")

import base64
from pathlib import Path
import streamlit as st

from rag_engine import SilantRAG
from parser import load_models_registry, clean_special_symbols
from indexer import build_vector_db, remove_pdf_from_db, index_specific_pdfs

# Конфигурация страницы браузера
st.set_page_config(
    page_title="СИЛАНТ — Сервисный портал ЧЗСА",
    page_icon="🚜",
    layout="wide"
)

# Резервный SVG-логотип СИЛАНТ | ЧЗСА, если локальный файл photo.svg отсутствует
FALLBACK_LOGO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 340 72" height="46">
  <g fill="#163E6C">
    <path d="M12 22 h42 l-4 11 h-24 l-3 9 h24 l-4 11 h-37 l6 -18 l5 -3 z"/>
    <path d="M58 22 h14 l-6 17 l18 -17 h14 l-11 31 h-14 l6 -17 l-18 17 h-14 z"/>
    <path d="M108 22 h40 l-11 31 h-14 l7 -20 h-12 l-7 20 h-14 z"/>
    <path d="M154 22 h40 l-11 31 h-13 l2 -6 h-12 l-2 6 h-15 z M167 38 h10 l3 -8 h-10 z"/>
    <path d="M200 22 h14 l-4 10 h12 l4 -10 h14 l-11 31 h-14 l4 -11 h-12 l-4 11 h-14 z"/>
    <path d="M244 22 h42 l-4 11 h-14 l-7 20 h-14 l7 -20 h-14 z"/>
  </g>
  <line x1="296" y1="6" x2="296" y2="66" stroke="#8C9398" stroke-width="1.5"/>
  <g transform="translate(306, 4)">
    <path d="M2 6 Q16 0 30 6 L30 46 Q30 58 16 64 Q2 58 2 46 Z" fill="#1E242B" stroke="#A0A6AA" stroke-width="2.5"/>
    <path d="M4 8 Q16 3 28 8 L28 36 L4 36 Z" fill="#0B65B8"/>
    <path d="M28 12 L28 36 L4 36 Z" fill="#D20A11"/>
    <polygon points="24,11 11,24 16,24 8,34 22,22 17,22" fill="#FFFFFF"/>
    <text x="16" y="52" font-family="Arial, sans-serif" font-weight="bold" font-size="11" fill="#FFFFFF" text-anchor="middle">ЧЗСА</text>
  </g>
</svg>
"""

# Кастомные стили под корпоративную палитру silant.com (#EBE6D6, #163E6C, #D20A11)
CUSTOM_CSS = """
<style>
.stAppDeployButton, [data-testid="stAppDeployButton"], #MainMenu, [data-testid="stMainMenu"], footer {
    display: none !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
    height: 2.8rem !important;
    z-index: 999990 !important;
    pointer-events: none !important;
}

/* Оформление индикатора выполнения запроса в правом верхнем углу */
[data-testid="stStatusWidget"] {
    pointer-events: auto !important;
    display: flex !important;
    visibility: visible !important;
    background-color: #FFFFFF !important;
    border: 1px solid #163E6C !important;
    border-radius: 20px !important;
    padding: 3px 12px !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.12) !important;
}
[data-testid="stStatusWidget"] label {
    font-size: 0 !important;
}
[data-testid="stStatusWidget"] label::after {
    content: "Выполняется...";
    font-size: 0.82rem !important;
    color: #163E6C !important;
    font-weight: 600 !important;
    margin-left: 6px !important;
}
[data-testid="stStatusWidget"] button {
    font-size: 0 !important;
}
[data-testid="stStatusWidget"] button::after {
    content: "Стоп";
    font-size: 0.8rem !important;
    color: #D20A11 !important;
    font-weight: 600 !important;
}

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
}

[data-testid="stSidebar"] {
    background-color: #F4F1EA !important;
    border-right: 2px solid #E2DDD0 !important;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #163E6C !important;
}

.sidebar-subtitle {
    text-align: center;
    font-size: 0.82rem;
    color: #4A525D;
    line-height: 1.35;
    margin-top: 6px;
    margin-bottom: 4px;
    font-weight: 500;
}
.nowrap {
    white-space: nowrap;
}

[data-testid="stSidebar"] [data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 1px solid #163E6C !important;
    border-radius: 5px !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary p {
    font-weight: 700 !important;
    color: #163E6C !important;
    font-size: 0.92rem !important;
}

.pdf-card {
    background-color: #F9F8F5;
    border: 1px solid #DCD6C8;
    border-left: 3px solid #163E6C;
    border-radius: 4px;
    padding: 6px 8px;
    margin-bottom: 6px;
    font-size: 0.77rem;
    line-height: 1.28;
}
.pdf-card-title {
    font-weight: 700;
    color: #163E6C;
    word-break: break-word;
    display: block;
    margin-bottom: 2px;
}
.pdf-card-models {
    color: #5A626A;
    font-size: 0.72rem;
}

.stButton > button {
    border-radius: 4px !important;
    border: 1px solid #163E6C !important;
    color: #163E6C !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background-color: #D20A11 !important;
    border-color: #D20A11 !important;
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] .stButton > button {
    padding: 0 !important;
    width: 28px !important;
    min-width: 28px !important;
    max-width: 28px !important;
    height: 28px !important;
    min-height: 28px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: 0.85rem !important;
    line-height: 1 !important;
    margin: 4px auto 0 auto !important;
}

.silant-topbar {
    background-color: #EBE6D6;
    padding: 12px 24px;
    border-radius: 6px 6px 0 0;
    border-bottom: 3px solid #D20A11;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 14px;
}
.silant-topbar-contacts {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 18px;
    font-size: 0.92rem;
    color: #1E242B;
    font-weight: 600;
}
.silant-red-badge {
    background-color: #D20A11;
    color: #FFFFFF !important;
    padding: 6px 14px;
    border-radius: 3px;
    font-weight: 700;
    font-size: 0.82rem;
    letter-spacing: 0.4px;
    text-transform: uppercase;
    white-space: nowrap;
}

.silant-hero {
    background: linear-gradient(110deg, #163E6C 0%, #163E6C 68%, #1E4E87 68.1%, #123258 100%);
    color: #FFFFFF;
    padding: 20px 28px;
    border-radius: 0 0 6px 6px;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
    box-shadow: 0 4px 12px rgba(22, 62, 108, 0.15);
}
.silant-hero-title {
    font-size: 1.5rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin: 0 0 6px 0;
    color: #FFFFFF;
}
.silant-hero-accent {
    color: #FF3B42;
}
.silant-hero-sub {
    font-size: 0.94rem;
    color: #E0E8F0;
    margin: 0;
    line-height: 1.45;
}

.stMarkdown table {
    display: block;
    overflow-x: auto;
    white-space: nowrap;
}

@media (max-width: 768px) {
    .block-container {
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
        padding-top: 1rem !important;
    }
    .silant-topbar {
        flex-direction: column;
        align-items: flex-start;
        padding: 10px 14px;
        gap: 8px;
    }
    .silant-topbar-contacts {
        flex-direction: column;
        align-items: flex-start;
        gap: 6px;
        font-size: 0.85rem;
    }
    .silant-hero {
        flex-direction: column;
        text-align: left;
        padding: 16px;
        background: #163E6C;
    }
    .silant-hero-title {
        font-size: 1.2rem;
    }
    .silant-hero-sub {
        font-size: 0.85rem;
    }
}

/* Русификация стандартного компонента st.file_uploader */
[data-testid="stFileUploaderDropzoneInstructions"] > div > span {
    display: none !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] > div::before {
    content: "Перетащите PDF-руководство сюда";
    font-weight: 600;
    color: #163E6C;
    font-size: 0.88rem;
    display: block;
    margin-bottom: 2px;
}
[data-testid="stFileUploaderDropzoneInstructions"] > div > small {
    display: none !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] > div::after {
    content: "Формат PDF • до 200 МБ";
    font-size: 0.76rem;
    color: #5A626A;
    display: block;
}
[data-testid="stFileUploaderDropzone"] button {
    font-size: 0 !important;
}
[data-testid="stFileUploaderDropzone"] button * {
    display: none !important;
    font-size: 0 !important;
}
[data-testid="stFileUploaderDropzone"] button::after {
    content: "📁 Выбрать PDF-файл";
    font-size: 0.85rem !important;
    display: inline-block !important;
}
</style>
"""


def get_initial_greeting() -> list:
    """Формирует приветственное сообщение при старте сессии и после сброса диалога."""
    return [
        {
            "role": "assistant",
            "content": (
                "Здравствуйте! Я официальный сервисный ассистент по технике **СИЛАНТ** (ООО\u00a0«ЧЗСА»).\n\n"
                "Выберите модель вилочного погрузчика в панели слева (или укажите её прямо в вопросе) и напишите, "
                "что требуется найти: регламент ТО, заправочные объемы и марки ГСМ, коды ошибок контроллеров "
                "или безопасный порядок выполнения работ."
            ),
            "sources": [],
            "elapsed": None
        }
    ]


def get_logo_html(height: int = 44) -> str:
    """Загружает логотип из локального SVG-файла либо возвращает встроенный векторный шаблон."""
    candidates = [Path("photo.svg"), Path("data/photo.svg"), Path("assets/photo.svg")]
    for p in candidates:
        if p.exists():
            b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
            return f'<img src="data:image/svg+xml;base64,{b64}" height="{height}" style="max-width:100%; object-fit:contain;" alt="СИЛАНТ ЧЗСА">'
    return FALLBACK_LOGO_SVG


def get_forklift_img_html() -> str:
    """Подгружает изображение погрузчика для верхнего баннера."""
    candidates = [
        Path("forklift.png"),
        Path("изображение_5.png"),
        Path("data/forklift.png"),
        Path("data/изображение_5.png")
    ]
    for p in candidates:
        if p.exists():
            b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
            return f'<img src="data:image/png;base64,{b64}" height="105" style="object-fit:contain; filter: drop-shadow(0 4px 8px rgba(0,0,0,0.35));" alt="Погрузчик СИЛАНТ">'
    return ""


@st.cache_resource(show_spinner=False)
def load_rag_engine():
    """Кэширует экземпляр RAG-ядра в памяти процесса Streamlit."""
    return SilantRAG(top_k=14)


def format_chunk_preview(raw_chunk_text: str) -> str:
    """Отделяет служебный заголовок чанка от текста и сохраняет разметку Markdown-таблиц."""
    cleaned = clean_special_symbols(raw_chunk_text)
    sep = "----------------------------------------\n"
    if sep in cleaned:
        body = cleaned.split(sep, 1)[1]
    else:
        body = cleaned

    out_lines = []
    for line in body.strip().split("\n"):
        s = line.strip()
        if s.startswith("|") and s.endswith("|"):
            out_lines.append(s)
        elif s:
            out_lines.append(s + "  ")
        else:
            out_lines.append("")
    return "\n".join(out_lines)


def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0

    registry = load_models_registry()
    model_options = registry.get("ui_options", ["Не выбрана"])
    files_models_map = registry.get("files_models", {})

    top_logo_html = get_logo_html(height=42)
    sidebar_logo_html = get_logo_html(height=38)
    forklift_html = get_forklift_img_html()

    # Верхняя шапка и баннер в фирменном стиле silant.com
    st.markdown(
        f"""
        <div class="silant-topbar">
            <div>{top_logo_html}</div>
            <div class="silant-topbar-contacts">
                <span class="nowrap">📞 Горячая линия: 8 (800) 700-52-65</span>
                <span class="nowrap">🔧 Сервис ЧЗСА: +7 (835) 220-42-43</span>
                <span class="nowrap">✉️ mail@chzsa.ru</span>
                <span class="silant-red-badge">Сервисная документация</span>
            </div>
        </div>
        <div class="silant-hero">
            <div>
                <div class="silant-hero-title">
                    Интеллектуальный сервис <span class="silant-hero-accent">СИЛАНТ</span>
                </div>
                <p class="silant-hero-sub">
                    Инженерный RAG-ассистент по эксплуатации, техническому обслуживанию и диагностике вилочных погрузчиков <span class="nowrap">ПД / ПГ / ПЭ</span>.<br>
                    Все ответы формируются строго по заводским руководствам <span class="nowrap">ООО «ЧЗСА»</span> с привязкой к модели и проверкой первоисточника.
                </p>
            </div>
            <div>{forklift_html}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Боковая панель управления
    with st.sidebar:
        st.markdown(
            f"""
            <div style="text-align:center;">
                {sidebar_logo_html}
                <div class="sidebar-subtitle">
                    Чебоксарский завод силовых агрегатов<br>
                    <span class="nowrap">(ООО «ЧЗСА»)</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.divider()

        st.subheader("1. Выбор техники")
        selected_model = st.selectbox(
            "Укажите модель погрузчика:",
            options=model_options,
            index=0,
            help="Список моделей формируется автоматически из загруженных в базу PDF-руководств."
        )

        if st.button("🗑 Очистить историю диалога", use_container_width=True):
            st.session_state.messages = get_initial_greeting()
            st.rerun()

        st.divider()

        manuals_dir = Path("data/manuals")
        manuals_dir.mkdir(parents=True, exist_ok=True)
        current_pdfs = sorted(manuals_dir.glob("*.pdf"))

        if st.session_state.get("upload_error_msg"):
            st.warning(st.session_state.pop("upload_error_msg"))

        # Сворачиваемый блок управления PDF-документами
        with st.expander(f"📂 2. База PDF (подключено: {len(current_pdfs)})", expanded=False):
            st.caption("Список активных руководств и загрузка новых документов:")

            for pdf in current_pdfs:
                col_file, col_del = st.columns([3.4, 1])
                found_m = files_models_map.get(pdf.name, [])
                models_hint = f"Модели: {', '.join(found_m)}" if found_m else "В базе знаний"
                with col_file:
                    st.markdown(
                        f"""
                        <div class="pdf-card">
                            <span class="pdf-card-title">📄 {pdf.name}</span>
                            <span class="pdf-card-models">{models_hint}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with col_del:
                    if st.button("✕", key=f"del_{pdf.name}", help=f"Удалить {pdf.name} из базы"):
                        try:
                            removed = remove_pdf_from_db(pdf.name)
                            st.cache_resource.clear()
                            st.toast(f"Удален файл {pdf.name} ({removed} стр.)")
                        except Exception as e:
                            st.session_state["upload_error_msg"] = f"Не удалось удалить файл {pdf.name}: {e}"
                        st.rerun()

            uploaded_files = st.file_uploader(
                "Добавить новое PDF-руководство:",
                type=["pdf"],
                accept_multiple_files=True,
                key=f"pdf_uploader_{st.session_state['uploader_key']}"
            )

            if uploaded_files:
                saved_filenames = []
                saved_paths = []
                for uf in uploaded_files:
                    save_path = manuals_dir / uf.name
                    with open(save_path, "wb") as f:
                        f.write(uf.getbuffer())
                    saved_filenames.append(uf.name)
                    saved_paths.append(save_path)
                
                prog_bar = st.progress(0.0, text="Чтение документа и выявление моделей...")
                def update_inc_prog(frac, txt):
                    prog_bar.progress(min(max(float(frac), 0.0), 1.0), text=txt)

                try:
                    added_pages, found_models = index_specific_pdfs(saved_filenames, progress_callback=update_inc_prog)
                    prog_bar.empty()
                    st.cache_resource.clear()
                    models_msg = f" Выявлены модели: {', '.join(found_models)}" if found_models else ""
                    st.toast(f"Готово! Добавлено страниц: {added_pages}.{models_msg}")
                except Exception as e:
                    prog_bar.empty()
                    for bad_path in saved_paths:
                        bad_path.unlink(missing_ok=True)
                    st.session_state["upload_error_msg"] = (
                        f"⚠️ Загрузка документа отменена: не удалось обработать файл. "
                        f"Причина: {str(e)[:180]}"
                    )

                st.session_state["uploader_key"] += 1
                st.rerun()

            if st.button("🔄 Полная пересборка базы", use_container_width=True, help="Перечитать все PDF с нуля"):
                prog_bar = st.progress(0.0, text="Обновление базы знаний: 0%")
                def update_full_prog(frac, txt):
                    prog_bar.progress(min(max(float(frac), 0.0), 1.0), text=txt)

                try:
                    build_vector_db(rebuild_pdfs=True, progress_callback=update_full_prog)
                    prog_bar.empty()
                    st.cache_resource.clear()
                    st.toast("База знаний успешно обновлена на 100%!")
                except Exception as e:
                    prog_bar.empty()
                    st.session_state["upload_error_msg"] = f"⚠️ Не удалось пересобрать базу знаний. Причина: {str(e)[:180]}"
                st.rerun()

        st.divider()
        show_demo_btns = st.checkbox("Показать быстрые примеры запросов", value=False)

        st.divider()
        st.markdown(
            "**Сервисная служба ООО\u00a0«ЧЗСА»:**\n"
            "- 📞 `+7 (835) 220-42-43`\n"
            "- ☎️ `8 (800) 700-52-65`\n"
            "- ✉️ `mail@chzsa.ru`\n"
            "- 📍 г.\u00a0Чебоксары, пр-т\u00a0Тракторостроителей,\u00a0109"
        )

    # Быстрые демонстрационные вопросы (не зависят от случайно выбранной в сайдбаре сторонней модели)
    quick_q = None
    force_no_model = False
    if show_demo_btns:
        st.write("**Примеры типовых запросов для быстрой проверки:**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("🛢 Масло и ОЖ для ПД5,0", use_container_width=True):
                quick_q = "Какое моторное масло и в каком объеме заливать на ПД5,0, и какая охлаждающая жидкость допустима?"
                force_no_model = True
        with col2:
            if st.button("⏱ ТО-2 (без модели)", use_container_width=True):
                quick_q = "Через сколько моточасов проводится ТО-2?"
                force_no_model = True
        with col3:
            if st.button("⚠️ Лампа трансмиссии", use_container_width=True):
                quick_q = "Горит контрольная лампа температуры гидрожидкости трансмиссии на автопогрузчике. Каковы возможные причины?"
                force_no_model = True
        with col4:
            if st.button("🚫 Вопрос вне базы (ГБЦ)", use_container_width=True):
                quick_q = "Какой момент затяжки болтов головки блока цилиндров (ГБЦ) двигателя на ПД2,0?"
                force_no_model = True
        st.divider()

    if not st.session_state.get("messages"):
        st.session_state.messages = get_initial_greeting()

    chat_input_val = st.chat_input("Введите вопрос по эксплуатации, ТО или диагностике погрузчика СИЛАНТ...")
    user_query = quick_q or chat_input_val
    if user_query:
        st.session_state.messages.append({
            "role": "user",
            "content": user_query,
            "force_no_model": force_no_model
        })

    # Отрисовка истории диалога
    for msg in st.session_state.messages:
        avatar = "🚜" if msg["role"] == "assistant" else "👷‍♂️"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            
            if msg.get("elapsed"):
                st.caption(f"⏱ Время ответа: {msg['elapsed']} сек.")

            if msg.get("sources"):
                with st.expander(f"📖 Показать проверенные фрагменты документации ({len(msg['sources'])} стр.)"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(
                            f"**{i}. {src['doc_title']}**  \n"
                            f"📌 **Раздел:** {src['section']} | **Модели:** {src['models']} | **Стр.:** {src['page']}"
                        )
                        clean_preview = format_chunk_preview(src["text"])
                        st.markdown(clean_preview)
                        st.divider()

    # Обработка последнего сообщения пользователя (защищена от сброса при переключении чекбоксов)
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        pending_msg = st.session_state.messages[-1]
        pending_q = pending_msg["content"]
        is_forced_no_model = pending_msg.get("force_no_model", False)

        history_for_rag = [] if is_forced_no_model else [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]
        model_for_rag = "Не выбрана" if is_forced_no_model else selected_model

        with st.chat_message("assistant", avatar="🚜"):
            try:
                rag = load_rag_engine()
                with st.spinner("Поиск в заводских руководствах СИЛАНТ..."):
                    result = rag.ask(
                        question=pending_q,
                        selected_model=model_for_rag,
                        chat_history=history_for_rag
                    )
            except Exception as e:
                result = {
                    "answer": f"⚠️ Произошла техническая задержка при обработке запроса. Пожалуйста, повторите вопрос. (Детали: {str(e)[:150]})",
                    "sources": [],
                    "elapsed_sec": 0.1
                }

            st.markdown(result["answer"])
            st.caption(f"⏱ Время ответа: {result['elapsed_sec']} сек.")

            if result["sources"]:
                with st.expander(f"📖 Показать проверенные фрагменты документации ({len(result['sources'])} стр.)"):
                    for i, src in enumerate(result["sources"], 1):
                        st.markdown(
                            f"**{i}. {src['doc_title']}**  \n"
                            f"📌 **Раздел:** {src['section']} | **Модели:** {src['models']} | **Стр.:** {src['page']}"
                        )
                        clean_preview = format_chunk_preview(src["text"])
                        st.markdown(clean_preview)
                        st.divider()

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "elapsed": result["elapsed_sec"]
        })
        
        if quick_q:
            st.rerun()


if __name__ == "__main__":
    main()