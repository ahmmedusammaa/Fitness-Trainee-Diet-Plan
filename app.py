import os
import re
import json
import sys
import io
import html
from typing import Dict, Any, List

import streamlit as st

# ReportLab imports for professional PDF generation
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.pdfgen import canvas
except ImportError:
    pass

# Configure UTF-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ==============================================================================
# CONFIGURATION & CONSTANTS (Hidden from end users)
# ==============================================================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL_NAME = "openai/gpt-oss-120b"
PDF_PATH = r"E:\LLMs\gorilla-fitness-recipe-book.pdf"

# ==============================================================================
# DEFENSIVE IMPORTS
# ==============================================================================
try:
    from langchain_community.document_loaders import PyPDFLoader
except ImportError:
    PyPDFLoader = None

try:
    import pypdf
except ImportError:
    pypdf = None

try:
    from langchain_core.documents import Document
except ImportError:
    class Document:
        def __init__(self, page_content: str, metadata: dict = None):
            self.page_content = page_content
            self.metadata = metadata or {}

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        class RecursiveCharacterTextSplitter:
            def __init__(self, chunk_size=650, chunk_overlap=150, separators=None):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap
            def split_documents(self, documents):
                chunks = []
                for doc in documents:
                    text = doc.page_content
                    step = max(100, self.chunk_size - self.chunk_overlap)
                    for i in range(0, len(text), step):
                        chunk = text[i:i + self.chunk_size]
                        if chunk.strip():
                            chunks.append(Document(page_content=chunk, metadata=doc.metadata))
                return chunks

try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

try:
    from groq import Groq
except ImportError:
    Groq = None


# ==============================================================================
# STREAMLIT PAGE CONFIGURATION & NO-SIDEBAR STYLING
# ==============================================================================
st.set_page_config(
    page_title="Gorilla Fitness Trainee Diet Planner",
    page_icon="🦍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Completely hide sidebar and collapse toggle
st.markdown("""
<style>
    /* Completely hide sidebar and controls */
    [data-testid="stSidebar"], [data-testid="collapsedControl"] {
        display: none !important;
    }
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    button[kind="header"] {
        display: none !important;
    }

    /* Modern Aesthetics */
    .hero-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #10B981, #06B6D4, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.15rem;
        margin-bottom: 1.5rem;
    }
    .form-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }
    .meal-box {
        background: #0F172A;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .badge-cals { background-color: #1E3A8A; color: #93C5FD; }
    .badge-protein { background-color: #064E3B; color: #6EE7B7; }
    .badge-carbs { background-color: #78350F; color: #FCD34D; }
    .badge-fats { background-color: #831843; color: #F9A8D4; }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# RECIPE BOOK RETRIEVER (CACHED)
# ==============================================================================
@st.cache_resource(show_spinner="Preparing Gorilla Fitness Recipe Book...")
def get_recipe_retriever(pdf_path: str):
    """
    Loads recipes from PDF and builds an efficient semantic/keyword retriever.
    """
    if not os.path.exists(pdf_path):
        return None, 0, f"Recipe book not found at: {pdf_path}"

    docs = []
    # 1. PyPDFLoader
    if PyPDFLoader is not None:
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
        except Exception:
            docs = []

    # 2. pypdf fallback
    if not docs and pypdf is not None:
        try:
            reader = pypdf.PdfReader(pdf_path)
            for i, page in enumerate(reader.pages):
                txt = page.extract_text() or ""
                if txt.strip():
                    docs.append(Document(page_content=txt, metadata={"page": i + 1}))
        except Exception as e:
            return None, 0, f"Error reading PDF: {e}"

    if not docs:
        return None, 0, "No content extracted from recipe book."

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=650,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = splitter.split_documents(docs)

    # 3. Attempt FAISS + Embeddings
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(splits, embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 8}), len(docs), None
    except Exception:
        # High speed keyword retriever fallback
        class KeywordRetriever:
            def __init__(self, documents):
                self.documents = documents
            def invoke(self, query: str):
                keywords = [w.lower() for w in query.split() if len(w) > 2]
                scored = []
                for d in self.documents:
                    content = d.page_content.lower()
                    score = sum(1 for kw in keywords if kw in content)
                    scored.append((score, d))
                scored.sort(key=lambda x: x[0], reverse=True)
                return [d for s, d in scored[:8]]

        return KeywordRetriever(splits), len(docs), None


# ==============================================================================
# NUTRITIONAL TARGET CALCULATOR (Mifflin-St Jeor)
# ==============================================================================
def calculate_trainee_targets(weight_kg, height_cm, age, gender, activity_level, goal):
    if gender == "Male":
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161

    multipliers = {
        "Sedentary (desk job, no workout)": 1.2,
        "Light (exercise 1-3 days/week)": 1.375,
        "Moderate (exercise 3-5 days/week)": 1.55,
        "Very Active (hard exercise 6-7 days/week)": 1.725,
        "Extra Active (intense job & daily training)": 1.9
    }
    mult = multipliers.get(activity_level, 1.55)
    tdee = bmr * mult

    if "Fat Loss" in goal:
        target_calories = round(tdee - 500)
        protein_g = round(2.2 * weight_kg)
        fats_g = round((target_calories * 0.25) / 9)
        carbs_g = round((target_calories - (protein_g * 4 + fats_g * 9)) / 4)
    elif "Muscle Gain" in goal:
        target_calories = round(tdee + 350)
        protein_g = round(2.0 * weight_kg)
        fats_g = round((target_calories * 0.25) / 9)
        carbs_g = round((target_calories - (protein_g * 4 + fats_g * 9)) / 4)
    else:  # Maintenance
        target_calories = round(tdee)
        protein_g = round(1.8 * weight_kg)
        fats_g = round((target_calories * 0.28) / 9)
        carbs_g = round((target_calories - (protein_g * 4 + fats_g * 9)) / 4)

    return {
        "bmr": round(bmr),
        "tdee": round(tdee),
        "target_calories": target_calories,
        "protein_g": protein_g,
        "carbs_g": max(35, carbs_g),
        "fats_g": fats_g
    }


# ==============================================================================
# JSON EXTRACTION HELPER
# ==============================================================================
def parse_diet_plan_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Model returned an empty response. Please try again.")

    # 1. Direct JSON parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2. Try markdown json block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except Exception:
            pass

    # 3. Try from first { to last }
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end+1].strip())
        except Exception:
            pass

    raise ValueError(f"LLM response did not contain a valid JSON object. (Received {len(text)} characters)")


# ==============================================================================
# PDF GENERATION ENGINE
# ==============================================================================
class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print total page count."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont('Helvetica', 8)
        self.setFillColor(colors.HexColor('#64748B'))
        footer_text = f"Gorilla Fitness Diet Planner  •  Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 36, 20, footer_text)
        self.drawString(36, 20, "Confidential & Personalized Trainee Meal Plan")
        self.setStrokeColor(colors.HexColor('#CBD5E1'))
        self.setLineWidth(0.5)
        self.line(36, 30, 612 - 36, 30)
        self.restoreState()


def safe_pdf_str(val) -> str:
    if val is None:
        return ""
    return html.escape(str(val))


def generate_diet_plan_pdf(plan: Dict[str, Any], trainee_meta: Dict[str, Any] = None) -> bytes:
    """Renders a beautifully styled, branded Gorilla Fitness PDF document."""
    trainee_meta = trainee_meta or {}
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=42
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=2
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#0284C7'),
        spaceAfter=8
    )
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=8,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#1E293B')
    )
    body_bold = ParagraphStyle(
        'BodyDarkBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#0F172A')
    )
    meal_title_style = ParagraphStyle(
        'MealTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor('#0369A1')
    )
    meta_style = ParagraphStyle(
        'MetaStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#64748B'),
        alignment=2  # Right-aligned
    )

    story = []

    # 1. Header Banner
    story.append(Paragraph('GORILLA FITNESS', title_style))
    story.append(Paragraph('PERSONALIZED NUTRITION & RECIPE SCHEDULE', subtitle_style))
    story.append(HRFlowable(width='100%', thickness=1.5, color=colors.HexColor('#0284C7'), spaceBefore=0, spaceAfter=8))

    # 2. Trainee Info Table
    trainee_summary = plan.get('trainee_summary', {})
    name = trainee_meta.get('name') or trainee_summary.get('name', 'Trainee')
    goal = trainee_meta.get('goal') or trainee_summary.get('goal', 'Fitness Goal')
    age = trainee_meta.get('age', 'N/A')
    gender = trainee_meta.get('gender', 'N/A')
    weight = trainee_meta.get('weight', 'N/A')
    height = trainee_meta.get('height', 'N/A')
    activity = trainee_meta.get('activity', 'N/A')
    restrictions = trainee_meta.get('restrictions') or 'None'

    info_data = [
        [
            Paragraph(f"<b>Trainee:</b> {safe_pdf_str(name)}", body_style),
            Paragraph(f"<b>Goal:</b> {safe_pdf_str(goal)}", body_style),
            Paragraph(f"<b>Gender:</b> {safe_pdf_str(gender)}", body_style)
        ],
        [
            Paragraph(f"<b>Age:</b> {safe_pdf_str(age)} yrs", body_style),
            Paragraph(f"<b>Weight:</b> {safe_pdf_str(weight)} kg", body_style),
            Paragraph(f"<b>Height:</b> {safe_pdf_str(height)} cm", body_style)
        ],
        [
            Paragraph(f"<b>Activity:</b> {safe_pdf_str(activity)}", body_style),
            Paragraph(f"<b>Dietary Notes:</b> {safe_pdf_str(restrictions)}", body_style),
            Paragraph("<b>Plan Source:</b> Recipe Book Curated", body_style)
        ]
    ]
    info_table = Table(info_data, colWidths=[180, 180, 180])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#E2E8F0')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#F1F5F9')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 8))

    # 3. Targets vs Totals Table
    story.append(Paragraph('Daily Nutritional Breakdown (Target vs Plan)', section_heading))
    daily_totals = plan.get('daily_totals', {})

    nutri_data = [
        [
            Paragraph('<b>Nutritional Metric</b>', body_bold),
            Paragraph('<b>Daily Target</b>', body_bold),
            Paragraph('<b>Plan Total</b>', body_bold),
            Paragraph('<b>Adherence Status</b>', body_bold)
        ],
        [
            Paragraph('🔥 Total Calories', body_style),
            Paragraph(f"{safe_pdf_str(trainee_summary.get('target_calories', 'N/A'))} kcal", body_style),
            Paragraph(f"{safe_pdf_str(daily_totals.get('total_calories', 'N/A'))} kcal", body_style),
            Paragraph('Target Aligned', body_style)
        ],
        [
            Paragraph('🥩 Protein', body_style),
            Paragraph(f"{safe_pdf_str(trainee_summary.get('protein_g', 'N/A'))} g", body_style),
            Paragraph(f"{safe_pdf_str(daily_totals.get('total_protein_g', daily_totals.get('protein', 'N/A')))} g", body_style),
            Paragraph('Target Aligned', body_style)
        ],
        [
            Paragraph('🍚 Carbohydrates', body_style),
            Paragraph(f"{safe_pdf_str(trainee_summary.get('carbs_g', 'N/A'))} g", body_style),
            Paragraph(f"{safe_pdf_str(daily_totals.get('total_carbs_g', daily_totals.get('carbs', 'N/A')))} g", body_style),
            Paragraph('Target Aligned', body_style)
        ],
        [
            Paragraph('🥑 Healthy Fats', body_style),
            Paragraph(f"{safe_pdf_str(trainee_summary.get('fats_g', 'N/A'))} g", body_style),
            Paragraph(f"{safe_pdf_str(daily_totals.get('total_fats_g', daily_totals.get('fats', 'N/A')))} g", body_style),
            Paragraph('Target Aligned', body_style)
        ],
    ]
    nutri_table = Table(nutri_data, colWidths=[135, 135, 135, 135])
    nutri_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0F172A')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(nutri_table)

    adherence = daily_totals.get('adherence_comparison')
    if adherence:
        story.append(Spacer(1, 4))
        story.append(Paragraph(f"<b>Target Evaluation:</b> <i>{safe_pdf_str(adherence)}</i>", body_style))

    story.append(Spacer(1, 8))

    # 4. Meals & Recipes
    story.append(Paragraph('Curated Daily Meal Schedule & Recipes', section_heading))
    meals = plan.get('meals', [])
    if isinstance(meals, list):
        for idx, m in enumerate(meals, 1):
            m_type = m.get('meal_type', f'Meal {idx}')
            r_name = m.get('recipe_name', 'Fitness Recipe')
            page = m.get('page_reference')
            cals = m.get('calories', 'N/A')
            p = m.get('protein_g', 'N/A')
            c = m.get('carbs_g', 'N/A')
            f = m.get('fats_g', 'N/A')
            prep = m.get('preparation_notes', '')
            ingredients = m.get('key_ingredients', [])

            page_str = f"Book Reference: {safe_pdf_str(page)}" if page else "Gorilla Recipe Book"

            meal_header_data = [
                [
                    Paragraph(f"<b>[{safe_pdf_str(m_type).upper()}]</b> {safe_pdf_str(r_name)}", meal_title_style),
                    Paragraph(page_str, meta_style)
                ]
            ]
            meal_header_table = Table(meal_header_data, colWidths=[370, 170])
            meal_header_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#E0F2FE')),
                ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ]))

            macros_line = f"<b>Energy:</b> {cals} kcal &nbsp;&nbsp;|&nbsp;&nbsp; <b>Protein:</b> {p}g &nbsp;&nbsp;|&nbsp;&nbsp; <b>Carbs:</b> {c}g &nbsp;&nbsp;|&nbsp;&nbsp; <b>Fats:</b> {f}g"
            macro_table = Table([[Paragraph(macros_line, body_style)]], colWidths=[540])
            macro_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
                ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ]))

            if isinstance(ingredients, list):
                ing_text = "<br/>".join([f"• {safe_pdf_str(item)}" for item in ingredients])
            else:
                ing_text = safe_pdf_str(ingredients)

            details_data = [
                [
                    Paragraph("<b>Ingredients & Portions:</b><br/>" + ing_text, body_style),
                    Paragraph("<b>Preparation Instructions:</b><br/>" + (safe_pdf_str(prep) or "Standard recipe book prep."), body_style)
                ]
            ]
            details_table = Table(details_data, colWidths=[240, 300])
            details_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#FFFFFF')),
                ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ]))

            story.append(KeepTogether([
                meal_header_table,
                macro_table,
                details_table,
                Spacer(1, 7)
            ]))

    # 5. Grocery List
    grocery = plan.get('grocery_list', {})
    if grocery:
        story.append(Spacer(1, 4))
        story.append(Paragraph('🛒 Grocery Shopping Checklist', section_heading))
        g_rows = []
        if isinstance(grocery, dict):
            for cat, items in grocery.items():
                if isinstance(items, list):
                    items_str = "<br/>".join([f"[ ] {safe_pdf_str(it)}" for it in items])
                else:
                    items_str = safe_pdf_str(items)
                g_rows.append([
                    Paragraph(f"<b>{safe_pdf_str(cat)}</b>", body_bold),
                    Paragraph(items_str, body_style)
                ])
        elif isinstance(grocery, list):
            items_str = "<br/>".join([f"[ ] {safe_pdf_str(it)}" for it in grocery])
            g_rows.append([Paragraph("Items", body_bold), Paragraph(items_str, body_style)])

        if g_rows:
            g_table = Table(g_rows, colWidths=[140, 400])
            g_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
                ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ]))
            story.append(KeepTogether([g_table, Spacer(1, 6)]))

    # 6. Coach Advice
    advice = plan.get('coaching_advice', [])
    if advice:
        story.append(Paragraph('💡 Coach Advice & Performance Tips', section_heading))
        if isinstance(advice, list):
            advice_str = "<br/>".join([f"✓ {safe_pdf_str(tip)}" for tip in advice])
        else:
            advice_str = safe_pdf_str(advice)
        advice_table = Table([[Paragraph(advice_str, body_style)]], colWidths=[540])
        advice_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F0FDF4')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#86EFAC')),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ]))
        story.append(KeepTogether([advice_table, Spacer(1, 6)]))

    doc.build(story, canvasmaker=NumberedCanvas)
    return buf.getvalue()


# ==============================================================================
# HEADER
# ==============================================================================
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🦍 Gorilla Fitness Diet Planner</div>
    <div class="hero-subtitle">Fill in your profile below to get your personalized, chef-curated fitness diet plan</div>
</div>
""", unsafe_allow_html=True)

# Initialize Retriever silently
retriever, total_pages, ret_err = get_recipe_retriever(PDF_PATH)
if ret_err:
    st.error(f"⚠️ {ret_err}")

# ==============================================================================
# TRAINEE REQUIREMENT FORM
# ==============================================================================
with st.container():
    st.markdown("### 📋 Your Profile & Requirements")

    c1, c2, c3 = st.columns(3)
    with c1:
        name = st.text_input("Full Name", value="", placeholder="Enter your full name...")
        gender = st.selectbox("Gender", options=["Male", "Female"], index=0)
        age = st.number_input("Age (years)", min_value=14, max_value=85, value=None, placeholder="Enter your age (e.g. 25)")

    with c2:
        weight = st.number_input("Weight (kg)", min_value=35.0, max_value=200.0, value=None, placeholder="Enter weight in kg (e.g. 80.0)", step=0.5)
        height = st.number_input("Height (cm)", min_value=120.0, max_value=230.0, value=None, placeholder="Enter height in cm (e.g. 180.0)", step=1.0)
        meals_count = st.selectbox("Preferred Meals / Day", options=[3, 4, 5], index=1)

    with c3:
        activity = st.selectbox(
            "Daily Activity Level",
            options=[
                "Sedentary (desk job, no workout)",
                "Light (exercise 1-3 days/week)",
                "Moderate (exercise 3-5 days/week)",
                "Very Active (hard exercise 6-7 days/week)",
                "Extra Active (intense job & daily training)"
            ],
            index=2
        )
        goal = st.selectbox(
            "Primary Fitness Goal",
            options=[
                "Fat Loss / Cutting (-500 kcal)",
                "Muscle Gain / Bulking (+350 kcal)",
                "Maintenance (Maintain Weight)"
            ],
            index=0
        )
        restrictions = st.text_input(
            "Dietary Preferences / Restrictions",
            value="",
            placeholder="e.g. High protein, no seafood (leave blank if none)"
        )

# Real-time Calorie & Macro Target Card
if weight is not None and height is not None and age is not None:
    targets = calculate_trainee_targets(weight, height, age, gender, activity, goal)
    st.markdown("#### 🎯 Your Calibrated Daily Targets")
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.metric("🔥 Daily Calories", f"{targets['target_calories']} kcal", help=f"BMR: {targets['bmr']} | TDEE: {targets['tdee']}")
    with mc2:
        st.metric("🥩 Protein Target", f"{targets['protein_g']} g", delta=f"{round(targets['protein_g'] * 4)} kcal")
    with mc3:
        st.metric("🍚 Carbs Target", f"{targets['carbs_g']} g", delta=f"{round(targets['carbs_g'] * 4)} kcal")
    with mc4:
        st.metric("🥑 Fats Target", f"{targets['fats_g']} g", delta=f"{round(targets['fats_g'] * 9)} kcal")
else:
    targets = None
    st.info("💡 Fill in your profile above to automatically calculate your daily calorie and macronutrient targets.")

st.markdown("<br>", unsafe_allow_html=True)


# ==============================================================================
# GENERATION ACTION BUTTON
# ==============================================================================
generate_clicked = st.button("🚀 Create My Diet Plan", type="primary", use_container_width=True)

if generate_clicked:
    if not name or not name.strip():
        st.warning("⚠️ Please enter your Full Name before generating the diet plan.")
        st.stop()
    if age is None:
        st.warning("⚠️ Please enter your Age before generating the diet plan.")
        st.stop()
    if weight is None:
        st.warning("⚠️ Please enter your Weight before generating the diet plan.")
        st.stop()
    if height is None:
        st.warning("⚠️ Please enter your Height before generating the diet plan.")
        st.stop()

    if targets is None:
        targets = calculate_trainee_targets(weight, height, age, gender, activity, goal)
    st.session_state["active_targets"] = targets

    if not retriever:
        st.error("Recipe book could not be loaded. Please ensure the recipe book PDF is in the directory.")
        st.stop()

    with st.spinner("🍳 Reading Gorilla Fitness Recipe Book & creating your custom diet plan..."):
        try:
            # 1. Retrieve relevant recipe chunks
            query = f"healthy {goal} {restrictions} breakfast lunch dinner snack"
            recipe_docs = retriever.invoke(query)
            context = "\n\n---\n\n".join([
                f"[Recipe Book Page {d.metadata.get('page', 'N/A')}]:\n{d.page_content}"
                for d in recipe_docs[:6]
            ])

            # 2. Build Prompt
            prompt = f"""
You are an expert sports nutritionist and personal trainer.
Create a customized 1-day meal plan for the trainee below strictly using recipes from the Gorilla Fitness Recipe Book context.

TRAINEE INFO:
- Name: {name}
- Age: {age} | Gender: {gender}
- Weight: {weight} kg | Height: {height} cm
- Goal: {goal}
- Target Calories: {targets['target_calories']} kcal
- Target Protein: {targets['protein_g']} g
- Target Carbs: {targets['carbs_g']} g
- Target Fats: {targets['fats_g']} g
- Dietary Restrictions: {restrictions}
- Meals to Plan: {meals_count} meals

AVAILABLE RECIPES FROM BOOK:
{context}

CRITICAL INSTRUCTIONS:
Select recipes from the context and adjust portion sizes to hit the trainee's targets as closely as possible.
Return STRICTLY a JSON object formatted as:
```json
{{
  "trainee_summary": {{
    "name": "{name}",
    "goal": "{goal}",
    "target_calories": {targets['target_calories']},
    "protein_g": {targets['protein_g']},
    "carbs_g": {targets['carbs_g']},
    "fats_g": {targets['fats_g']}
  }},
  "meals": [
    {{
      "meal_type": "Breakfast",
      "recipe_name": "Recipe Name From Book",
      "page_reference": "Page number if found, else null",
      "key_ingredients": ["ingredient 1 with portion", "ingredient 2 with portion"],
      "calories": 450,
      "protein_g": 35,
      "carbs_g": 30,
      "fats_g": 15,
      "preparation_notes": "Preparation instructions from the recipe book."
    }}
  ],
  "daily_totals": {{
    "total_calories": {targets['target_calories']},
    "total_protein_g": {targets['protein_g']},
    "total_carbs_g": {targets['carbs_g']},
    "total_fats_g": {targets['fats_g']},
    "adherence_comparison": "Brief comment on how close the plan matches targets"
  }},
  "grocery_list": {{
    "Proteins": ["item 1", "item 2"],
    "Produce": ["item 1", "item 2"],
    "Pantry & Dairy": ["item 1", "item 2"]
  }},
  "coaching_advice": [
    "Practical coach tip 1 regarding water or timing",
    "Practical coach tip 2 regarding pre/post workout"
  ]
}}
```
Do not include any conversational preamble or text outside the markdown code block.
"""

            # 3. Call Groq with openai/gpt-oss-120b in streaming mode
            from groq import Groq
            client = Groq(api_key=GROQ_API_KEY)

            # Using max_completion_tokens=4096 and reasoning_effort="low" so reasoning does not exhaust the token budget
            completion = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"},
                temperature=1,
                max_completion_tokens=4096,
                top_p=1,
                reasoning_effort="low",
                stream=True,
                stop=None
            )

            full_content = ""
            for chunk in completion:
                delta = chunk.choices[0].delta.content or ""
                full_content += delta

            # Safety fallback if empty
            if not full_content.strip():
                fallback_comp = client.chat.completions.create(
                    model="qwen/qwen3.8-27b",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=3000
                )
                full_content = fallback_comp.choices[0].message.content

            # 4. Parse JSON
            parsed_plan = parse_diet_plan_json(full_content)
            st.session_state["active_diet_plan"] = parsed_plan
            st.session_state["trainee_meta"] = {
                "name": name,
                "gender": gender,
                "age": age,
                "weight": weight,
                "height": height,
                "meals_count": meals_count,
                "activity": activity,
                "goal": goal,
                "restrictions": restrictions
            }
            st.success("✅ Your Personalized Diet Plan is ready!")

        except Exception as e:
            st.error(f"Failed to generate diet plan: {e}")


# ==============================================================================
# DISPLAY GENERATED PLAN
# ==============================================================================
if "active_diet_plan" in st.session_state and st.session_state["active_diet_plan"]:
    plan = st.session_state["active_diet_plan"]
    saved_meta = st.session_state.get("trainee_meta", {})
    trainee_name = saved_meta.get("name") or name or "trainee"
    clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', str(trainee_name).strip()) or "trainee"

    # Pre-generate PDF document for responsive export
    try:
        pdf_bytes = generate_diet_plan_pdf(plan, saved_meta)
    except Exception as pdf_err:
        pdf_bytes = None
        pdf_err_msg = str(pdf_err)
    else:
        pdf_err_msg = None

    st.markdown("---")
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.markdown("### 🥗 Your Personalized Diet Plan & Recipe Schedule")
    with header_col2:
        if pdf_bytes:
            st.download_button(
                label="📄 Download Plan as PDF",
                data=pdf_bytes,
                file_name=f"{clean_name}_Gorilla_Diet_Plan.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
                key="top_pdf_download"
            )

    tab_meals, tab_totals, tab_grocery, tab_tips, tab_download = st.tabs([
        "🍽️ Meals & Recipes",
        "📊 Nutrition Totals",
        "🛒 Grocery List",
        "💡 Coach's Tips",
        "📥 Download Plan"
    ])

    # 1. Meals Tab
    with tab_meals:
        meals = plan.get("meals", [])
        for i, m in enumerate(meals, 1):
            m_type = m.get("meal_type", f"Meal {i}")
            r_name = m.get("recipe_name", "Fitness Recipe")
            page = m.get("page_reference")
            cals = m.get("calories", "N/A")
            p = m.get("protein_g", "N/A")
            c = m.get("carbs_g", "N/A")
            f = m.get("fats_g", "N/A")
            prep = m.get("preparation_notes", "")
            ingredients = m.get("key_ingredients", [])

            st.markdown(f"""
            <div class="meal-box">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h3 style="margin:0; color:#38BDF8;">[{m_type.upper()}] {r_name}</h3>
                    <span style="color:#94A3B8; font-size:0.9rem;">Book Page: {page if page else 'Recipe Book'}</span>
                </div>
                <div style="margin-top:10px;">
                    <span class="badge badge-cals">⚡ {cals} kcal</span>
                    <span class="badge badge-protein">🥩 {p}g Protein</span>
                    <span class="badge badge-carbs">🍚 {c}g Carbs</span>
                    <span class="badge badge-fats">🥑 {f}g Fats</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            col_ing, col_prep = st.columns([1, 1])
            with col_ing:
                st.markdown("**Ingredients & Portions:**")
                if isinstance(ingredients, list):
                    for ing in ingredients:
                        st.markdown(f"- {ing}")
                else:
                    st.write(ingredients)
            with col_prep:
                st.markdown("**Preparation & Cooking Instructions:**")
                st.info(prep if prep else "Follow standard instructions from Gorilla Fitness recipe book.")

            st.markdown("<hr style='border:0.5px solid #334155; margin:15px 0;'>", unsafe_allow_html=True)

    # 2. Nutrition Totals Tab
    with tab_totals:
        totals = plan.get("daily_totals", {})
        st.subheader("Daily Total Consumption vs Target")

        active_t = st.session_state.get("active_targets") or targets or {}
        tc1, tc2, tc3, tc4 = st.columns(4)
        with tc1:
            st.metric("Total Calories", f"{totals.get('total_calories', 'N/A')} kcal", delta=f"Target: {active_t.get('target_calories', 'N/A')} kcal")
        with tc2:
            st.metric("Total Protein", f"{totals.get('total_protein_g', totals.get('protein', 'N/A'))} g", delta=f"Target: {active_t.get('protein_g', 'N/A')} g")
        with tc3:
            st.metric("Total Carbs", f"{totals.get('total_carbs_g', totals.get('carbs', 'N/A'))} g", delta=f"Target: {active_t.get('carbs_g', 'N/A')} g")
        with tc4:
            st.metric("Total Fats", f"{totals.get('total_fats_g', totals.get('fats', 'N/A'))} g", delta=f"Target: {active_t.get('fats_g', 'N/A')} g")

        if "adherence_comparison" in totals:
            st.success(f"**Target Evaluation:** {totals['adherence_comparison']}")

    # 3. Grocery List Tab
    with tab_grocery:
        st.subheader("🛒 Interactive Shopping Checklist")
        grocery = plan.get("grocery_list", {})
        if isinstance(grocery, dict):
            for cat, items in grocery.items():
                st.markdown(f"**{cat.upper()}**")
                if isinstance(items, list):
                    for item in items:
                        st.checkbox(str(item), key=f"item_{cat}_{item}")
                else:
                    st.write(items)
        elif isinstance(grocery, list):
            for it in grocery:
                st.checkbox(str(it), key=f"item_{it}")
        else:
            st.write(grocery)

    # 4. Coach's Tips Tab
    with tab_tips:
        st.subheader("💡 Coach's Performance & Timing Recommendations")
        advice = plan.get("coaching_advice", [])
        if isinstance(advice, list):
            for tip in advice:
                st.success(f"✓ {tip}")
        else:
            st.info(advice)

    # 5. Download Tab
    with tab_download:
        st.subheader("📥 Export Your Meal Plan")
        st.markdown("Choose your preferred export format below:")

        col_pdf_card, col_json_card = st.columns(2)
        with col_pdf_card:
            st.markdown("""
            <div style="background:#1E293B; border:1px solid #38BDF8; border-radius:12px; padding:20px; text-align:center;">
                <div style="font-size:2.5rem; margin-bottom:8px;">📄</div>
                <h3 style="color:#F8FAFC; margin:0 0 6px 0;">PDF Document (Recommended)</h3>
                <p style="color:#94A3B8; font-size:0.88rem; margin-bottom:16px;">
                    Branded, print-ready document complete with meal schedules, recipes, exact portions, grocery checklist, and coaching tips.
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if pdf_bytes:
                st.download_button(
                    label="📄 Download Complete Plan as PDF",
                    data=pdf_bytes,
                    file_name=f"{clean_name}_Gorilla_Diet_Plan.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True,
                    key="tab_pdf_download"
                )
            elif pdf_err_msg:
                st.error(f"Could not generate PDF: {pdf_err_msg}")

        with col_json_card:
            st.markdown("""
            <div style="background:#1E293B; border:1px solid #334155; border-radius:12px; padding:20px; text-align:center;">
                <div style="font-size:2.5rem; margin-bottom:8px;">💻</div>
                <h3 style="color:#F8FAFC; margin:0 0 6px 0;">JSON Raw Data</h3>
                <p style="color:#94A3B8; font-size:0.88rem; margin-bottom:16px;">
                    Machine-readable structured format containing raw trainee requirements, calculated macronutrients, and recipe metadata.
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            plan_json_str = json.dumps(plan, indent=2)
            st.download_button(
                label="📥 Download Plan Data (JSON)",
                data=plan_json_str,
                file_name=f"{clean_name}_diet_plan.json",
                mime="application/json",
                use_container_width=True,
                key="tab_json_download"
            )

        with st.expander("🔍 Preview Structured JSON"):
            st.json(plan)
