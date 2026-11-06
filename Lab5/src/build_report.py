from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _register_fonts() -> None:
    """Подбирает TTF-шрифт с кириллицей под текущую ОС."""
    candidates_regular: list[str] = []
    candidates_bold: list[str] = []

    if sys.platform.startswith("win"):
        win_fonts = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
        candidates_regular += [
            os.path.join(win_fonts, "arial.ttf"),
            os.path.join(win_fonts, "segoeui.ttf"),
        ]
        candidates_bold += [
            os.path.join(win_fonts, "arialbd.ttf"),
            os.path.join(win_fonts, "segoeuib.ttf"),
        ]
    elif sys.platform == "darwin":
        candidates_regular.append("/System/Library/Fonts/Supplemental/Arial.ttf")
        candidates_bold.append("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
    else:
        candidates_regular += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
        candidates_bold += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]

    regular = next((p for p in candidates_regular if os.path.exists(p)), None)
    bold = next((p for p in candidates_bold if os.path.exists(p)), None)
    if regular is None or bold is None:
        raise FileNotFoundError(
            "Не найден TTF-шрифт с кириллицей. "
            f"Проверенные пути: {candidates_regular + candidates_bold}"
        )
    pdfmetrics.registerFont(TTFont("Arial", regular))
    pdfmetrics.registerFont(TTFont("Arial-Bold", bold))


_register_fonts()


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
    canvas.line(2 * cm, 1.35 * cm, 19 * cm, 1.35 * cm)
    canvas.setFont("Arial", 8)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(2 * cm, 0.95 * cm, "Разделительная кластеризация")
    canvas.drawRightString(19 * cm, 0.95 * cm, str(doc.page))
    canvas.restoreState()


def scaled_image(path: Path, max_width: float, max_height: float) -> Image:
    with PILImage.open(path) as source:
        width, height = source.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def fmt(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def pct(value: float) -> str:
    return f"{100 * value:.0f}%"


def make_table(data, widths, *, header=True, font_size=8.6, align="CENTER") -> Table:
    style = [
        ("FONT", (0, 0), (-1, -1), "Arial", font_size),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B7CBD2")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), align),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        style += [
            ("FONT", (0, 0), (-1, 0), "Arial-Bold", font_size),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#246B78")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F8FA")]),
        ]
    return Table(data, colWidths=widths, repeatRows=1 if header else 0, style=TableStyle(style))


def main() -> None:
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(RESULTS / "experiment_metrics.csv")

    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body", parent=styles["BodyText"], fontName="Arial", fontSize=10.2,
        leading=14.6, alignment=TA_JUSTIFY, spaceAfter=7,
    )
    h1 = ParagraphStyle(
        "H1", parent=styles["Heading1"], fontName="Arial-Bold", fontSize=18,
        leading=22, textColor=colors.HexColor("#17324D"), spaceBefore=7, spaceAfter=10,
    )
    h2 = ParagraphStyle(
        "H2", parent=styles["Heading2"], fontName="Arial-Bold", fontSize=13.2,
        leading=16, textColor=colors.HexColor("#246B78"), spaceBefore=8, spaceAfter=6,
    )
    caption = ParagraphStyle(
        "Caption", parent=body, fontSize=8.8, leading=11.5, alignment=TA_CENTER,
        textColor=colors.HexColor("#475569"), spaceBefore=4, spaceAfter=9,
    )
    bullet = ParagraphStyle("Bullet", parent=body, leftIndent=15, firstLineIndent=-8, bulletIndent=4)
    formula = ParagraphStyle(
        "Formula", parent=body, fontSize=9.2, leading=13, leftIndent=8,
        backColor=colors.HexColor("#F1F5F9"), borderPadding=7,
    )
    note = ParagraphStyle(
        "Note", parent=body, fontSize=9.2, leading=13.2,
        backColor=colors.HexColor("#FFF7E0"), borderColor=colors.HexColor("#E7B94B"),
        borderWidth=0.5, borderPadding=7,
    )

    # Документ собирается в память, а не в файл
    buffer = io.BytesIO()
    doc = BaseDocTemplate(
        buffer, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=1.7 * cm, bottomMargin=1.75 * cm,
        title="Разделительная кластеризация", author="Лабораторная работа",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates(PageTemplate(id="default", frames=frame, onPage=footer))
    story = []

    title = ParagraphStyle("Title", parent=h1, alignment=TA_CENTER, fontSize=25, leading=31, spaceAfter=15)
    kicker = ParagraphStyle(
        "Kicker", parent=body, alignment=TA_CENTER, fontName="Arial-Bold",
        fontSize=12, textColor=colors.HexColor("#246B78"),
    )
    center = ParagraphStyle("Center", parent=body, alignment=TA_CENTER)
    story += [
        Spacer(1, 2.0 * cm),
        Paragraph("ОТЧЁТ ПО ЛАБОРАТОРНОЙ РАБОТЕ", kicker),
        Spacer(1, 0.7 * cm),
        Paragraph("Разделительная кластеризация", title),
        Spacer(1, 0.6 * cm),
        make_table([
            ["Набор данных", "customers, сведения о клиентах банка"],
            ["Алгоритмы", "k-Means и k-Medoids"],
            ["Число кластеров", "k = 3, 4, 5, 6, 7, 8, 9"],
            ["Доля шума", "0%, 1%, 3%, 5%, 10%"],
            ["Количество запусков", "70"],
        ], [5.1 * cm, 10.7 * cm], header=False, font_size=10, align="LEFT"),
        Spacer(1, 3.0 * cm),
        Paragraph("Кластеризация выполнена в полном пространстве признаков; двумерные проекции используются только для визуализации.", center),
        PageBreak(),
    ]

    story += [Paragraph("1. Формулировка задания", h1)]
    tasks = [
        "Разработать программу кластеризации заданного набора данных алгоритмами k-Means и k-Medoids. Параметры программы: набор данных и число кластеров. Результат должен содержать координаты точек, номера кластеров и ошибку кластеризации.",
        "Провести эксперименты на наборе customers и построить точечные диаграммы принадлежности к кластерам, а также графики зависимости ошибки от k.",
        "Добавить параметр доли зашумленных объектов и случайное изменение координат выбранной части набора.",
        "Повторить эксперименты для долей шума 1%, 3%, 5% и 10% при k от 3 до 9, визуализировать и объяснить результаты.",
        "Подготовить связный отчёт с формулировкой, рисунками и выводами.",
    ]
    for item in tasks:
        story.append(Paragraph("• " + item, bullet))
    story += [
        Paragraph("2. Материалы проекта", h1),
        Paragraph(
            "Исходные тексты, набор данных, таблицы назначений кластеров, метрики и рисунки находятся "
            "в каталоге lab5/ проекта (подкаталоги src/, data/, results/).", body,
        ),
        Paragraph(
            "Архив содержит 850 строк и 10 столбцов. Row является номером строки, CustomerId - уникальным идентификатором; "
            "эти два столбца не несут содержательной информации о сходстве клиентов и исключены из расчётов. "
            "Оставлены восемь признаков: Age, Education, YearsEmployed, Income, CardDebt, OtherDebt, Defaulted и DebtIncomeRatio.", body,
        ),
        Paragraph("Описание исходных признаков", h2),
        make_table([
            ["Признак", "Смысл"],
            ["Age", "возраст, полных лет"],
            ["Education", "целочисленный код образования"],
            ["YearsEmployed", "трудовой стаж, полных лет"],
            ["Income", "доход, тыс. долларов в год"],
            ["CardDebt", "задолженность по карте"],
            ["OtherDebt", "прочая задолженность"],
            ["Defaulted", "признак банкротства: 1 или 0"],
            ["DebtIncomeRatio", "отношение задолженности к доходу"],
        ], [5.0 * cm, 11.0 * cm], font_size=9, align="LEFT"),
        PageBreak(),
        Paragraph("3. Подготовка данных", h1),
        Paragraph(
            "В столбце Defaulted отсутствуют 150 значений. Они заменяются медианой столбца. После этого каждый признак "
            "стандартизуется: из значения вычитается среднее и результат делится на стандартное отклонение. "
            "Стандартизация нужна потому, что Income измеряется десятками и сотнями, а долговые показатели могут быть единицами; "
            "без неё признаки с большим численным диапазоном почти полностью определяли бы евклидово расстояние.", body,
        ),
        Paragraph("z=(x-mean)/std", formula),
        Paragraph("Внесение шума", h2),
        Paragraph(
            "Для каждой заданной доли воспроизводимо выбирается соответствующее число объектов: 8, 26, 42 или 85 из 850. "
            "У каждого выбранного объекта случайно определяется от одной до восьми координат. К ним добавляются независимые "
            "значения из нормального распределения N(0, 3) уже в стандартизованном пространстве. Один и тот же seed 42 обеспечивает "
            "повторяемость. На точечных графиках изменённые объекты обведены чёрным контуром.", body,
        ),
        Paragraph(
            "Такой шум является сильным: стандартное отклонение добавки в три раза превышает стандартное отклонение исходной "
            "координаты. Поэтому даже небольшая доля изменённых строк позволяет явно увидеть устойчивость методов.", note,
        ),
        Paragraph("4. Алгоритмы и критерии", h1),
        Paragraph("k-Means", h2),
        Paragraph(
            "Алгоритм поочерёдно назначает объект ближайшему центроиду и заменяет каждый центроид средним объектов кластера. "
            "Использована инициализация k-means++ и 20 независимых стартов; сохраняется решение с наименьшей ошибкой.", body,
        ),
        Paragraph("k-Medoids", h2),
        Paragraph(
            "В k-Medoids центр каждого кластера обязан быть реальным объектом набора. После назначения точек ближайшему медоиду "
            "новым медоидом становится объект с минимальной суммой расстояний до остальных объектов своего кластера. "
            "Инициализация аналогична k-means++, число итераций ограничено 100.", body,
        ),
        Paragraph("Ошибка и дополнительная оценка", h2),
        Paragraph(
            "Для сопоставимости методов ошибка определена одинаково как сумма квадратов евклидовых расстояний от каждой точки "
            "до назначенного центра. Для k-Means центром является центроид, для k-Medoids - медоид.", body,
        ),
        Paragraph("SSE = sum(i=1..n) ||x_i - c_cluster(i)||^2", formula),
        Paragraph(
            "SSE неизбежно не возрастает при увеличении k, поэтому она не определяет оптимальное k сама по себе. "
            "Для интерпретации дополнительно вычислен коэффициент силуэта: большее значение означает более компактные и лучше разделённые кластеры.", body,
        ),
        PageBreak(),
        Paragraph("5. Эксперимент без шума", h1),
    ]

    clean = metrics[metrics.noise_fraction == 0]
    table = [["Метод", "k", "SSE", "Силуэт", "Мин. кластер", "Макс. кластер"]]
    for _, row in clean.iterrows():
        table.append([
            "k-Means" if row.algorithm == "kmeans" else "k-Medoids", str(int(row.k)),
            fmt(row.error_sse), fmt(row.silhouette, 3), str(int(row.min_cluster_size)), str(int(row.max_cluster_size)),
        ])
    story += [
        make_table(table, [3.2 * cm, 1.3 * cm, 3.0 * cm, 2.2 * cm, 3.0 * cm, 3.0 * cm], font_size=8.2),
        Spacer(1, 8),
        Paragraph(
            "У обоих методов SSE монотонно снижается с ростом k, поскольку дополнительные центры уменьшают расстояние до ближайшего "
            "представителя. Для k-Means уменьшение наиболее заметно между k=3 и k=4: 4339,0 до 3763,0. Наибольший силуэт k-Means "
            "равен 0,287 при k=3. Для k-Medoids максимум силуэта 0,200 достигнут при k=7, но значения в целом ниже.", body,
        ),
        Paragraph(
            "При одинаковом k ошибка k-Means ниже, чем k-Medoids. Это ожидаемо: среднее арифметическое минимизирует сумму квадратов "
            "расстояний, тогда как ограничение центра реальным объектом не позволяет k-Medoids достичь меньшей SSE.", body,
        ),
        scaled_image(RESULTS / "error_by_k.png", 17 * cm, 9.1 * cm),
        Paragraph("Рисунок 1. Зависимость SSE от k для обоих алгоритмов и всех долей шума.", caption),
        PageBreak(),
        Paragraph("5.1. Точечная визуализация", h2),
        scaled_image(RESULTS / "clusters_clean_all_k.png", 17 * cm, 8.5 * cm),
        Paragraph("Рисунок 2. Цвет точки показывает кластер при k от 3 до 9 на данных без шума.", caption),
        Paragraph(
            "Для рисунка данные спроецированы методом главных компонент (PCA) на плоскость. Цвета отражают результат кластеризации "
            "в исходном восьмимерном пространстве, а не повторную кластеризацию двух координат. Поэтому визуально соседние точки иногда "
            "имеют разные цвета: часть различий находится в компонентах, не показанных на плоскости.", body,
        ),
        Paragraph(
            "При малых k k-Means формирует один крупный центральный кластер и несколько групп с более высокими значениями долговых "
            "или доходных признаков. С ростом k крупные области дробятся, а минимальный размер кластера уменьшается до 8 объектов при k=9. "
            "Это объясняет снижение силуэта: некоторые новые группы описывают локальные или редкие профили, а не хорошо отделённые сегменты.", body,
        ),
        PageBreak(),
        Paragraph("6. Эксперимент с шумом", h1),
        Paragraph("Сравнение при k=5", h2),
    ]
    k5 = metrics[metrics.k == 5]
    table = [["Метод", "Шум", "Изменено", "SSE", "Рост SSE", "Силуэт"]]
    for algorithm in ("kmeans", "kmedoids"):
        base = float(k5[(k5.algorithm == algorithm) & (k5.noise_fraction == 0)].error_sse.iloc[0])
        for _, row in k5[k5.algorithm == algorithm].sort_values("noise_fraction").iterrows():
            growth = (row.error_sse / base - 1) * 100
            table.append([
                "k-Means" if algorithm == "kmeans" else "k-Medoids", pct(row.noise_fraction),
                str(int(row.noisy_rows)), fmt(row.error_sse), fmt(growth) + "%", fmt(row.silhouette, 3),
            ])
    story += [
        make_table(table, [3.0 * cm, 1.6 * cm, 2.3 * cm, 3.0 * cm, 3.0 * cm, 2.4 * cm], font_size=8.2),
        Spacer(1, 8),
        Paragraph(
            "При k=5 увеличение доли шума от 0% до 10% повышает SSE k-Means с 3411,1 до 5974,7, то есть на 75,2%. "
            "Для k-Medoids ошибка возрастает с 3927,3 до 6828,8, на 73,9%. Коэффициент силуэта k-Means снижается с 0,250 до 0,213, "
            "что отражает размывание структуры. У k-Medoids силуэт меняется немонотонно, поскольку отдельные выбросы могут становиться "
            "маленькими кластерами или медоидами; при 10% шума минимальный кластер содержит один объект.", body,
        ),
        Paragraph(
            "Нельзя заключать об устойчивости только по абсолютной SSE: k-Medoids оптимизирует сумму обычных расстояний через выбор "
            "реальных представителей, а в отчёте для сравнения показана квадратная ошибка. Поэтому одновременно анализируются силуэт, "
            "размеры кластеров и визуальная структура.", note,
        ),
        PageBreak(),
        scaled_image(RESULTS / "clusters_by_noise_k5.png", 14.8 * cm, 20.8 * cm),
        Paragraph("Рисунок 3. Кластеры при k=5 для всех уровней шума; изменённые объекты обведены чёрным.", caption),
        PageBreak(),
        Paragraph("6.1. Чувствительность к шуму", h2),
        scaled_image(RESULTS / "noise_sensitivity.png", 17 * cm, 8.1 * cm),
        Paragraph("Рисунок 4. Относительный рост SSE по сравнению с соответствующим запуском без шума.", caption),
        Paragraph(
            "Рост SSE усиливается как с долей шума, так и в большинстве случаев с k. При 10% шума для k-Means относительное "
            "увеличение меняется от 61,3% при k=3 до 92,9% при k=9. Для k-Medoids соответствующий диапазон составляет 44,2-85,8%, "
            "а при k=9 наблюдается 77,8%. Дополнительные кластеры частично выделяют выбросы в отдельные группы, но оставшаяся "
            "неоднородность внутри кластеров продолжает увеличивать квадратную ошибку.", body,
        ),
        Paragraph(
            "Кривые ошибки на рисунке 1 сохраняют общий нисходящий вид при любом уровне шума. Следовательно, эффект k и эффект шума "
            "действуют в противоположных направлениях: большее k уменьшает расстояния до центра, а большее число искажённых объектов "
            "увеличивает их. Стабильного единственного локтя для всех сценариев нет.", body,
        ),
        Paragraph("7. Выводы", h1),
        Paragraph(
            "Разработана параметризуемая программа для k-Means и k-Medoids. Она загружает CSV, подготавливает признаки, при необходимости "
            "зашумляет заданную долю объектов, сохраняет стандартизованные координаты и назначенные кластеры и выводит SSE, размеры групп, "
            "центры или медоиды. Для customers выполнены 70 запусков при k от 3 до 9 и пяти уровнях шума.", body,
        ),
        Paragraph(
            "На исходных данных k-Means даёт наибольший силуэт 0,287 при k=3. Это наиболее обоснованный выбор, если главным критерием "
            "считать разделённость сегментов. Более крупные k уменьшают SSE, но создают малые группы и не улучшают силуэт. "
            "k-Medoids имеет более высокую квадратную ошибку, зато его центры являются реальными клиентами и могут быть проще для предметной интерпретации.", body,
        ),
        Paragraph(
            "Внесение сильного шума закономерно повышает ошибку и визуально создаёт удалённые точки. При k=5 и 10% шума SSE выросла "
            "примерно на три четверти у обоих методов. k-Means демонстрирует плавное ухудшение силуэта; k-Medoids иногда выделяет выбросы "
            "в очень малые кластеры, поэтому для практического применения полезно контролировать минимальный размер сегмента и повторять "
            "запуски с несколькими seed.", body,
        ),
    ]
    doc.build(story)
    buffer.close()
    print("Отчёт собран в памяти (файл не сохранён)")


if __name__ == "__main__":
    main()