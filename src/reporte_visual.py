import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import re
import io
import base64
import webbrowser
from collections import Counter
 
# ─────────────────────────────────────────────
# 0. CONFIGURACIÓN VISUAL GLOBAL
# ─────────────────────────────────────────────
 
COLOR_ROJO      = "#E63946"
COLOR_AZUL      = "#1D3557"
COLOR_AZUL_CLAR = "#457B9D"
COLOR_VERDE     = "#2A9D8F"
COLOR_FONDO     = "#F8F9FA"
COLOR_GRIS      = "#6C757D"
 
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "figure.facecolor" : COLOR_FONDO,
    "axes.facecolor"   : COLOR_FONDO,
    "axes.edgecolor"   : "#DEE2E6",
    "axes.titlesize"   : 14,
    "axes.titleweight" : "bold",
    "axes.labelsize"   : 11,
    "font.family"      : "sans-serif",
})
 
BASE_DIR    = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
DATA_DIR    = os.path.join(BASE_DIR, "data")
os.makedirs(REPORTS_DIR, exist_ok=True)
 
 
# ─────────────────────────────────────────────
# 1. UTILIDAD — CONVERTIR FIGURA A BASE64
# ─────────────────────────────────────────────
 
def fig_a_base64(fig):
    """
    Convierte una figura Matplotlib a string base64
    para embeber directamente en HTML sin archivos externos.
 
    Args:
        fig: Figura Matplotlib
 
    Returns:
        str: Imagen codificada en base64
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150,
                bbox_inches="tight", facecolor=COLOR_FONDO)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_b64
 
 
# ─────────────────────────────────────────────
# 2. CARGAR DATOS
# ─────────────────────────────────────────────
 
def cargar_datos():
    """
    Carga el dataset limpio generado por procesar_kaggle.py.
 
    Returns:
        pd.DataFrame: Dataset con columnas review_text, rating, brand, label
    """
    ruta = os.path.join(DATA_DIR, "reviews_clean.csv")
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encontro: {ruta}\n"
            "Ejecuta primero: python src/procesar_kaggle.py"
        )
    df = pd.read_csv(ruta, low_memory=False)
    print(f"[HardwareGuard] Dataset cargado: {len(df):,} resenas")
    return df
 
 
# ─────────────────────────────────────────────
# 3. GRAFICO 1 — PALABRAS CLAVE 1 ESTRELLA
# ─────────────────────────────────────────────
 
STOPWORDS = {
    'the','a','an','and','or','but','in','on','at','to','for','of','with',
    'is','was','are','were','be','been','have','has','had','do','does','did',
    'will','would','could','should','this','that','these','those','it','its',
    'i','my','me','we','our','you','your','he','she','they','their','not',
    'no','so','if','as','by','from','about','up','out','very','just','also',
    'more','most','some','any','all','get','got','one','two','use','used',
    'works','work','still','even','really','good','great','nice','love','like',
    'would','back','time','day','days','month','year','after','before','when',
    'than','then','there','here','what','which','well','much','many','too',
    'way','made','make','need','want','know','product','amazon','item',
    'order','bought','received','came','come','buy','purchased','using',
    'el','la','los','las','de','que','en','un','una','es','se','no','al',
    'con','por','su','para','pero','mas','lo','le','me','ya','muy'
}
 
def grafico_palabras_1estrella(df):
    """
    Barras horizontales con las 20 palabras mas frecuentes
    en resenas de 1 estrella (defectos graves).
 
    Args:
        df (pd.DataFrame): Dataset con review_text y rating
 
    Returns:
        str: Imagen en base64
    """
    print("[HardwareGuard] Generando grafico 1: Palabras clave 1 estrella...")
 
    resenas_1 = df[df["rating"] == 1]["review_text"].dropna().tolist()
 
    contador = Counter()
    for texto in resenas_1:
        if not isinstance(texto, str):
            continue
        texto = texto.lower()
        texto = re.sub(r"[^a-z\s]", " ", texto)
        tokens = [t for t in texto.split()
                  if t not in STOPWORDS and len(t) > 3]
        contador.update(tokens)
 
    top = contador.most_common(20)
    palabras    = [p[0] for p in top][::-1]
    frecuencias = [p[1] for p in top][::-1]
 
    fig, ax = plt.subplots(figsize=(11, 7))
    colores = sns.color_palette("Reds_r", len(palabras))
    bars = ax.barh(palabras, frecuencias, color=colores,
                   edgecolor="white", linewidth=0.5, height=0.7)
 
    for bar, freq in zip(bars, frecuencias):
        ax.text(bar.get_width() + max(frecuencias) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{freq:,}", va="center", ha="left",
                fontsize=9, color=COLOR_GRIS, fontweight="bold")
 
    ax.set_title("Top 20 Palabras en Resenas de 1 Estrella",
                 pad=15, color=COLOR_AZUL)
    ax.set_xlabel("Frecuencia de aparicion", color=COLOR_GRIS)
    ax.set_ylabel("Palabra clave", color=COLOR_GRIS)
    ax.axvline(x=0, color=COLOR_ROJO, linewidth=2)
    fig.text(0.99, 0.01, f"Basado en {len(resenas_1):,} resenas de 1 estrella",
             ha="right", fontsize=8, color=COLOR_GRIS, style="italic")
 
    plt.tight_layout()
    return fig_a_base64(fig)
 
 
# ─────────────────────────────────────────────
# 4. GRAFICO 2 — DISTRIBUCION DE RATINGS
# ─────────────────────────────────────────────
 
def grafico_distribucion_ratings(df):
    """
    Barras con distribucion de calificaciones 1-5.
    Rojo para zona de riesgo (1-2), verde para satisfechos (3-5).
 
    Args:
        df (pd.DataFrame): Dataset con columna rating
 
    Returns:
        str: Imagen en base64
    """
    print("[HardwareGuard] Generando grafico 2: Distribucion de ratings...")
 
    conteo      = df["rating"].value_counts().sort_index()
    ratings     = conteo.index.tolist()
    frecuencias = conteo.values.tolist()
    colores     = [COLOR_ROJO if r <= 2 else COLOR_VERDE for r in ratings]
    etiquetas   = [f"{r} estrella{'s' if r>1 else ''}" for r in ratings]
 
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(etiquetas, frecuencias, color=colores,
                  edgecolor="white", linewidth=1, width=0.6)
 
    for bar, freq in zip(bars, frecuencias):
        pct = freq / sum(frecuencias) * 100
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(frecuencias) * 0.01,
                f"{freq:,}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=9,
                color=COLOR_AZUL, fontweight="bold")
 
    parche_rojo  = mpatches.Patch(color=COLOR_ROJO,  label="Zona de Riesgo (1-2 estrellas)")
    parche_verde = mpatches.Patch(color=COLOR_VERDE, label="Satisfecho (3-5 estrellas)")
    ax.legend(handles=[parche_rojo, parche_verde], loc="upper left", framealpha=0.8)
    ax.set_title("Distribucion de Calificaciones del Dataset",
                 pad=15, color=COLOR_AZUL)
    ax.set_xlabel("Calificacion", color=COLOR_GRIS)
    ax.set_ylabel("Numero de resenas", color=COLOR_GRIS)
    fig.text(0.99, 0.01, f"Total: {len(df):,} resenas",
             ha="right", fontsize=8, color=COLOR_GRIS, style="italic")
 
    plt.tight_layout()
    return fig_a_base64(fig)
 
 
# ─────────────────────────────────────────────
# 5. GRAFICO 3 — TOP MARCAS CON DEFECTOS
# ─────────────────────────────────────────────
 
def grafico_marcas_defectos(df):
    """
    Doble grafico: cantidad absoluta y porcentaje de resenas
    negativas por marca (top 10 marcas con mas defectos).
 
    Args:
        df (pd.DataFrame): Dataset con brand, rating, label
 
    Returns:
        str: Imagen en base64
    """
    print("[HardwareGuard] Generando grafico 3: Top marcas con defectos...")
 
    conteo_total  = df["brand"].value_counts()
    marcas_validas = conteo_total[conteo_total >= 50].index
    df_f = df[df["brand"].isin(marcas_validas)].copy()
 
    stats = df_f.groupby("brand").agg(
        total     = ("rating", "count"),
        negativos = ("label", "sum"),
    ).reset_index()
    stats["pct_negativo"] = (stats["negativos"] / stats["total"] * 100).round(2)
    stats = stats.sort_values("pct_negativo", ascending=False).head(10)
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
 
    colores1 = sns.color_palette("Reds_r", len(stats))
    bars1 = ax1.barh(stats["brand"][::-1], stats["negativos"][::-1],
                     color=colores1, edgecolor="white", height=0.7)
    for bar, val in zip(bars1, stats["negativos"][::-1]):
        ax1.text(bar.get_width() + stats["negativos"].max() * 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 f"{int(val):,}", va="center", fontsize=9,
                 color=COLOR_GRIS, fontweight="bold")
    ax1.set_title("Resenas Negativas (cantidad)", color=COLOR_AZUL)
    ax1.set_xlabel("Numero de resenas 1-2 estrellas", color=COLOR_GRIS)
 
    colores2 = sns.color_palette("YlOrRd", len(stats))
    bars2 = ax2.barh(stats["brand"][::-1], stats["pct_negativo"][::-1],
                     color=colores2, edgecolor="white", height=0.7)
    for bar, val in zip(bars2, stats["pct_negativo"][::-1]):
        ax2.text(bar.get_width() + stats["pct_negativo"].max() * 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}%", va="center", fontsize=9,
                 color=COLOR_GRIS, fontweight="bold")
    ax2.set_title("% de Resenas Negativas", color=COLOR_AZUL)
    ax2.set_xlabel("Porcentaje de resenas 1-2 estrellas", color=COLOR_GRIS)
 
    fig.suptitle("Top 10 Marcas con Mayor Tasa de Defectos Reportados",
                 fontsize=14, fontweight="bold", color=COLOR_AZUL)
    fig.text(0.99, 0.01, "Solo marcas con 50 o mas resenas",
             ha="right", fontsize=8, color=COLOR_GRIS, style="italic")
 
    plt.tight_layout()
    return fig_a_base64(fig)
 
 
# ─────────────────────────────────────────────
# 6. GRAFICO 4 — MATRIZ DE CONFUSION
# ─────────────────────────────────────────────
 
def grafico_matriz_confusion():
    """
    Heatmap estilizado de la matriz de confusion
    con los resultados del modelo PR #4.
 
    Returns:
        str: Imagen en base64
    """
    print("[HardwareGuard] Generando grafico 4: Matriz de confusion...")
 
    cm    = np.array([[8463, 190], [170, 184]])
    total = cm.sum()
    cm_pct = cm / total * 100
 
    fig, ax = plt.subplots(figsize=(8, 6))
 
    sns.heatmap(cm, annot=False, cmap="Blues",
                xticklabels=["Sin Defecto (0)", "DEFECTO (1)"],
                yticklabels=["Sin Defecto (0)", "DEFECTO (1)"],
                linewidths=2, linecolor="white",
                ax=ax, cbar=False)
 
    celdas = [
        (0, 0, cm[0,0], cm_pct[0,0], "TN - Verdadero Negativo", COLOR_VERDE),
        (0, 1, cm[0,1], cm_pct[0,1], "FP - Falso Positivo",     COLOR_ROJO),
        (1, 0, cm[1,0], cm_pct[1,0], "FN - Falso Negativo",     COLOR_ROJO),
        (1, 1, cm[1,1], cm_pct[1,1], "TP - Verdadero Positivo", COLOR_VERDE),
    ]
 
    for (fila, col, valor, pct, etiqueta, color) in celdas:
        ax.text(col + 0.5, fila + 0.30, f"{valor:,}",
                ha="center", va="center", fontsize=18,
                fontweight="bold", color=color)
        ax.text(col + 0.5, fila + 0.55, f"({pct:.1f}%)",
                ha="center", va="center", fontsize=10, color="#555")
        ax.text(col + 0.5, fila + 0.78, etiqueta,
                ha="center", va="center", fontsize=8,
                color="#777", style="italic")
 
    ax.set_title(
        f"Matriz de Confusion — HardwareGuard\n"
        f"Accuracy: 96.00%  |  F1-Score: 0.5055  |  Total prueba: {total:,}",
        pad=15, color=COLOR_AZUL, fontsize=12)
    ax.set_xlabel("Prediccion del Modelo", fontsize=11, color=COLOR_GRIS)
    ax.set_ylabel("Valor Real", fontsize=11, color=COLOR_GRIS)
 
    plt.tight_layout()
    return fig_a_base64(fig)
 
 
# ─────────────────────────────────────────────
# 7. GENERAR REPORTE HTML
# ─────────────────────────────────────────────
 
def generar_html(df, img1, img2, img3, img4):
    """
    Genera el reporte HTML completo con los 4 graficos embebidos.
 
    Args:
        df            : DataFrame con estadisticas
        img1,2,3,4    : Imagenes en base64
 
    Returns:
        str: HTML completo como string
    """
    total       = len(df)
    negativos   = int(df["label"].sum())
    pct_neg     = df["label"].mean() * 100
    n_marcas    = df["brand"].nunique()
 
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HardwareGuard — Reporte de Defectos</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            background: #f0f2f5;
            color: #1D3557;
        }}
        header {{
            background: linear-gradient(135deg, #1D3557 0%, #457B9D 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        header h1 {{ font-size: 2.2rem; margin-bottom: 8px; }}
        header p  {{ font-size: 1rem; opacity: 0.85; }}
        .kpi-row {{
            display: flex;
            justify-content: center;
            gap: 20px;
            padding: 30px 40px;
            flex-wrap: wrap;
        }}
        .kpi {{
            background: white;
            border-radius: 12px;
            padding: 24px 36px;
            text-align: center;
            box-shadow: 0 2px 12px rgba(0,0,0,0.08);
            min-width: 160px;
        }}
        .kpi .valor {{
            font-size: 2rem;
            font-weight: 800;
            color: #1D3557;
        }}
        .kpi .valor.rojo {{ color: #E63946; }}
        .kpi .valor.verde {{ color: #2A9D8F; }}
        .kpi .etiqueta {{
            font-size: 0.82rem;
            color: #6C757D;
            margin-top: 4px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .seccion {{
            background: white;
            border-radius: 12px;
            margin: 0 40px 28px 40px;
            padding: 30px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        }}
        .seccion h2 {{
            font-size: 1.2rem;
            color: #1D3557;
            margin-bottom: 6px;
            padding-bottom: 10px;
            border-bottom: 3px solid #E63946;
            display: inline-block;
        }}
        .seccion p.desc {{
            color: #6C757D;
            font-size: 0.9rem;
            margin: 10px 0 20px 0;
        }}
        .seccion img {{
            width: 100%;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }}
        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 28px;
            margin: 0 40px 28px 40px;
        }}
        .alerta {{
            background: #fff5f5;
            border-left: 4px solid #E63946;
            padding: 16px 20px;
            border-radius: 0 8px 8px 0;
            margin: 0 40px 28px 40px;
            font-size: 0.92rem;
            color: #555;
        }}
        footer {{
            text-align: center;
            padding: 30px;
            color: #aaa;
            font-size: 0.82rem;
        }}
    </style>
</head>
<body>
 
<header>
    <h1>HardwareGuard — Detector de Defectos</h1>
    <p>Reporte Visual para el Departamento de Compras &nbsp;|&nbsp; Analisis de {total:,} resenas reales de Amazon</p>
</header>
 
<div class="kpi-row">
    <div class="kpi">
        <div class="valor">{total:,}</div>
        <div class="etiqueta">Total Resenas</div>
    </div>
    <div class="kpi">
        <div class="valor rojo">{negativos:,}</div>
        <div class="etiqueta">Resenas Negativas</div>
    </div>
    <div class="kpi">
        <div class="valor rojo">{pct_neg:.1f}%</div>
        <div class="etiqueta">Tasa de Defectos</div>
    </div>
    <div class="kpi">
        <div class="valor verde">96.00%</div>
        <div class="etiqueta">Accuracy Modelo</div>
    </div>
    <div class="kpi">
        <div class="valor">{n_marcas}</div>
        <div class="etiqueta">Marcas Analizadas</div>
    </div>
</div>
 
<div class="alerta">
    <strong>Alerta para Compras:</strong> Se detectaron {negativos:,} resenas con quejas graves
    ({pct_neg:.1f}% del total). Las palabras mas frecuentes en resenas de 1 estrella
    indican problemas recurrentes de hardware que podrian corresponder a lotes defectuosos.
    Se recomienda revision antes de nuevas ordenes de compra.
</div>
 
<div class="seccion">
    <h2>Palabras Clave en Resenas de 1 Estrella</h2>
    <p class="desc">
        Terminos mas frecuentes cuando un cliente califica con 1 estrella.
        Estas palabras son las senales de alerta que el modelo aprende a detectar.
    </p>
    <img src="data:image/png;base64,{img1}" alt="Palabras clave 1 estrella">
</div>
 
<div class="grid-2">
    <div class="seccion" style="margin:0">
        <h2>Distribucion de Calificaciones</h2>
        <p class="desc">
            Vista general del dataset. Las barras rojas (1-2 estrellas)
            representan la zona de riesgo de defecto de fabrica.
        </p>
        <img src="data:image/png;base64,{img2}" alt="Distribucion ratings">
    </div>
    <div class="seccion" style="margin:0">
        <h2>Matriz de Confusion del Modelo</h2>
        <p class="desc">
            Resultado del modelo NLP entrenado. Los Falsos Negativos (FN)
            son defectos no detectados — el indicador mas critico a minimizar.
        </p>
        <img src="data:image/png;base64,{img4}" alt="Matriz confusion">
    </div>
</div>
 
<div class="seccion">
    <h2>Top 10 Marcas con Mayor Tasa de Defectos</h2>
    <p class="desc">
        Marcas con mayor porcentaje y cantidad de resenas negativas.
        Recomendacion: priorizar auditoria de calidad en estas marcas antes de comprar lotes nuevos.
    </p>
    <img src="data:image/png;base64,{img3}" alt="Top marcas defectos">
</div>
 
<footer>
    HardwareGuard — Detector de Defectos via Analisis de Sentimiento &nbsp;|&nbsp;
    Dataset: Amazon Consumer Reviews (Kaggle) &nbsp;|&nbsp;
    Modelo: Red Neuronal PyTorch &nbsp;|&nbsp; Frank 2026
</footer>
 
</body>
</html>"""
 
 
# ─────────────────────────────────────────────
# 8. EJECUCION PRINCIPAL
# ─────────────────────────────────────────────
 
if __name__ == "__main__":
    print("=" * 65)
    print("  HARDWAREGUARD — GENERANDO REPORTE VISUAL")
    print("=" * 65 + "\n")
 
    # Cargar datos
    df = cargar_datos()
 
    # Generar los 4 graficos
    img1 = grafico_palabras_1estrella(df)
    img2 = grafico_distribucion_ratings(df)
    img3 = grafico_marcas_defectos(df)
    img4 = grafico_matriz_confusion()
 
    # Generar HTML
    print("[HardwareGuard] Generando reporte HTML...")
    html = generar_html(df, img1, img2, img3, img4)
 
    # Guardar HTML
    ruta_html = os.path.join(REPORTS_DIR, "reporte_hardwareguard.html")
    with open(ruta_html, "w", encoding="utf-8") as f:
        f.write(html)
 
    print(f"[HardwareGuard] Reporte guardado en: {ruta_html}")
 
    # Abrir automaticamente en el navegador
    print("[HardwareGuard] Abriendo en el navegador...")
    webbrowser.open(f"file:///{ruta_html.replace(os.sep, '/')}")
 
    print(f"\n{'=' * 65}")
    print(f"  PR #5 completado — Reporte HTML generado y abierto")
    print(f"  Archivo: reports/reporte_hardwareguard.html")
    print(f"{'=' * 65}")
 
import matplotlib.patches as mpatches
import seaborn as sns
import os
import re
from collections import Counter
 
# ─────────────────────────────────────────────
# 0. CONFIGURACIÓN VISUAL GLOBAL
# ─────────────────────────────────────────────
 
# Paleta de colores HardwareGuard
COLOR_ROJO      = "#E63946"
COLOR_AZUL      = "#1D3557"
COLOR_AZUL_CLAR = "#457B9D"
COLOR_VERDE     = "#2A9D8F"
COLOR_FONDO     = "#F8F9FA"
COLOR_GRIS      = "#6C757D"
 
# Estilo global de Seaborn
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "figure.facecolor" : COLOR_FONDO,
    "axes.facecolor"   : COLOR_FONDO,
    "axes.edgecolor"   : "#DEE2E6",
    "axes.titlesize"   : 14,
    "axes.titleweight" : "bold",
    "axes.labelsize"   : 11,
    "xtick.labelsize"  : 10,
    "ytick.labelsize"  : 10,
    "font.family"      : "sans-serif",
})
 
# Directorio de salida para los reportes
BASE_DIR    = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
DATA_DIR    = os.path.join(BASE_DIR, "data")
os.makedirs(REPORTS_DIR, exist_ok=True)
 
 
# ─────────────────────────────────────────────
# 1. CARGAR DATOS
# ─────────────────────────────────────────────
 
def cargar_datos():
    """
    Carga el dataset limpio generado por procesar_kaggle.py.
 
    Returns:
        pd.DataFrame: Dataset con columnas review_text, rating, brand, label
    
    Raises:
        FileNotFoundError: Si no existe reviews_clean.csv
    """
    ruta = os.path.join(DATA_DIR, "reviews_clean.csv")
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encontró: {ruta}\n"
            "Ejecuta primero: python src/procesar_kaggle.py"
        )
    df = pd.read_csv(ruta, low_memory=False)
    print(f"[HardwareGuard] Dataset cargado: {len(df):,} reseñas")
    return df
 
 
# ─────────────────────────────────────────────
# 2. GRÁFICO 1 — PALABRAS CLAVE 1 ESTRELLA
# ─────────────────────────────────────────────
 
STOPWORDS = {
    'the','a','an','and','or','but','in','on','at','to','for','of','with',
    'is','was','are','were','be','been','have','has','had','do','does','did',
    'will','would','could','should','this','that','these','those','it','its',
    'i','my','me','we','our','you','your','he','she','they','their','not',
    'no','so','if','as','by','from','about','up','out','very','just','also',
    'more','most','some','any','all','get','got','one','two','use','used',
    'product','amazon','item','order','bought','received','came','come',
    'buy','purchased','using','works','work','still','even','really','good',
    'great','nice','love','like','would','back','time','day','days','month',
    'year','after','before','when','than','then','there','here','what','which',
    'well','much','many','too','way','made','make','need','want','know','said',
    'el','la','los','las','de','que','en','un','una','es','se','no','al',
    'con','por','su','para','pero','más','lo','le','me','ya','muy'
}
 
def extraer_palabras_frecuentes(textos, top_n=20):
    """
    Extrae las palabras más frecuentes de una lista de textos.
    Filtra stopwords y tokens muy cortos.
 
    Args:
        textos (list): Lista de textos de reseñas
        top_n  (int) : Número de palabras a retornar. Default: 20
 
    Returns:
        tuple: (palabras, frecuencias) — listas ordenadas por frecuencia
    """
    contador = Counter()
    for texto in textos:
        if not isinstance(texto, str):
            continue
        # Limpiar y tokenizar
        texto = texto.lower()
        texto = re.sub(r"[^a-záéíóúüñ\s]", " ", texto)
        tokens = [t for t in texto.split()
                  if t not in STOPWORDS and len(t) > 3]
        contador.update(tokens)
 
    top = contador.most_common(top_n)
    palabras    = [p[0] for p in top]
    frecuencias = [p[1] for p in top]
    return palabras, frecuencias
 
 
def grafico_palabras_1estrella(df):
    """
    Gráfico de barras horizontal con las palabras más frecuentes
    en reseñas de 1 estrella (defectos graves).
 
    Este gráfico responde: ¿Qué palabras usa un cliente
    cuando su producto tiene un defecto de fábrica?
 
    Args:
        df (pd.DataFrame): Dataset con columnas review_text y rating
 
    Saves:
        reports/top_palabras_1estrella.png
    """
    print("[HardwareGuard] Generando gráfico 1: Palabras clave 1 estrella...")
 
    # Filtrar solo reseñas de 1 estrella
    resenas_1 = df[df["rating"] == 1]["review_text"].dropna().tolist()
    palabras, frecuencias = extraer_palabras_frecuentes(resenas_1, top_n=20)
 
    # Crear gradiente de colores rojo → naranja
    n = len(palabras)
    colores = [plt.cm.RdYlBu_r(i / n) for i in range(n)]
 
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(COLOR_FONDO)
 
    bars = ax.barh(palabras[::-1], frecuencias[::-1],
                   color=colores, edgecolor="white", linewidth=0.5, height=0.7)
 
    # Agregar valores al final de cada barra
    for bar, freq in zip(bars, frecuencias[::-1]):
        ax.text(bar.get_width() + max(frecuencias) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{freq:,}", va="center", ha="left",
                fontsize=9, color=COLOR_GRIS, fontweight="bold")
 
    # Títulos y etiquetas
    ax.set_title("🔴 Palabras Más Frecuentes en Reseñas de 1 Estrella",
                 pad=20, color=COLOR_AZUL)
    ax.set_xlabel("Frecuencia de aparición", color=COLOR_GRIS)
    ax.set_ylabel("Palabra clave", color=COLOR_GRIS)
 
    # Nota al pie
    fig.text(0.99, 0.01,
             f"HardwareGuard | Basado en {len(resenas_1):,} reseñas de 1 estrella",
             ha="right", fontsize=8, color=COLOR_GRIS, style="italic")
 
    # Línea de acento superior
    ax.axvline(x=0, color=COLOR_ROJO, linewidth=2)
 
    plt.tight_layout()
    ruta = os.path.join(REPORTS_DIR, "top_palabras_1estrella.png")
    plt.savefig(ruta, dpi=300, bbox_inches="tight", facecolor=COLOR_FONDO)
    plt.close()
    print(f"[HardwareGuard] ✅ Guardado: {ruta}")
 
 
# ─────────────────────────────────────────────
# 3. GRÁFICO 2 — DISTRIBUCIÓN DE RATINGS
# ─────────────────────────────────────────────
 
def grafico_distribucion_ratings(df):
    """
    Gráfico de barras con la distribución de calificaciones (1-5 estrellas).
    Las barras 1-2 se muestran en rojo (zona de riesgo).
 
    Args:
        df (pd.DataFrame): Dataset con columna rating
 
    Saves:
        reports/distribucion_ratings.png
    """
    print("[HardwareGuard] Generando gráfico 2: Distribución de ratings...")
 
    conteo = df["rating"].value_counts().sort_index()
    ratings     = conteo.index.tolist()
    frecuencias = conteo.values.tolist()
    colores = [COLOR_ROJO if r <= 2 else COLOR_VERDE for r in ratings]
 
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(COLOR_FONDO)
 
    bars = ax.bar([str(r) for r in ratings], frecuencias,
                  color=colores, edgecolor="white", linewidth=1, width=0.6)
 
    # Etiquetas encima de cada barra
    for bar, freq in zip(bars, frecuencias):
        pct = freq / sum(frecuencias) * 100
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(frecuencias) * 0.01,
                f"{freq:,}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=9,
                color=COLOR_AZUL, fontweight="bold")
 
    # Leyenda
    parche_rojo  = mpatches.Patch(color=COLOR_ROJO,  label="Zona de Riesgo (1-2 ⭐)")
    parche_verde = mpatches.Patch(color=COLOR_VERDE, label="Satisfecho (3-5 ⭐)")
    ax.legend(handles=[parche_rojo, parche_verde], loc="upper left", framealpha=0.8)
 
    ax.set_title("📊 Distribución de Calificaciones del Dataset",
                 pad=20, color=COLOR_AZUL)
    ax.set_xlabel("Calificación (estrellas)", color=COLOR_GRIS)
    ax.set_ylabel("Número de reseñas", color=COLOR_GRIS)
 
    fig.text(0.99, 0.01,
             f"HardwareGuard | Total: {len(df):,} reseñas",
             ha="right", fontsize=8, color=COLOR_GRIS, style="italic")
 
    plt.tight_layout()
    ruta = os.path.join(REPORTS_DIR, "distribucion_ratings.png")
    plt.savefig(ruta, dpi=300, bbox_inches="tight", facecolor=COLOR_FONDO)
    plt.close()
    print(f"[HardwareGuard] ✅ Guardado: {ruta}")
 
 
# ─────────────────────────────────────────────
# 4. GRÁFICO 3 — TOP MARCAS CON DEFECTOS
# ─────────────────────────────────────────────
 
def grafico_marcas_defectos(df):
    """
    Gráfico de barras con las 10 marcas que tienen más reseñas
    de 1-2 estrellas en términos absolutos y porcentuales.
 
    Args:
        df (pd.DataFrame): Dataset con columnas brand, rating, label
 
    Saves:
        reports/top_marcas_defectos.png
    """
    print("[HardwareGuard] Generando gráfico 3: Top marcas con defectos...")
 
    # Filtrar marcas con al menos 50 reseñas para ser representativas
    conteo_total = df["brand"].value_counts()
    marcas_validas = conteo_total[conteo_total >= 50].index
 
    df_filtrado = df[df["brand"].isin(marcas_validas)].copy()
 
    # Calcular % de reseñas negativas por marca
    stats = df_filtrado.groupby("brand").agg(
        total     = ("rating", "count"),
        negativos = ("label", "sum"),
    ).reset_index()
    stats["pct_negativo"] = (stats["negativos"] / stats["total"] * 100).round(2)
    stats = stats.sort_values("pct_negativo", ascending=False).head(10)
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor(COLOR_FONDO)
 
    # ── Subplot 1: Cantidad absoluta ─────────
    colores1 = sns.color_palette("Reds_r", len(stats))
    bars1 = ax1.barh(stats["brand"][::-1], stats["negativos"][::-1],
                     color=colores1, edgecolor="white", height=0.7)
    for bar, val in zip(bars1, stats["negativos"][::-1]):
        ax1.text(bar.get_width() + stats["negativos"].max() * 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 f"{int(val):,}", va="center", fontsize=9,
                 color=COLOR_GRIS, fontweight="bold")
    ax1.set_title("Reseñas Negativas (cantidad)", color=COLOR_AZUL)
    ax1.set_xlabel("Número de reseñas 1-2 ⭐", color=COLOR_GRIS)
 
    # ── Subplot 2: Porcentaje ────────────────
    colores2 = sns.color_palette("YlOrRd", len(stats))
    bars2 = ax2.barh(stats["brand"][::-1], stats["pct_negativo"][::-1],
                     color=colores2, edgecolor="white", height=0.7)
    for bar, val in zip(bars2, stats["pct_negativo"][::-1]):
        ax2.text(bar.get_width() + stats["pct_negativo"].max() * 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}%", va="center", fontsize=9,
                 color=COLOR_GRIS, fontweight="bold")
    ax2.set_title("% de Reseñas Negativas", color=COLOR_AZUL)
    ax2.set_xlabel("Porcentaje de reseñas 1-2 ⭐", color=COLOR_GRIS)
 
    fig.suptitle("🏭 Top 10 Marcas con Mayor Tasa de Defectos Reportados",
                 fontsize=15, fontweight="bold", color=COLOR_AZUL, y=1.02)
 
    fig.text(0.99, 0.01,
             "HardwareGuard | Solo marcas con ≥50 reseñas",
             ha="right", fontsize=8, color=COLOR_GRIS, style="italic")
 
    plt.tight_layout()
    ruta = os.path.join(REPORTS_DIR, "top_marcas_defectos.png")
    plt.savefig(ruta, dpi=300, bbox_inches="tight", facecolor=COLOR_FONDO)
    plt.close()
    print(f"[HardwareGuard] ✅ Guardado: {ruta}")
 
 
# ─────────────────────────────────────────────
# 5. GRÁFICO 4 — MATRIZ DE CONFUSIÓN VISUAL
# ─────────────────────────────────────────────
 
def grafico_matriz_confusion():
    """
    Genera una matriz de confusión visual y estilizada
    con los resultados del modelo entrenado en el PR #4.
 
    Valores hardcodeados del último entrenamiento:
        TN=8463, FP=190, FN=170, TP=184
        Accuracy=96.00%, F1=0.5055
 
    Saves:
        reports/matriz_confusion.png
    """
    print("[HardwareGuard] Generando gráfico 4: Matriz de confusión...")
 
    # Valores del entrenamiento PR #4
    cm = np.array([[8463, 190],
                   [170,  184]])
 
    etiquetas   = ["Sin Defecto (0)", "DEFECTO (1)"]
    total       = cm.sum()
    cm_pct      = cm / total * 100
 
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(COLOR_FONDO)
 
    # Heatmap con Seaborn
    mascara_colores = np.array([
        [COLOR_VERDE, COLOR_ROJO],
        [COLOR_ROJO,  COLOR_VERDE]
    ])
 
    sns.heatmap(
        cm, annot=False, fmt="d",
        cmap="Blues",
        xticklabels=etiquetas,
        yticklabels=etiquetas,
        linewidths=2, linecolor="white",
        ax=ax, cbar=False
    )
 
    # Anotaciones personalizadas en cada celda
    celdas = [
        (0, 0, cm[0,0], cm_pct[0,0], "TN\nVerdadero Negativo", COLOR_VERDE, "✅"),
        (0, 1, cm[0,1], cm_pct[0,1], "FP\nFalso Positivo",     COLOR_ROJO,  "⚠️"),
        (1, 0, cm[1,0], cm_pct[1,0], "FN\nFalso Negativo",     COLOR_ROJO,  "🚨"),
        (1, 1, cm[1,1], cm_pct[1,1], "TP\nVerdadero Positivo", COLOR_VERDE, "✅"),
    ]
 
    for (fila, col, valor, pct, etiqueta, color, icono) in celdas:
        ax.text(col + 0.5, fila + 0.30, f"{icono} {valor:,}",
                ha="center", va="center", fontsize=16,
                fontweight="bold", color=color)
        ax.text(col + 0.5, fila + 0.55, f"({pct:.1f}%)",
                ha="center", va="center", fontsize=10, color="#555555")
        ax.text(col + 0.5, fila + 0.78, etiqueta,
                ha="center", va="center", fontsize=8,
                color="#777777", style="italic")
 
    ax.set_title(
        f"🧠 Matriz de Confusión — HardwareGuard\n"
        f"Accuracy: 96.00%  |  F1-Score: 0.5055  |  Total reseñas prueba: {total:,}",
        pad=20, color=COLOR_AZUL, fontsize=13
    )
    ax.set_xlabel("Predicción del Modelo", fontsize=11, color=COLOR_GRIS)
    ax.set_ylabel("Valor Real", fontsize=11, color=COLOR_GRIS)
 
    fig.text(0.99, 0.01,
             "HardwareGuard | Entrenado con dataset Amazon Consumer Reviews (Kaggle)",
             ha="right", fontsize=8, color=COLOR_GRIS, style="italic")
 
    plt.tight_layout()
    ruta = os.path.join(REPORTS_DIR, "matriz_confusion.png")
    plt.savefig(ruta, dpi=300, bbox_inches="tight", facecolor=COLOR_FONDO)
    plt.close()
    print(f"[HardwareGuard] ✅ Guardado: {ruta}")
 
 
# ─────────────────────────────────────────────
# 6. EJECUCIÓN PRINCIPAL
# ─────────────────────────────────────────────
 
if __name__ == "__main__":
    print("=" * 65)
    print("  📊 HARDWAREGUARD — GENERANDO REPORTE VISUAL")
    print("=" * 65 + "\n")
 
    # Cargar datos
    df = cargar_datos()
 
    # Generar los 4 gráficos
    grafico_palabras_1estrella(df)
    grafico_distribucion_ratings(df)
    grafico_marcas_defectos(df)
    grafico_matriz_confusion()
 
    print(f"\n{'=' * 65}")
    print(f"  ✅ PR #5 completado — Reporte visual generado")
    print(f"  📁 Archivos guardados en: reports/")
    print(f"     → top_palabras_1estrella.png")
    print(f"     → distribucion_ratings.png")
    print(f"     → top_marcas_defectos.png")
    print(f"     → matriz_confusion.png")
    print(f"{'=' * 65}")