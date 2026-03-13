import pandas as pd
import numpy as np
import re
import os
import nltk
 
# ─────────────────────────────────────────────
# 1. DESCARGAR RECURSOS NLTK
# ─────────────────────────────────────────────
 
def descargar_recursos_nltk():
    """
    Descarga los recursos necesarios de NLTK si no están disponibles.
    Solo se descarga una vez, luego se reutiliza del cache local.
 
    Recursos descargados:
        - stopwords : Palabras vacías en español
        - punkt     : Tokenizador de oraciones/palabras
        - punkt_tab : Tokenizador actualizado (NLTK >= 3.8)
    """
    print("[HardwareGuard] Verificando recursos NLTK...")
    recursos = ["stopwords", "punkt", "punkt_tab"]
    for recurso in recursos:
        try:
            nltk.download(recurso, quiet=True)
        except Exception:
            pass
    print("[HardwareGuard] Recursos NLTK listos ✅")
 
 
# ─────────────────────────────────────────────
# 2. INICIALIZAR HERRAMIENTAS NLTK
# ─────────────────────────────────────────────
 
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
 
# Stemmer en español — reduce palabras a su raíz morfológica
# Ejemplo: "fallando" → "fall", "defectuoso" → "defectu"
STEMMER = SnowballStemmer("spanish")
 
# Stopwords en español — palabras vacías que no aportan significado
# Ejemplo: "el", "la", "de", "que", "y", "en"
STOPWORDS_ES = set(stopwords.words("spanish"))
 
 
# ─────────────────────────────────────────────
# 3. FUNCIONES DEL PIPELINE
# ─────────────────────────────────────────────
 
def paso1_minusculas(texto):
    """
    PASO 1: Convierte todo el texto a minúsculas.
 
    Ejemplo:
        "DEFECTO DE FÁBRICA" → "defecto de fábrica"
 
    Args:
        texto (str): Texto original
 
    Returns:
        str: Texto en minúsculas
    """
    return texto.lower() if isinstance(texto, str) else ""
 
 
def paso2_limpiar_caracteres(texto):
    """
    PASO 2: Elimina caracteres especiales, puntuación y números.
    Conserva letras (incluye acentos y ñ) y espacios.
 
    Ejemplo:
        "¡¡terrible!! llegó con 3 fallas..." → "terrible llegó con fallas"
 
    Args:
        texto (str): Texto en minúsculas
 
    Returns:
        str: Texto sin caracteres especiales
    """
    # Conservar letras con acento, ñ y espacios
    texto = re.sub(r"[^a-záéíóúüñ\s]", " ", texto)
    # Eliminar espacios múltiples
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto
 
 
def paso3_tokenizar(texto):
    """
    PASO 3: Divide el texto en tokens (palabras individuales).
 
    Ejemplo:
        "producto defectuoso de fábrica" → ["producto", "defectuoso", "de", "fábrica"]
 
    Args:
        texto (str): Texto limpio
 
    Returns:
        list: Lista de tokens (palabras)
    """
    try:
        return word_tokenize(texto, language="spanish")
    except Exception:
        return texto.split()
 
 
def paso4_eliminar_stopwords(tokens):
    """
    PASO 4: Elimina palabras vacías (stopwords) en español.
    Las stopwords no aportan significado al análisis de sentimiento.
 
    Ejemplo:
        ["el", "producto", "es", "muy", "defectuoso"]
        → ["producto", "defectuoso"]
 
    Args:
        tokens (list): Lista de tokens
 
    Returns:
        list: Tokens sin stopwords
    """
    return [t for t in tokens if t not in STOPWORDS_ES and len(t) > 2]
 
 
def paso5_stemming(tokens):
    """
    PASO 5: Aplica Stemming — reduce cada palabra a su raíz morfológica.
    Esto agrupa palabras con el mismo significado base.
 
    Ejemplo:
        ["fallando", "fallas", "falló"] → ["fall", "fall", "fall"]
        ["defectuoso", "defecto", "defectos"] → ["defectu", "defect", "defect"]
 
    Args:
        tokens (list): Lista de tokens sin stopwords
 
    Returns:
        list: Tokens reducidos a su raíz
    """
    return [STEMMER.stem(t) for t in tokens]
 
 
def pipeline_completo(texto):
    """
    Aplica el pipeline completo de limpieza a un texto.
 
    Pipeline:
        Texto crudo
            ↓ Paso 1: Minúsculas
            ↓ Paso 2: Eliminar caracteres especiales
            ↓ Paso 3: Tokenizar
            ↓ Paso 4: Eliminar stopwords
            ↓ Paso 5: Stemming
        Texto procesado
 
    Args:
        texto (str): Texto crudo de la reseña
 
    Returns:
        str: Texto completamente procesado listo para vectorización
    """
    if not isinstance(texto, str) or texto.strip() == "":
        return ""
 
    texto = paso1_minusculas(texto)
    texto = paso2_limpiar_caracteres(texto)
    tokens = paso3_tokenizar(texto)
    tokens = paso4_eliminar_stopwords(tokens)
    tokens = paso5_stemming(tokens)
 
    return " ".join(tokens)
 
 
# ─────────────────────────────────────────────
# 4. APLICAR PIPELINE AL DATASET
# ─────────────────────────────────────────────
 
def aplicar_pipeline_dataset(df):
    """
    Aplica el pipeline de limpieza a todas las reseñas del dataset.
 
    Args:
        df (pd.DataFrame): Dataset con columna 'review_text'
 
    Returns:
        pd.DataFrame: Dataset con nueva columna 'review_nltk'
    """
    print(f"[HardwareGuard] Aplicando pipeline NLTK a {len(df)} reseñas...")
    df = df.copy()
    df["review_nltk"] = df["review_text"].apply(pipeline_completo)
    print(f"[HardwareGuard] Pipeline completado ✅")
    return df
 
 
# ─────────────────────────────────────────────
# 5. DEMOSTRACIÓN VISUAL DEL PIPELINE
# ─────────────────────────────────────────────
 
def demostrar_pipeline():
    """
    Muestra ejemplos paso a paso de cómo funciona el pipeline.
    Útil para entender qué hace cada etapa.
    """
    ejemplos = [
        "¡¡El producto llegó DEFECTUOSO de fábrica!! No enciende.",
        "Excelente producto, funciona PERFECTAMENTE desde el primer día.",
        "Pésima calidad, se descompuso al primer MES de uso. TERRIBLE!!",
    ]
 
    sep = "-" * 60
    print(f"\n{'='*60}")
    print("  DEMOSTRACIÓN DEL PIPELINE NLTK — PASO A PASO")
    print(f"{'='*60}\n")
 
    for i, texto in enumerate(ejemplos, 1):
        print(f"  Ejemplo #{i}")
        print(f"  {sep}")
 
        t1 = paso1_minusculas(texto)
        t2 = paso2_limpiar_caracteres(t1)
        t3 = paso3_tokenizar(t2)
        t4 = paso4_eliminar_stopwords(t3)
        t5 = paso5_stemming(t4)
 
        print(f"  Original   : {texto}")
        print(f"  Paso 1     : {t1}")
        print(f"  Paso 2     : {t2}")
        print(f"  Paso 3     : {t3}")
        print(f"  Paso 4     : {t4}")
        print(f"  Paso 5     : {t5}")
        print(f"  RESULTADO  : {' '.join(t5)}")
        print()
 
 
# ─────────────────────────────────────────────
# 6. EJECUCIÓN PRINCIPAL
# ─────────────────────────────────────────────
 
if __name__ == "__main__":
    # Descargar recursos NLTK
    descargar_recursos_nltk()
 
    # Mostrar demostración del pipeline
    demostrar_pipeline()
 
    # Cargar dataset
    base_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
    input_path = os.path.join(base_dir, "hardware_reviews_clean.csv")
 
    print(f"[HardwareGuard] Cargando dataset desde: {input_path}")
    df = pd.read_csv(input_path, encoding="utf-8")
    print(f"[HardwareGuard] Dataset cargado: {len(df)} reseñas")
 
    # Aplicar pipeline
    df = aplicar_pipeline_dataset(df)
 
    # Guardar resultado
    output_path = os.path.join(base_dir, "hardware_reviews_nltk.csv")
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[HardwareGuard] Dataset guardado en: {output_path}")
 
    # Mostrar comparación antes/después
    print(f"\n{'='*60}")
    print("  COMPARACIÓN ANTES / DESPUÉS DEL PIPELINE")
    print(f"{'='*60}")
    for _, row in df.head(4).iterrows():
        print(f"\n  Original : {row['review_text']}")
        print(f"  NLTK     : {row['review_nltk']}")
    print(f"\n[HardwareGuard] PR #2 completado exitosamente ✅")