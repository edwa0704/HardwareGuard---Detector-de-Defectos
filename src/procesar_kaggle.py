import pandas as pd
import numpy as np
import re
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
 
# ─────────────────────────────────────────────
# 1. CARGAR Y COMBINAR ARCHIVOS KAGGLE
# ─────────────────────────────────────────────
 
def cargar_datasets_kaggle():
    """
    Carga y combina todos los archivos CSV del dataset de Kaggle.
 
    Returns:
        pd.DataFrame: Dataset combinado con columnas estandarizadas
    """
    base_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
 
    archivos = [f for f in os.listdir(base_dir) if f.endswith(".csv") and
                ("Datafiniti" in f or "1429" in f)]
 
    if not archivos:
        raise FileNotFoundError(
            "No se encontraron archivos de Kaggle en data/\n"
            "Ejecuta: kaggle datasets download -d datafiniti/consumer-reviews-of-amazon-products -p data/ --unzip"
        )
 
    dfs = []
    for archivo in archivos:
        ruta = os.path.join(base_dir, archivo)
        print(f"[HardwareGuard] Cargando: {archivo}")
        df = pd.read_csv(ruta, low_memory=False)
        dfs.append(df)
 
    df = pd.concat(dfs, ignore_index=True)
    print(f"[HardwareGuard] Total combinado: {len(df):,} reseñas")
    return df
 
 
# ─────────────────────────────────────────────
# 2. LIMPIAR Y ESTANDARIZAR
# ─────────────────────────────────────────────
 
def limpiar_dataset(df):
    """
    Limpia y estandariza el dataset real de Kaggle.
 
    Pasos:
        1. Seleccionar columnas relevantes
        2. Eliminar filas con texto o rating nulo
        3. Convertir rating a entero
        4. Eliminar duplicados por texto
        5. Crear columna label (1=defecto, 0=sin defecto)
 
    Args:
        df (pd.DataFrame): Dataset crudo de Kaggle
 
    Returns:
        pd.DataFrame: Dataset limpio y estandarizado
    """
    print(f"[HardwareGuard] Limpiando dataset...")
    total_inicial = len(df)
 
    # Seleccionar columnas útiles
    columnas = ["name", "brand", "reviews.text", "reviews.rating"]
    columnas_presentes = [c for c in columnas if c in df.columns]
    df = df[columnas_presentes].copy()
 
    # Renombrar columnas
    df = df.rename(columns={
        "name"           : "product",
        "brand"          : "brand",
        "reviews.text"   : "review_text",
        "reviews.rating" : "rating",
    })
 
    # Eliminar nulos en columnas críticas
    df = df.dropna(subset=["review_text", "rating"])
 
    # Convertir rating a numérico
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    df["rating"] = df["rating"].astype(int)
 
    # Filtrar ratings válidos (1-5)
    df = df[df["rating"].between(1, 5)]
 
    # Eliminar reseñas muy cortas (menos de 5 palabras)
    df = df[df["review_text"].str.split().str.len() >= 5]
 
    # Eliminar duplicados por texto
    df = df.drop_duplicates(subset=["review_text"])
 
    # Agregar review_id
    df = df.reset_index(drop=True)
    df["review_id"] = df.index.map(lambda i: f"REV_{i+1:05d}")
 
    # Crear label binario: 1=negativo(1-2 estrellas), 0=positivo/neutral(3-5)
    df["label"] = (df["rating"] <= 2).astype(int)
 
    # Limpiar brand y product
    df["brand"]   = df["brand"].fillna("Desconocida").str.strip()
    df["product"] = df["product"].fillna("Desconocido").str.strip()
 
    total_final = len(df)
    print(f"[HardwareGuard] Filas originales : {total_inicial:,}")
    print(f"[HardwareGuard] Filas finales     : {total_final:,}")
    print(f"[HardwareGuard] Eliminadas        : {total_inicial - total_final:,}")
    print(f"\nDistribución de ratings:")
    print(df["rating"].value_counts().sort_index().to_string())
    print(f"\nReseñas negativas (label=1): {df['label'].sum():,} ({df['label'].mean()*100:.1f}%)")
    print(f"Reseñas positivas (label=0): {(df['label']==0).sum():,} ({(df['label']==0).mean()*100:.1f}%)")
 
    return df
 
 
# ─────────────────────────────────────────────
# 3. PIPELINE DE TEXTO
# ─────────────────────────────────────────────
 
STOPWORDS_EN = {
    'the','a','an','and','or','but','in','on','at','to','for','of','with',
    'is','was','are','were','be','been','have','has','had','do','does','did',
    'will','would','could','should','may','might','this','that','these','those',
    'it','its','i','my','me','we','our','you','your','he','she','they','their',
    'not','no','so','if','as','by','from','about','up','out','than','then',
    'very','just','also','more','most','some','any','all','both','each',
    'there','here','when','where','who','which','what','how','can','get','got'
}
 
STOPWORDS_ES = {
    'el','la','los','las','de','del','que','y','en','un','una','es','se',
    'no','al','con','por','su','para','como','pero','más','este','lo','le',
    'me','si','ya','o','muy','fue','ser','te','hay','son','era','entre',
    'cuando','todo','bien','también','así','donde','desde','hasta','sobre'
}
 
STOPWORDS = STOPWORDS_EN | STOPWORDS_ES
 
 
def limpiar_texto(texto):
    """
    Limpia y normaliza el texto de una reseña real.
 
    Pasos:
        1. Convertir a minúsculas
        2. Eliminar URLs
        3. Eliminar caracteres especiales
        4. Eliminar stopwords
        5. Eliminar tokens muy cortos
 
    Args:
        texto (str): Texto crudo de la reseña
 
    Returns:
        str: Texto limpio listo para vectorización
    """
    if not isinstance(texto, str) or texto.strip() == "":
        return ""
 
    # Minúsculas
    texto = texto.lower()
 
    # Eliminar URLs
    texto = re.sub(r'http\S+|www\S+', ' ', texto)
 
    # Eliminar caracteres especiales (conservar letras y espacios)
    texto = re.sub(r"[^a-záéíóúüñ\s]", " ", texto)
 
    # Eliminar espacios múltiples
    texto = re.sub(r"\s+", " ", texto).strip()
 
    # Eliminar stopwords y tokens cortos
    tokens = [t for t in texto.split() if t not in STOPWORDS and len(t) > 2]
 
    return " ".join(tokens)
 
 
# ─────────────────────────────────────────────
# 4. VECTORIZACIÓN TF-IDF
# ─────────────────────────────────────────────
 
def vectorizar(textos, max_features=8000):
    """
    Vectoriza los textos usando TF-IDF.
 
    Args:
        textos (list)     : Lista de textos limpios
        max_features (int): Máximo vocabulario. Default: 8000
 
    Returns:
        tuple: (matriz_tfidf, vectorizador)
    """
    vec = TfidfVectorizer(
        max_features=max_features,
        min_df=3,
        max_df=0.90,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    matriz = vec.fit_transform(textos)
    return matriz, vec
 
 
# ─────────────────────────────────────────────
# 5. EJECUCIÓN PRINCIPAL
# ─────────────────────────────────────────────
 
if __name__ == "__main__":
    base_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
 
    # Cargar datasets Kaggle
    df_raw = cargar_datasets_kaggle()
 
    # Limpiar
    df = limpiar_dataset(df_raw)
 
    # Aplicar pipeline de texto
    print(f"\n[HardwareGuard] Aplicando pipeline de texto...")
    df["review_clean"] = df["review_text"].apply(limpiar_texto)
 
    # Filtrar textos vacíos tras limpieza
    df = df[df["review_clean"].str.strip() != ""]
    print(f"[HardwareGuard] Reseñas tras limpieza: {len(df):,}")
 
    # Guardar dataset limpio
    ruta_clean = os.path.join(base_dir, "reviews_clean.csv")
    df.to_csv(ruta_clean, index=False, encoding="utf-8")
    print(f"[HardwareGuard] Dataset guardado: {ruta_clean}")
 
    # Vectorizar
    print(f"\n[HardwareGuard] Vectorizando con TF-IDF...")
    matriz, vec = vectorizar(df["review_clean"].tolist())
    print(f"[HardwareGuard] Matriz: {matriz.shape[0]:,} reseñas × {matriz.shape[1]:,} features")
 
    # Guardar matriz y labels
    save_npz(os.path.join(base_dir, "matriz_tfidf.npz"), matriz)
    df[["review_id", "label"]].to_csv(os.path.join(base_dir, "labels.csv"), index=False)
 
    vocab = {p: int(i) for i, p in enumerate(vec.get_feature_names_out())}
    with open(os.path.join(base_dir, "vocabulario.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
 
    print(f"\n[HardwareGuard] ✅ Pipeline completo")
    print(f"  Reseñas        : {len(df):,}")
    print(f"  Features TF-IDF: {matriz.shape[1]:,}")
    print(f"  Label=1 (defecto)   : {df['label'].sum():,} ({df['label'].mean()*100:.1f}%)")
    print(f"  Label=0 (sin defecto): {(df['label']==0).sum():,} ({(df['label']==0).mean()*100:.1f}%)")
 
    # Mostrar ejemplos
    print(f"\nEjemplos de reseñas reales procesadas:")
    for _, row in df.sample(4, random_state=1).iterrows():
        print(f"\n  Original : {row['review_text'][:80]}...")
        print(f"  Limpia   : {row['review_clean'][:80]}")
        print(f"  Label    : {'🔴 DEFECTO' if row['label']==1 else '🟢 OK'} (rating={row['rating']})")