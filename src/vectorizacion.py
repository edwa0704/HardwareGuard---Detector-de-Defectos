import pandas as pd
import numpy as np
import os
import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import save_npz
 
# ─────────────────────────────────────────────
# 1. CARGAR DATASET PROCESADO POR NLTK
# ─────────────────────────────────────────────
 
def cargar_dataset_nltk():
    """
    Carga el dataset con texto procesado por el pipeline NLTK (PR #2).
 
    Returns:
        pd.DataFrame: Dataset con columna 'review_nltk' lista para vectorizar
    
    Raises:
        FileNotFoundError: Si no existe el dataset NLTK procesado
    """
    base_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
    path = os.path.join(base_dir, "hardware_reviews_nltk.csv")
 
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No se encontró: {path}\n"
            "Ejecuta primero: python src/nlp_pipeline.py"
        )
 
    df = pd.read_csv(path, encoding="utf-8")
    
    # Limpiar filas con texto vacío tras el pipeline NLTK
    df = df.dropna(subset=["review_nltk"])
    df = df[df["review_nltk"].str.strip() != ""]
    
    print(f"[HardwareGuard] Dataset cargado: {len(df)} reseñas listas para vectorizar")
    return df
 
 
# ─────────────────────────────────────────────
# 2. MÉTODO 1 — BAG OF WORDS (BOW)
# ─────────────────────────────────────────────
 
def vectorizar_bow(textos, max_features=5000):
    """
    Método 1: Bag of Words — Bolsa de Palabras.
    
    Crea una matriz de conteo donde cada celda indica
    cuántas veces aparece una palabra en una reseña.
 
    Ejemplo con 3 reseñas y vocabulario ["defecto", "fabric", "product"]:
        reseña 1: "defecto fabric"    → [1, 1, 0]
        reseña 2: "product defecto"   → [1, 0, 1]
        reseña 3: "product"           → [0, 0, 1]
    
    Matriz resultante (3 reseñas × 3 palabras):
        [[1, 1, 0],
         [1, 0, 1],
         [0, 0, 1]]
 
    Args:
        textos (list): Lista de textos procesados por NLTK
        max_features (int): Máximo de palabras en el vocabulario. Default: 5000
 
    Returns:
        tuple: (matriz_bow, vectorizador_bow)
            - matriz_bow      : Matriz dispersa (sparse) de conteos
            - vectorizador_bow: Objeto CountVectorizer entrenado
    """
    vectorizador = CountVectorizer(
        max_features=max_features,  # Usar solo las N palabras más frecuentes
        min_df=2,                   # Ignorar palabras que aparecen en < 2 reseñas
        ngram_range=(1, 2),         # Incluir bigramas: "defect fabric" como una unidad
    )
 
    matriz = vectorizador.fit_transform(textos)
    return matriz, vectorizador
 
 
# ─────────────────────────────────────────────
# 3. MÉTODO 2 — TF-IDF (PRINCIPAL)
# ─────────────────────────────────────────────
 
def vectorizar_tfidf(textos, max_features=5000):
    """
    Método 2: TF-IDF — Term Frequency - Inverse Document Frequency.
    
    A diferencia de BOW, TF-IDF penaliza palabras que aparecen
    en MUCHOS documentos (poco informativas) y premia palabras
    raras pero relevantes.
 
    Fórmula:
        TF(t,d)    = (veces que t aparece en d) / (total palabras en d)
        IDF(t)     = log(N / df(t))   [N=total docs, df=docs con t]
        TF-IDF     = TF × IDF
 
    Ejemplo:
        La palabra "product" aparece en TODAS las reseñas
        → IDF bajo → TF-IDF bajo → poca importancia
 
        La palabra "defectu" aparece solo en reseñas negativas
        → IDF alto → TF-IDF alto → muy importante para el modelo
 
    Args:
        textos (list): Lista de textos procesados por NLTK
        max_features (int): Máximo de palabras en el vocabulario. Default: 5000
 
    Returns:
        tuple: (matriz_tfidf, vectorizador_tfidf)
            - matriz_tfidf      : Matriz dispersa con valores TF-IDF
            - vectorizador_tfidf: Objeto TfidfVectorizer entrenado
    """
    vectorizador = TfidfVectorizer(
        max_features=max_features,  # Vocabulario máximo
        min_df=2,                   # Ignorar términos en < 2 documentos
        max_df=0.95,                # Ignorar términos en > 95% de documentos
        ngram_range=(1, 2),         # Unigramas y bigramas
        sublinear_tf=True,          # Aplicar log al TF para suavizar frecuencias altas
    )
 
    matriz = vectorizador.fit_transform(textos)
    return matriz, vectorizador
 
 
# ─────────────────────────────────────────────
# 4. ANÁLISIS DE DIMENSIONES
# ─────────────────────────────────────────────
 
def analizar_dimensiones(matriz_bow, matriz_tfidf, vec_bow, vec_tfidf):
    """
    Demuestra cómo las dimensiones de la matriz crecen
    según el tamaño del vocabulario global.
 
    Args:
        matriz_bow   : Matriz BOW generada
        matriz_tfidf : Matriz TF-IDF generada
        vec_bow      : Vectorizador BOW entrenado
        vec_tfidf    : Vectorizador TF-IDF entrenado
    """
    sep = "=" * 65
    print(f"\n{sep}")
    print("  📐 ANÁLISIS DE DIMENSIONES DE LA MATRIZ")
    print(f"{sep}\n")
 
    # BOW
    n_docs_bow, n_vocab_bow = matriz_bow.shape
    densidad_bow = matriz_bow.nnz / (n_docs_bow * n_vocab_bow) * 100
 
    print("  📦 MÉTODO 1: Bag of Words (BOW)")
    print(f"     Documentos (filas)    : {n_docs_bow:,}")
    print(f"     Vocabulario (columnas): {n_vocab_bow:,}")
    print(f"     Dimensiones matriz    : {n_docs_bow:,} × {n_vocab_bow:,}")
    print(f"     Total celdas          : {n_docs_bow * n_vocab_bow:,}")
    print(f"     Celdas con valor > 0  : {matriz_bow.nnz:,}")
    print(f"     Densidad              : {densidad_bow:.4f}%")
    print(f"     → Matriz MUY dispersa (sparse): el {100-densidad_bow:.1f}% son ceros")
 
    print()
 
    # TF-IDF
    n_docs_tfidf, n_vocab_tfidf = matriz_tfidf.shape
    densidad_tfidf = matriz_tfidf.nnz / (n_docs_tfidf * n_vocab_tfidf) * 100
 
    print("  🎯 MÉTODO 2: TF-IDF (Principal)")
    print(f"     Documentos (filas)    : {n_docs_tfidf:,}")
    print(f"     Vocabulario (columnas): {n_vocab_tfidf:,}")
    print(f"     Dimensiones matriz    : {n_docs_tfidf:,} × {n_vocab_tfidf:,}")
    print(f"     Total celdas          : {n_docs_tfidf * n_vocab_tfidf:,}")
    print(f"     Celdas con valor > 0  : {matriz_tfidf.nnz:,}")
    print(f"     Densidad              : {densidad_tfidf:.4f}%")
    print(f"     → Matriz MUY dispersa (sparse): el {100-densidad_tfidf:.1f}% son ceros")
 
    print(f"\n  💡 CONCLUSIÓN MATEMÁTICA:")
    print(f"     Si el vocabulario crece de {n_vocab_tfidf:,} a 10,000 palabras,")
    print(f"     la matriz pasa de {n_docs_tfidf:,}×{n_vocab_tfidf:,} a {n_docs_tfidf:,}×10,000")
    print(f"     → Las dimensiones crecen LINEALMENTE con el vocabulario")
    print(f"     → Por eso usamos matrices DISPERSAS (sparse) para ahorrar memoria")
    print(f"{sep}\n")
 
 
# ─────────────────────────────────────────────
# 5. TOP PALABRAS MÁS IMPORTANTES
# ─────────────────────────────────────────────
 
def mostrar_top_palabras(vec_tfidf, matriz_tfidf, df, top_n=10):
    """
    Muestra las palabras con mayor peso TF-IDF en reseñas
    positivas vs negativas.
 
    Args:
        vec_tfidf   : Vectorizador TF-IDF entrenado
        matriz_tfidf: Matriz TF-IDF
        df          : DataFrame con columna 'label'
        top_n       : Número de palabras a mostrar. Default: 10
    """
    vocabulario = vec_tfidf.get_feature_names_out()
    
    # Separar reseñas positivas y negativas (convertir a array numpy)
    idx_neg = (df["label"] == 1).values
    idx_pos = (df["label"] == 0).values
 
    # Promedio TF-IDF por clase
    media_neg = np.asarray(matriz_tfidf[idx_neg].mean(axis=0)).flatten()
    media_pos = np.asarray(matriz_tfidf[idx_pos].mean(axis=0)).flatten()
 
    top_neg = pd.Series(media_neg, index=vocabulario).sort_values(ascending=False).head(top_n)
    top_pos = pd.Series(media_pos, index=vocabulario).sort_values(ascending=False).head(top_n)
 
    sep = "=" * 65
    print(f"{sep}")
    print(f"  🔑 TOP {top_n} PALABRAS MÁS IMPORTANTES (TF-IDF)")
    print(f"{sep}\n")
 
    print(f"  🔴 Reseñas NEGATIVAS (defectos):")
    for palabra, peso in top_neg.items():
        barra = "█" * int(peso * 500)
        print(f"     {palabra:<20} {peso:.6f}  {barra}")
 
    print(f"\n  🟢 Reseñas POSITIVAS (buenas):")
    for palabra, peso in top_pos.items():
        barra = "█" * int(peso * 500)
        print(f"     {palabra:<20} {peso:.6f}  {barra}")
 
    print(f"\n{sep}\n")
 
 
# ─────────────────────────────────────────────
# 6. EJECUCIÓN PRINCIPAL
# ─────────────────────────────────────────────
 
if __name__ == "__main__":
    print("[HardwareGuard] Iniciando PR #3: Vectorización TF-IDF...\n")
 
    # Cargar dataset NLTK
    df = cargar_dataset_nltk()
    textos = df["review_nltk"].tolist()
 
    # ── Vectorización BOW ────────────────────
    print("[HardwareGuard] Vectorizando con Bag of Words...")
    matriz_bow, vec_bow = vectorizar_bow(textos, max_features=5000)
    print(f"[HardwareGuard] BOW listo: {matriz_bow.shape[0]:,} reseñas × {matriz_bow.shape[1]:,} palabras")
 
    # ── Vectorización TF-IDF ─────────────────
    print("[HardwareGuard] Vectorizando con TF-IDF...")
    matriz_tfidf, vec_tfidf = vectorizar_tfidf(textos, max_features=5000)
    print(f"[HardwareGuard] TF-IDF listo: {matriz_tfidf.shape[0]:,} reseñas × {matriz_tfidf.shape[1]:,} palabras")
 
    # ── Análisis de dimensiones ──────────────
    analizar_dimensiones(matriz_bow, matriz_tfidf, vec_bow, vec_tfidf)
 
    # ── Top palabras por sentimiento ─────────
    mostrar_top_palabras(vec_tfidf, matriz_tfidf, df)
 
    # ── Guardar matrices ─────────────────────
    base_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
 
    # Guardar matriz TF-IDF en formato sparse (eficiente en memoria)
    ruta_matriz = os.path.join(base_dir, "matriz_tfidf.npz")
    save_npz(ruta_matriz, matriz_tfidf)
    print(f"[HardwareGuard] Matriz TF-IDF guardada en: {ruta_matriz}")
 
    # Guardar etiquetas (labels) para el modelo
    ruta_labels = os.path.join(base_dir, "labels.csv")
    df[["review_id", "label", "is_defective"]].to_csv(ruta_labels, index=False)
    print(f"[HardwareGuard] Labels guardados en: {ruta_labels}")
 
    # Guardar vocabulario como referencia
    ruta_vocab = os.path.join(base_dir, "vocabulario.json")
    vocabulario = {palabra: int(idx) for idx, palabra in enumerate(vec_tfidf.get_feature_names_out())}
    with open(ruta_vocab, "w", encoding="utf-8") as f:
        json.dump(vocabulario, f, ensure_ascii=False, indent=2)
    print(f"[HardwareGuard] Vocabulario guardado en: {ruta_vocab}")
 
    print(f"\n[HardwareGuard] ✅ PR #3 completado — Tensores listos para alimentar el modelo")
    print(f"[HardwareGuard] 📐 Dimensiones finales: {matriz_tfidf.shape[0]:,} × {matriz_tfidf.shape[1]:,}")