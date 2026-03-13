import pandas as pd
import numpy as np
import re
import os
 
# ─────────────────────────────────────────────
# 1. FUNCIONES DE LIMPIEZA
# ─────────────────────────────────────────────
 
def limpiar_texto(texto):
    """
    Limpia y normaliza el texto de una reseña.
    
    Pasos:
    - Convertir a minúsculas
    - Eliminar caracteres especiales (mantiene letras, números y espacios)
    - Eliminar espacios múltiples
    - Strip de espacios al inicio y fin
    
    Args:
        texto (str): Texto crudo de la reseña
    
    Returns:
        str: Texto limpio y normalizado
    """
    if not isinstance(texto, str):
        return ""
    
    # Convertir a minúsculas
    texto = texto.lower()
    
    # Eliminar caracteres especiales (mantener letras con acento y ñ)
    texto = re.sub(r"[^a-záéíóúüñ0-9\s]", " ", texto)
    
    # Eliminar espacios múltiples
    texto = re.sub(r"\s+", " ", texto)
    
    # Strip
    texto = texto.strip()
    
    return texto
 
 
def contar_palabras(texto):
    """
    Cuenta el número de palabras en un texto.
    
    Args:
        texto (str): Texto de la reseña
    
    Returns:
        int: Número de palabras
    """
    if not isinstance(texto, str) or texto == "":
        return 0
    return len(texto.split())
 
 
def validar_rating(rating):
    """
    Valida que el rating esté entre 1 y 5.
    
    Args:
        rating: Calificación a validar
    
    Returns:
        bool: True si es válido, False si no
    """
    try:
        r = int(rating)
        return 1 <= r <= 5
    except (ValueError, TypeError):
        return False
 
 
# ─────────────────────────────────────────────
# 2. FUNCIÓN PRINCIPAL DE PREPROCESAMIENTO
# ─────────────────────────────────────────────
 
def preprocesar_dataset(input_path, output_path, min_palabras=5):
    """
    Carga, limpia y guarda el dataset preprocesado.
    
    Args:
        input_path  (str): Ruta del CSV crudo
        output_path (str): Ruta donde guardar el CSV limpio
        min_palabras (int): Mínimo de palabras por reseña. Default: 5
    
    Returns:
        pd.DataFrame: Dataset limpio
    """
    print("[HardwareGuard] Iniciando preprocesamiento...")
    print(f"[HardwareGuard] Cargando dataset desde: {input_path}")
    
    # Cargar dataset crudo
    df = pd.read_csv(input_path, encoding="utf-8")
    total_inicial = len(df)
    print(f"[HardwareGuard] Filas cargadas: {total_inicial}")
    
    # ── Paso 1: Eliminar duplicados ──────────────
    df = df.drop_duplicates(subset=["review_id"])
    print(f"[HardwareGuard] Tras eliminar duplicados: {len(df)} filas")
    
    # ── Paso 2: Eliminar nulos ───────────────────
    df = df.dropna(subset=["review_text", "rating", "sentiment"])
    print(f"[HardwareGuard] Tras eliminar nulos: {len(df)} filas")
    
    # ── Paso 3: Validar ratings ──────────────────
    df = df[df["rating"].apply(validar_rating)]
    df["rating"] = df["rating"].astype(int)
    print(f"[HardwareGuard] Tras validar ratings: {len(df)} filas")
    
    # ── Paso 4: Limpiar texto ────────────────────
    df["review_clean"] = df["review_text"].apply(limpiar_texto)
    print(f"[HardwareGuard] Texto limpiado correctamente")
    
    # ── Paso 5: Contar palabras ──────────────────
    df["word_count"] = df["review_clean"].apply(contar_palabras)
    
    # ── Paso 6: Filtrar reseñas muy cortas ───────
    df = df[df["word_count"] >= min_palabras]
    print(f"[HardwareGuard] Tras filtrar reseñas cortas (< {min_palabras} palabras): {len(df)} filas")
    
    # ── Paso 7: Agregar columna binaria ──────────
    # 1 = negativo (rating 1-2), 0 = no negativo (rating 3-5)
    df["label"] = (df["rating"] <= 2).astype(int)
    
    # ── Paso 8: Reordenar columnas ───────────────
    df = df[[
        "review_id", "product", "category", "rating",
        "review_text", "review_clean", "word_count",
        "sentiment", "label", "is_defective"
    ]]
    
    # ── Resumen final ────────────────────────────
    total_final = len(df)
    eliminadas = total_inicial - total_final
    
    print(f"\n{'='*50}")
    print(f"[HardwareGuard] RESUMEN DE PREPROCESAMIENTO")
    print(f"{'='*50}")
    print(f"  Filas originales  : {total_inicial}")
    print(f"  Filas eliminadas  : {eliminadas}")
    print(f"  Filas finales     : {total_final}")
    print(f"  Reseñas negativas : {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
    print(f"  Defectos fábrica  : {df['is_defective'].sum()} ({df['is_defective'].mean()*100:.1f}%)")
    print(f"\nDistribución por categoría:")
    print(df["category"].value_counts().to_string())
    print(f"{'='*50}")
    
    # Guardar dataset limpio
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n[HardwareGuard] ✅ Dataset limpio guardado en: {output_path}")
    
    return df
 
 
# ─────────────────────────────────────────────
# 3. EJECUCIÓN PRINCIPAL
# ─────────────────────────────────────────────
 
if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    base_dir = os.path.normpath(base_dir)
    
    input_path  = os.path.join(base_dir, "hardware_reviews_raw.csv")
    output_path = os.path.join(base_dir, "hardware_reviews_clean.csv")
    
    df_limpio = preprocesar_dataset(input_path, output_path)
    
    print(f"\nPrimeras 3 filas del dataset limpio:")
    print(df_limpio[["review_id", "category", "rating", "review_clean", "label"]].head(3).to_string())