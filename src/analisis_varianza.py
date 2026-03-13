import pandas as pd
import numpy as np
import os
 
# ─────────────────────────────────────────────
# 1. CARGAR DATASET
# ─────────────────────────────────────────────
 
def cargar_dataset():
    """
    Carga el dataset limpio generado en el Paso 1.
 
    Returns:
        pd.DataFrame: Dataset de reseñas limpio
    """
    base_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    path = os.path.normpath(os.path.join(base_dir, "hardware_reviews_clean.csv"))
 
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No se encontró el dataset en: {path}\n"
            "Ejecuta primero: python src/generate_dataset.py y python src/preprocess.py"
        )
 
    df = pd.read_csv(path, encoding="utf-8")
    print(f"[HardwareGuard] Dataset cargado: {len(df)} reseñas")
    return df
 
 
# ─────────────────────────────────────────────
# 2. EXTRAER MARCA DEL PRODUCTO
# ─────────────────────────────────────────────
 
def extraer_marca(nombre_producto):
    """
    Extrae la marca del nombre del producto.
    La marca es siempre la primera palabra del nombre.
 
    Ejemplos:
        "HP Pavilion 15"          -> "HP"
        "Lenovo ThinkPad T14"     -> "Lenovo"
        "NVIDIA GTX 1660"         -> "NVIDIA"
        "Kingston Fury Beast 16GB"-> "Kingston"
 
    Args:
        nombre_producto (str): Nombre completo del producto
 
    Returns:
        str: Nombre de la marca
    """
    if not isinstance(nombre_producto, str):
        return "Desconocida"
    return nombre_producto.strip().split()[0]
 
 
# ─────────────────────────────────────────────
# 3. CALCULAR VARIANZA POR MARCA
# ─────────────────────────────────────────────
 
def calcular_varianza_por_marca(df):
    """
    Agrupa el dataset por marca y calcula estadísticas de calidad.
 
    Métricas calculadas por marca:
        - total_resenas  : Número total de reseñas
        - promedio       : Promedio de calificaciones (μ)
        - varianza       : Varianza de calificaciones (σ²) — indicador de inestabilidad
        - std            : Desviación estándar (σ)
        - min_rating     : Calificación mínima recibida
        - max_rating     : Calificación máxima recibida
        - pct_defectuoso : Porcentaje de reseñas con defecto de fábrica
 
    Args:
        df (pd.DataFrame): Dataset con columnas 'marca', 'rating', 'is_defective'
 
    Returns:
        pd.DataFrame: Tabla con estadísticas por marca, ordenada por varianza desc
    """
    # Extraer marca de cada producto
    df = df.copy()
    df["marca"] = df["product"].apply(extraer_marca)
 
    # Agrupar por marca y calcular métricas
    stats = df.groupby("marca").agg(
        total_resenas = ("rating", "count"),
        promedio      = ("rating", "mean"),
        varianza      = ("rating", "var"),       # Varianza muestral (ddof=1)
        std           = ("rating", "std"),
        min_rating    = ("rating", "min"),
        max_rating    = ("rating", "max"),
        defectuosos   = ("is_defective", "sum"),
    ).reset_index()
 
    # Calcular porcentaje de defectuosos
    stats["pct_defectuoso"] = (stats["defectuosos"] / stats["total_resenas"] * 100).round(1)
 
    # Calcular índice de riesgo combinado:
    # Combina varianza alta + porcentaje de defectos alto
    stats["indice_riesgo"] = (
        stats["varianza"] * 0.6 +
        stats["pct_defectuoso"] * 0.4
    ).round(4)
 
    # Redondear columnas numéricas
    stats["promedio"]  = stats["promedio"].round(2)
    stats["varianza"]  = stats["varianza"].round(4)
    stats["std"]       = stats["std"].round(4)
 
    # Ordenar por varianza descendente
    stats = stats.sort_values("varianza", ascending=False).reset_index(drop=True)
 
    return stats, df
 
 
# ─────────────────────────────────────────────
# 4. IMPRIMIR REPORTE
# ─────────────────────────────────────────────
 
def imprimir_reporte(stats):
    """
    Imprime el reporte de varianza en consola de forma clara y legible.
 
    Args:
        stats (pd.DataFrame): Tabla de estadísticas por marca
    """
    separador = "=" * 65
 
    print(f"\n{separador}")
    print(f"  🛡️  HARDWAREGUARD — REPORTE DE RIESGO POR MARCA")
    print(f"{separador}")
    print(f"  Análisis: Varianza de Calificaciones (Mayor varianza = Mayor riesgo)")
    print(f"  Total de marcas analizadas: {len(stats)}")
    print(f"{separador}\n")
 
    # ── TOP 5 MARCAS MÁS INESTABLES ──────────────
    print("  🔴 TOP 5 MARCAS CON MAYOR RIESGO (Mayor Varianza)\n")
    top5 = stats.head(5)
 
    for i, row in top5.iterrows():
        nivel = "🔴 ALTO" if row["varianza"] > 2.0 else "🟡 MEDIO" if row["varianza"] > 1.5 else "🟢 BAJO"
        print(f"  #{i+1} {row['marca']}")
        print(f"      Varianza         : {row['varianza']:.4f}  {nivel}")
        print(f"      Promedio         : {row['promedio']:.2f} ⭐")
        print(f"      Desv. Estándar   : {row['std']:.4f}")
        print(f"      Rango            : {row['min_rating']} - {row['max_rating']} estrellas")
        print(f"      Total reseñas    : {row['total_resenas']}")
        print(f"      Defectos fábrica : {row['defectuosos']} ({row['pct_defectuoso']}%)")
        print(f"      Índice de riesgo : {row['indice_riesgo']:.4f}")
        print()
 
    print(separador)
 
    # ── TABLA COMPLETA ───────────────────────────
    print("\n  📊 TABLA COMPLETA DE MARCAS\n")
    print(f"  {'Marca':<12} {'Reseñas':>8} {'Promedio':>9} {'Varianza':>9} {'Defectos%':>10} {'Riesgo':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*9} {'-'*9} {'-'*10} {'-'*8}")
 
    for _, row in stats.iterrows():
        print(
            f"  {row['marca']:<12} "
            f"{row['total_resenas']:>8} "
            f"{row['promedio']:>9.2f} "
            f"{row['varianza']:>9.4f} "
            f"{row['pct_defectuoso']:>9.1f}% "
            f"{row['indice_riesgo']:>8.4f}"
        )
 
    print(f"\n{separador}")
    print(f"  ✅ Reporte generado exitosamente")
    print(f"  📌 Conclusión: Las marcas con varianza > 2.0 requieren")
    print(f"     revisión inmediata del lote antes de su compra.")
    print(f"{separador}\n")
 
 
# ─────────────────────────────────────────────
# 5. EJECUCIÓN PRINCIPAL
# ─────────────────────────────────────────────
 
if __name__ == "__main__":
    # Cargar dataset
    df = cargar_dataset()
 
    # Calcular varianza por marca
    stats, df_con_marca = calcular_varianza_por_marca(df)
 
    # Imprimir reporte
    imprimir_reporte(stats)
 
    # Guardar reporte en CSV
    output_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "data", "reporte_varianza_marcas.csv")
    )
    stats.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[HardwareGuard] 💾 Reporte guardado en: {output_path}")