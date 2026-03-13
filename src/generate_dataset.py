import pandas as pd
import numpy as np
import random
import os
 
# Semilla para reproducibilidad
random.seed(42)
np.random.seed(42)
 
# ─────────────────────────────────────────────
# 1. DATOS BASE
# ─────────────────────────────────────────────
 
PRODUCTS = {
    "Laptop": [
        "HP Pavilion 15", "Lenovo ThinkPad T14", "ASUS VivoBook 14",
        "Dell Inspiron 15", "Acer Aspire 5"
    ],
    "Impresora": [
        "Epson EcoTank L3250", "HP LaserJet Pro M15w", "Canon PIXMA G3110",
        "Xerox Phaser 7100", "Epson L4260"
    ],
    "Tarjeta de Video": [
        "NVIDIA GTX 1660", "AMD RX 6600", "GIGABYTE RTX 3060",
        "ASUS GTX 1650", "MSI RX 580"
    ],
    "Memoria RAM": [
        "Kingston Fury Beast 16GB", "Corsair Vengeance 8GB",
        "HyperX Impact 16GB", "Crucial Ballistix 32GB", "ADATA XPG 8GB"
    ],
    "Procesador": [
        "Intel Core i5-12400", "AMD Ryzen 5 5600", "Intel Core i7-1355U",
        "AMD Ryzen 7 3700X", "Intel Core i3-13100"
    ],
    "Monitor": [
        "LG 24MK400H", "AOC 24G2", "Samsung T35F 27\"",
        "GIGABYTE G27FC", "BenQ GW2480"
    ],
}
 
# ─────────────────────────────────────────────
# 2. PLANTILLAS DE RESEÑAS
# ─────────────────────────────────────────────
 
POSITIVE_REVIEWS = [
    "Excelente producto, funciona perfectamente desde el primer día.",
    "Muy buena calidad, lo recomiendo ampliamente.",
    "Llegó en perfecto estado y funciona de maravilla.",
    "Superó mis expectativas, muy buen rendimiento.",
    "Producto de alta calidad, entrega rápida y sin problemas.",
    "Funciona perfectamente, muy satisfecho con la compra.",
    "Gran relación calidad-precio, lo recomiendo.",
    "Excelente desempeño, sin ningún problema hasta ahora.",
    "Muy contento con el producto, funciona como se describe.",
    "Calidad premium, vale cada centavo invertido.",
]
 
NEGATIVE_REVIEWS = [
    "El producto llegó defectuoso, no enciende desde la caja.",
    "Pésima calidad, se descompuso al primer mes de uso.",
    "Llegó con la pantalla rayada y el hardware fallando.",
    "No funciona correctamente, tiene muchos errores de fábrica.",
    "Muy decepcionante, el producto falló desde el primer día.",
    "Defecto de fábrica evidente, el ventilador hace ruido extraño.",
    "El lote que recibí tiene problemas graves de fabricación.",
    "Se sobrecalienta constantemente, claramente es un defecto.",
    "La batería no carga correctamente, viene dañada de fábrica.",
    "Producto con fallas evidentes, parece ser un lote defectuoso.",
    "El chip tiene fallas, no reconoce la memoria correctamente.",
    "Terrible experiencia, el producto murió a los pocos días.",
    "Fallo catastrófico desde la primera semana, muy mala calidad.",
    "El cable de alimentación viene defectuoso, genera chispas.",
    "Claramente un problema del lote, varios usuarios reportan lo mismo.",
]
 
NEUTRAL_REVIEWS = [
    "El producto está bien, aunque podría mejorar en algunos aspectos.",
    "Cumple con lo básico, nada extraordinario.",
    "Regular, tiene cosas buenas y malas.",
    "Funciona bien pero el empaque llegó dañado.",
    "Aceptable para el precio, no esperaba más.",
    "Tiene algunas limitaciones pero en general sirve.",
    "Ni muy bueno ni muy malo, hace lo que promete.",
    "Esperaba más por el precio pagado, pero cumple.",
]
 
# ─────────────────────────────────────────────
# 3. GENERACIÓN DEL DATASET
# ─────────────────────────────────────────────
 
def generate_review(product, category):
    """
    Genera una reseña aleatoria con su calificación y etiquetas.
    
    La distribución de sentimientos es:
    - 60% positivo (rating 4-5)
    - 25% negativo (rating 1-2) 
    - 15% neutral (rating 3)
    """
    rand = random.random()
 
    if rand < 0.60:
        # Reseña positiva
        rating = random.choice([4, 5])
        text = random.choice(POSITIVE_REVIEWS)
        sentiment = "positive"
        is_defective = 0
 
    elif rand < 0.85:
        # Reseña negativa
        rating = random.choice([1, 2])
        text = random.choice(NEGATIVE_REVIEWS)
        sentiment = "negative"
        # Si menciona defecto de fábrica, marcar como defectuoso
        defect_keywords = ["defecto", "fábrica", "lote", "fallo", "defectuoso"]
        is_defective = 1 if any(k in text.lower() for k in defect_keywords) else 0
 
    else:
        # Reseña neutral
        rating = 3
        text = random.choice(NEUTRAL_REVIEWS)
        sentiment = "neutral"
        is_defective = 0
 
    return {
        "product": product,
        "category": category,
        "rating": rating,
        "review_text": text,
        "sentiment": sentiment,
        "is_defective": is_defective,
    }
 
 
def generate_dataset(n_reviews=5000):
    """
    Genera el dataset completo con n_reviews reseñas.
    
    Args:
        n_reviews (int): Número de reseñas a generar. Default: 5000
    
    Returns:
        pd.DataFrame: Dataset completo con todas las columnas
    """
    print(f"[HardwareGuard] Generando {n_reviews} reseñas de hardware...")
    
    rows = []
    for i in range(n_reviews):
        # Seleccionar categoría y producto al azar
        category = random.choice(list(PRODUCTS.keys()))
        product = random.choice(PRODUCTS[category])
        
        review = generate_review(product, category)
        review["review_id"] = f"REV_{i+1:05d}"
        rows.append(review)
 
    df = pd.DataFrame(rows)
    
    # Reordenar columnas
    df = df[["review_id", "product", "category", "rating", 
             "review_text", "sentiment", "is_defective"]]
    
    print(f"[HardwareGuard] Dataset generado: {len(df)} reseñas")
    print(f"\nDistribución de sentimientos:")
    print(df["sentiment"].value_counts())
    print(f"\nDistribución de calificaciones:")
    print(df["rating"].value_counts().sort_index())
    print(f"\nProductos defectuosos detectados: {df['is_defective'].sum()}")
    print(f"Porcentaje defectuoso: {df['is_defective'].mean()*100:.1f}%")
    
    return df
 
 
# ─────────────────────────────────────────────
# 4. EJECUCIÓN PRINCIPAL
# ─────────────────────────────────────────────
 
if __name__ == "__main__":
    # Generar dataset
    df = generate_dataset(n_reviews=5000)
    
    # Guardar en carpeta data/
    output_path = os.path.join(os.path.dirname(__file__), "..", "data", "hardware_reviews_raw.csv")
    output_path = os.path.normpath(output_path)
    df.to_csv(output_path, index=False, encoding="utf-8")
    
    print(f"\n[HardwareGuard] ✅ Dataset guardado en: {output_path}")
    print(f"[HardwareGuard] Columnas: {list(df.columns)}")
    print(f"\nPrimeras 3 filas:")
    print(df.head(3).to_string())