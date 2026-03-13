HardwareGuard — Detector de Defectos de Fábrica

Motor NLP para detectar lotes de hardware defectuosos mediante análisis de reseñas públicas.


📋 Descripción del Proyecto
HardwareGuard es un sistema de Procesamiento de Lenguaje Natural (NLP) que analiza miles de reseñas de hardware para detectar automáticamente patrones que indican defectos de fábrica en lotes de productos.
Impacto empresarial: Previene que el departamento de compras adquiera lotes defectuosos de impresoras, laptops y otros equipos, reduciendo costos de garantía y devoluciones.

🗂️ Estructura del Proyecto

```bash 
HardwareGuard/
├── data/
│   ├── hardware_reviews_raw.csv      # Dataset crudo generado (5,000 reseñas)
│   └── hardware_reviews_clean.csv    # Dataset limpio y preprocesado
├── src/
│   ├── generate_dataset.py           # PASO 1A: Generación del dataset
│   └── preprocess.py                 # PASO 1B: Limpieza y preprocesamiento
├── notebooks/                        # Análisis exploratorio (próximos pasos)
├── docs/                             # Documentación adicional
├── requirements.txt                  # Dependencias del proyecto
└── README.md                         # Este archivo
```
⚙️ Requisitos del Sistema

Python 3.8 o superior
pip (gestor de paquetes de Python)

🚀 Instalación Paso a Paso
1. Clonar el repositorio
```bash
git clone https://github.com/edwa0704/HardwareGuard.git
cd HardwareGuard
```
2. Crear entorno virtual (recomendado)
```bash
python -m venv venv

# En Windows:
venv\Scripts\activate

# En Linux/Mac:
source venv/bin/activate
```
3. Instalar dependencias
```bash
pip install -r requirements.txt
```
▶️ Ejecución — Paso 1
Paso 1A: Generar el Dataset
```bash
python src/generate_dataset.py
```
Qué hace:

Genera 5,000 reseñas sintéticas de hardware (laptops, impresoras, tarjetas de video, etc.)
Cada reseña tiene: texto, calificación (1-5), sentimiento y etiqueta de defecto
Guarda el resultado en data/hardware_reviews_raw.csv

Salida esperada:
[HardwareGuard] Generando 5000 reseñas de hardware...
[HardwareGuard] Dataset generado: 5000 reseñas
Distribución de sentimientos:
  positive    2932
  negative    1272
  neutral      796
[HardwareGuard] ✅ Dataset guardado en: data/hardware_reviews_raw.csv

Paso 1B: Limpiar y Preprocesar
```bash
python src/preprocess.py
```
Qué hace:

Carga el dataset crudo
Elimina duplicados y valores nulos
Normaliza el texto (minúsculas, sin caracteres especiales)
Filtra reseñas con menos de 5 palabras
Agrega columna label (1=negativo, 0=no negativo)
Guarda el resultado en data/hardware_reviews_clean.csv

Salida esperada:
[HardwareGuard] RESUMEN DE PREPROCESAMIENTO
  Filas originales  : 5000
  Filas finales     : 5000
  Reseñas negativas : 1272 (25.4%)
  Defectos fábrica  : 848 (17.0%)
[HardwareGuard] ✅ Dataset limpio guardado en: data/hardware_reviews_clean.csv

📊 Descripción del Dataset
Columna   Tipo            Descripción
review_id     string        ID único de la reseña (REV_00001, REV_00002...)
product       string        Nombre del producto (HP Pavilion 15, etc.)
category      string        Categoría del hardware (Laptop, Impresora, etc.)
rating        int           Calificación del 1 al 5
review_text   string        Texto original de la reseña
review_clean  string        Texto limpio y normalizado
word_count    int           Número de palabras en la reseña
sentiment     string        Sentimiento: positive / negative / neutral
label         int           1 = reseña negativa (rating ≤ 2), 0 = no negativa
is_defective  int           1 = indica defecto de fábrica, 0 = no

📅 Plan de Entregas
Día      Paso     Descripción                                        Estado
1        1        Generación y preprocesamiento del dataset✅        Completado
2        2        Análisis exploratorio (EDA) y visualizaciones🔄    Pendiente
3        3        Entrenamiento del modelo NLP (TF-IDF + ML)🔄       Pendiente
4        4        Evaluación y métricas del modelo🔄                 Pendiente
5        5        Sistema de alertas de lotes defectuosos🔄          Pendiente

👨‍💻 Autor
Frank — Proyecto Semana 4
HardwareGuard: Detector de Defectos vía Análisis de Sentimiento Público
