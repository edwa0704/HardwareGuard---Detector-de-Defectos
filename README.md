🛡️ HardwareGuard — Detector de Defectos de Fábrica

Motor NLP que analiza miles de reseñas reales de Amazon para detectar automáticamente lotes de hardware defectuosos mediante análisis de sentimiento.

📋 Descripción del Proyecto
HardwareGuard es un sistema de Procesamiento de Lenguaje Natural (NLP) que analiza reseñas públicas de productos de hardware para detectar patrones que indican defectos de fábrica.
Impacto empresarial: Previene que el departamento de compras adquiera lotes defectuosos de impresoras, laptops y otros equipos, reduciendo costos de garantía y devoluciones.
Dataset real: 45,033 reseñas de Amazon Consumer Reviews (Kaggle)
Modelo: Red Neuronal PyTorch — Accuracy: 96%

🗂️ Estructura del Proyecto
```bash
HardwareGuard/
├── data/
│   ├── 1429_1.csv                                 # Dataset Kaggle (parte 1)
│   ├── Datafiniti_Amazon_Consumer_Reviews_*.csv   # Dataset Kaggle (parte 2 y 3)
│   ├── reviews_clean.csv                          # Dataset limpio y procesado
│   ├── matriz_tfidf.npz                           # Matriz TF-IDF
│   ├── labels.csv                                 # Etiquetas del modelo
│   ├── vocabulario.json                           # Vocabulario del modelo
│   └── hardwareguard_model.pth                    # Modelo entrenado (PyTorch)
├── src/
│   ├── generate_dataset.py     # Paso 1A: Genera dataset sintético
│   ├── preprocess.py           # Paso 1B: Limpieza del dataset
│   ├── analisis_varianza.py    # Paso 2 PR#1: Varianza por marca
│   ├── nlp_pipeline.py         # Paso 2 PR#2: Pipeline NLTK
│   ├── vectorizacion.py        # Paso 2 PR#3: Vectorización TF-IDF
│   ├── procesar_kaggle.py      # Paso 3: Procesa dataset real Kaggle
│   ├── entrenar_modelo.py      # Paso 4 PR#4: Entrena red neuronal
│   └── reporte_visual.py       # Paso 5 PR#5: Reporte HTML visual
├── reports/
│   └── reporte_hardwareguard.html
├── requirements.txt
├── .gitignore
└── README.md
```
⚙️ Requisitos del Sistema

Python 3.8 o superior
pip (gestor de paquetes de Python)
Conexión a internet (para descargar dataset de Kaggle)
Cuenta gratuita en kaggle.com

💻 Terminales compatibles
| Terminal | Sistema | Comandos del proyecto |
| CMD | Windows | python src/script.py |
| PowerShell | Windows | python src/script.py |
| Git Bash | Windows | python src/script.py |
| cmder | Windows | python src/script.py |
| Terminal | Linux / Mac | python3 src/script.py |

>✅ En Windows todos los terminales aceptan '/' en las rutas. No necesitas usar '\'.
>⚠️ En Linux y Mac usa 'python3' en lugar de 'python'.

🚀 Instalación Paso a Paso

1. Clonar el repositorio
```bash
# Windows (CMD / PowerShell / cmder)
git clone https://github.com/edwa0704/HardwareGuard---Detector-de-Defectos.git
cd HardwareGuard---Detector-de-Defectos

# Linux / Mac
git clone https://github.com/edwa0704/HardwareGuard---Detector-de-Defectos.git
cd HardwareGuard---Detector-de-Defectos
```
2. Crear entorno virtual
```bash
# Windows (CMD / PowerShell / cmder)
python -m venv venv
venv\Scripts\activate

# Linux / Mac
python3 -m venv venv
source venv/bin/activate
```
>✅ Sabrás que está activo porque verás '(venv)' al inicio de tu terminal.

3. Instalar dependencias
```bash
# Windows / Linux / Mac
pip install -r requirements.txt
```
4. Configurar Kaggle API
Paso 1: Regístrate en kaggle.com
Paso 2: Ve a tu perfil → Settings → API → Create Legacy API Key
Paso 3: Coloca el archivo 'kaggle.json' en:
```bash
# Windows — crea la carpeta y copia el archivo:
# C:\Users\TU_USUARIO\.kaggle\kaggle.json

# Linux / Mac
mkdir -p ~/.kaggle
cp ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```
Paso 4: Descarga el dataset
```bash
# Windows / Linux / Mac
kaggle datasets download -d datafiniti/consumer-reviews-of-amazon-products -p data/ --unzip
```
▶️ Ejecución del Proyecto

> ⚠️ Ejecuta los scripts en orden. Cada uno depende del anterior.
```bash
# Windows
python src\generate_dataset.py
python src\preprocess.py

# Linux / Mac
python src/generate_dataset.py
python src/preprocess.py
```
Salida esperada:
```bash
[HardwareGuard] Dataset generado: 5000 reseñas
[HardwareGuard] ✅ Dataset guardado en: 
data/hardware_reviews_raw.csv
```
Paso 2 — Análisis Descriptivo y NLP

PR #1 — Varianza por marca
```bash
# Windows
python src\analisis_varianza.py

# Linux / Mac
python src/analisis_varianza.py
```
Salida esperada:
```bash
#1 Samsung  Varianza: 2.1388  [ALTO]
#2 HyperX   Varianza: 2.0857  [ALTO]
#3 Acer     Varianza: 2.0176  [ALTO]
```
PR #2 — Pipeline NLTK
>Descarga recursos NLTK primero (solo una vez):
```bash
# Windows / Linux / Mac
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"
```

```bash
# Windows
python src\nlp_pipeline.py

# Linux / Mac
python src/nlp_pipeline.py
```
Salida esperada:
```bash
Original : El producto llegó DEFECTUOSO de fábrica
NLTK     : product lleg defectu fabric enciend
```
PR #3 — Vectorización TF-IDF
```bash
# Windows
python src\vectorizacion.py

# Linux / Mac
python src/vectorizacion.py
```
Salida esperada:
```bash
TF-IDF listo: 5,000 reseñas x 231 palabras
```

Paso 3 — Procesar Dataset Real de Kaggle
```bash
# Windows
python src\procesar_kaggle.py

# Linux / Mac
python src/procesar_kaggle.py
```
Salida esperada:
```bash
[HardwareGuard] Total combinado: 67,992 reseñas
[HardwareGuard] Filas finales  : 45,033
[HardwareGuard] Matriz: 45,033 reseñas x 8,000 features
```
Paso 4 — Entrenar el Modelo (PR #4)
```bash
# Windows
python src\entrenar_modelo.py

# Linux / Mac
python src/entrenar_modelo.py
```
Arquitectura de la red:
```bash
Entrada (8000) → Dense(128) → ReLU → Dropout(0.3)
              → Dense(64)  → ReLU → Dropout(0.3)
              → Dense(1)   → Sigmoid
              → Salida: 0 = Sin Defecto | 1 = DEFECTO
```
Salida esperada:
```bash
Epoch 1    Pérdida: 0.1591    Accuracy Test: 96.85%
Accuracy global : 96.00%  [EXCELENTE]
```
Paso 5 — Reporte Visual (PR #5)
```bash
# Windows
python src\reporte_visual.py

# Linux / Mac
python src/reporte_visual.py
```
Se abre automáticamente en el navegador con:

Palabras más frecuentes en reseñas de 1 estrella
Distribución de calificaciones
Top 10 marcas con mayor tasa de defectos
Matriz de confusión visual

📊 Resultados del Modelo
| Métrica | Valor |
|---|---|
| Dataset | 45,033 reseñas reales de Amazon |
| División | 80% entrenamiento / 20% prueba |
| Accuracy | **96.00%** |
| F1-Score (defectos) | 0.5055 |
| Verdaderos Positivos | 184 defectos detectados |
| Falsos Negativos | 170 defectos no detectados |

>Nota: El F1-Score bajo (0.50) es esperado por el desbalance natural del dataset (3.9% defectos vs 96.1% positivos). Esto refleja un escenario real de mercado.

📅 Historial de Pull Requests
| PR | Descripción | Estado |
|---|---|---|
| PR #1 | Análisis de varianza por marca — Top 5 marcas con mayor riesgo | ✅ |
| PR #2 | Pipeline NLTK — tokenización, stopwords y stemming | ✅ |
| PR #3 | Vectorización TF-IDF — matrices numéricas para el modelo | ✅ |
| PR #4 | Red neuronal PyTorch — entrenamiento y matriz de confusión | ✅ |
| PR #5 | Reporte visual HTML — gráficos para el departamento de compras | ✅ |

🧰 Tecnologías Utilizadas
| Librería | Uso |
|---|---|
| pandas / numpy | Manipulación de datos y estadísticas |
| nltk | Pipeline de limpieza de texto (stopwords, stemming) |
| scikit-learn | Vectorización TF-IDF y métricas |
| torch (PyTorch) | Red neuronal |
| matplotlib / seaborn | Gráficos y visualizaciones |
| scipy | Matrices dispersas (sparse) |
| kaggle | Descarga del dataset real de Amazon |

👨‍💻 Autor
Frank — Proyecto Semana 3

HardwareGuard: Detector de Defectos vía Análisis de Sentimiento Público