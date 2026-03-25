🛡️ HardwareGuard AI — Detector de Defectos de Hardware

>Motor NLP multicapa que analiza reseñas reales en tiempo real para detectar automáticamente lotes de hardware >defectuosos mediante análisis de sentimiento avanzado.

📋 Descripción del Proyecto
HardwareGuard AI es un sistema de Procesamiento de Lenguaje Natural (NLP) que analiza reseñas de productos de hardware para detectar patrones que indican defectos de fábrica. El sistema combina múltiples capas de análisis para lograr alta precisión incluso con reseñas ambiguas, sarcásticas o en diferentes idiomas.
Impacto empresarial: Previene que el departamento de compras adquiera lotes defectuosos de impresoras, laptops y otros equipos, reduciendo costos de garantía y devoluciones.

| Dato | Valor |
|------|-------|
| Dataset | 5,000 reseñas reales de hardware |
| Modelo  | Red Neuronal PyTorch |
| Precisión IA | 94% sobre dataset real |
| Correctos | 4,698 / 5,000 |
| Positivos detectados | 3,426 |
| Defectos detectados | 1,574 |
| Idiomas soportados | ES 🇵🇪 · EN 🇺🇸 · PT 🇧🇷 · FR 🇫🇷 · IT 🇮🇹 · DE 🇩🇪 |

🧠 Motor de Análisis NLP — 6 Capas
El sistema evalúa cada reseña en cascada hasta obtener un resultado confiable:

>Reseña → [1] Negaciones → [2] Señales Hardware → [3] Diccionario ES nativo
>       → [4] VADER (EN) → [5] Modelo .pth → [6] Fallback DEFECTO

| Capa | Descripción | Ejemplo |
|------|-------------|---------|
| 1️⃣ Negaciones | Detecta frases negadas antes que las señales | "no estoy contento" → DEFECTO |
| 2️⃣ Señales hardware | Palabras técnicas de fallo o éxito | "se calienta", "funciona perfecto" |
| 3️⃣ Diccionario ES | Expresiones peruanas y latinoamericanas | "plata botada", "una joya" |
| 4️⃣ VADER | Análisis semántico en inglés | Para texto no cubierto por capas anteriores |
| 5️⃣ Modelo .pth | Red neuronal entrenada con 5,000 reseñas | Último recurso inteligente |
| 6️⃣ Fallback | Si nada es concluyente → DEFECTO | Mejor alertar que ignorar un fallo |

Características adicionales:

Separa contexto del producto vs vendedor/envío automáticamente
Vocabulario que aprende palabras nuevas automáticamente (vocab_aprendido.json)
Solo 2 resultados posibles: POSITIVO ✅ o DEFECTO 🚨

🗂️ Estructura del Proyecto

```bash
HardwareGuard---Detector-de-Defectos/
├── data/
│   ├── hardware_reviews_clean.csv      # Dataset principal (5,000 reseñas)
│   ├── hardware_reviews_nltk.csv       # Dataset con tokens NLTK
│   ├── hardware_reviews_raw.csv        # Dataset original sin procesar
│   ├── hardwareguard_model.pth         # Modelo entrenado (PyTorch)
│   ├── vocab.json                      # Vocabulario TF-IDF del modelo
│   ├── labels.csv                      # Etiquetas del dataset
│   └── matriz_tfidf.npz                # Matriz TF-IDF
├── src/
│   ├── predictor.py          # 🌐 Servidor web principal (Flask)
│   ├── entrenar_modelo.py    # 🧠 Re-entrenamiento del modelo
│   ├── generate_dataset.py   # Genera dataset sintético
│   ├── preprocess.py         # Limpieza del dataset
│   ├── analisis_varianza.py  # Varianza por marca
│   ├── nlp_pipeline.py       # Pipeline NLTK
│   ├── vectorizacion.py      # Vectorización TF-IDF
│   ├── procesar_kaggle.py    # Procesa dataset Kaggle
│   ├── ajuste_umbral.py      # Ajuste de umbrales
│   ├── app.py                # Análisis del dataset Kaggle
│   └── reporte_visual.py     # Reporte HTML visual
├── reports/
│   └── reporte_hardwareguard.html
├── requirements.txt
├── .gitignore
└── README.md
```
⚙️ Requisitos del Sistema

Python 3.8 o superior
pip (gestor de paquetes)
Git

Terminales compatibles

| Terminal | Sistema | Comando Python |
|----------|---------|----------------|
| CMD | Windows | python |
| PowerShell | Windows | python |
| Git Bash | Windows | python |
| cmder | Windows | python |
| Terminal | Linux / Mac | python3 |

>✅ En Windows todos los terminales aceptan `/` en las rutas.
>⚠️ En Linux y Mac usa `python3` en lugar de `python`.

🚀 Instalación Paso a Paso

```bash
git clone https://github.com/edwa0704/HardwareGuard---Detector-de-Defectos.git
cd HardwareGuard---Detector-de-Defectos
```

2. Crear entorno virtual
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / Mac
python3 -m venv venv
source venv/bin/activate
```
>✅ Sabrás que está activo porque verás (venv) al inicio de tu terminal.

3. Instalar dependencias

Opción A — Solo el detector en tiempo real (más rápido, recomendado):
```bash
pip install -r requirements-minimal.txt
```
Opción B — Proyecto completo con dataset y entrenamiento:
```bash
pip install -r requirements.txt
```
4. Descargar recursos NLTK (solo una vez)
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"
```
▶️ Uso del Sistema
Iniciar el detector en tiempo real
```bash
# Windows
python src/predictor.py

# Linux / Mac
python3 src/predictor.py
```
Se abrirá automáticamente el navegador en http://127.0.0.1:5000

Interfaz web — Pantalla principal
1. Selecciona la Categoría del producto (Monitor, Laptop, Procesador, etc.)
2. Selecciona el Producto de la lista o escribe uno personalizado
3. Selecciona el Rating (★1 a ★5)
4. Ingresa el nombre del Analista
5. Escribe o pega la reseña del cliente
6. Presiona ▶ Analizar Producto

>⚠️ Todos los campos son obligatorios antes de analizar.
Botones de reportes

| Botón | Descripción |
|-------|-------------|
| 📊 Gráfico Tiempo Real | Estadísticas de las reseñas analizadas en vivo |
| 📁 Gráfico Dataset | Análisis completo de las 5,000 reseñas del dataset |

Gráfico Tiempo Real incluye:

KPIs: Total, Positivos, Defectos, % Defectuosos
Distribución general (dona)
Reseñas por idioma detectado (barras)
Defectos por categoría (barras)
Predicción vs Rating ★ (barras)
Historial: Hora · Producto · Categoría · Rating · Reseña · Veredicto · Analista

Gráfico Dataset incluye:

Precisión IA: 94% sobre 5,000 reseñas reales
Distribución, defectos por categoría, predicción vs rating
Comparación dataset vs tiempo real
Tabla con muestra de 200 reseñas analizadas

🧠 Re-entrenar el Modelo

```bash
# Windows
python src/entrenar_modelo.py

# Linux / Mac
python3 src/entrenar_modelo.py
```

Salida esperada:

```bash
==================================================
  HardwareGuard — Re-entrenamiento del Modelo
==================================================
✅ Dataset preparado:
   Positivos (dataset real): 2932
   Negativos (dataset real): 1272
   Neutrales etiquetados:    703
   TOTAL para entrenamiento: 4907

 Época       Loss  Acc Train   Acc Test
------------------------------------------
    60     0.0041     100.0%     100.0%

✅ Mejor accuracy en test: 100.0%
💾 Archivos guardados correctamente.
🎉 ¡Entrenamiento completado!
```
📊 Pipeline Completo (requiere requirements.txt)

```bash
# Paso 1 — Generar y limpiar dataset
python src/generate_dataset.py
python src/preprocess.py

# Paso 2 — Análisis descriptivo
python src/analisis_varianza.py
python src/nlp_pipeline.py
python src/vectorizacion.py

# Paso 3 — Entrenar modelo
python src/entrenar_modelo.py

# Paso 4 — Reporte visual
python src/reporte_visual.py
```
📊 Resultados del Modelo

| Métrica | Valor |
|---------|-------|
| Dataset | 5,000 reseñas de hardware |
| División | 80% entrenamiento / 20% prueba |
| Precisión IA | 94% sobre dataset completo |
| Correctos | 4,698 / 5,000 |
| Accuracy entrenamiento | 100% |
| Defectos en dataset | 1,574 (31.5%) |
| Positivos en dataset | 3,426 (68.5%) |

🧰 Tecnologías Utilizadas

| Librería | Uso |
|----------|-----|
| Flask | Servidor web y API REST |
| PyTorch | Red neuronal de clasificación |
| scikit-learn | Vectorización TF-IDF |
| VADER Sentiment | Análisis de sentimiento en inglés |
| TextBlob | Traducción automática de idiomas |
| langdetect | Detección de idioma |
| pandas / numpy | Manipulación de datos |
| nltk | Pipeline de limpieza de texto | 
| Chart.js | Gráficos interactivos en el navegador | 

📅 Historial de Versiones

| Versión | Descripción |
|---------|-------------| 
| v1.0 | Motor NLP 6 capas + modelo .pth + interfaz web |
| v1.1 | Gráfico Dataset (94% precisión) + Gráfico Tiempo Real |
| v1.2 | Formulario completo + 4 gráficos tiempo real + requirements-minimal.txt

👨‍💻 Autor Frank — Proyecto Semana 3
HardwareGuard AI: Detector de Defectos vía Análisis de Sentimiento en Tiempo Real

# 🏢 Grupo Almerco - Evaluación de Desempeño Técnico (Practicantes)

> **Proyecto:** Motor NLP para Detección de Defectos en Hardware y Análisis Web  
> **Nivel de Impacto:** Alto 🚀 (Proyecto con excelente base lógica, pero con oportunidades de mejora en robustez y buenas prácticas de ingeniería).

| Criterio | Resultado | Estado |
| :--- | :---: | :---: |
| **Calificación Final** | **16 / 20** | 🟡 Aprobado con Observaciones |
| **Arquitectura NLP** | Excelente | 🌟 |
| **Modularidad** | Bueno | ✅ |
| **Robustez y Manejo de Errores**| Deficiente | ❌ |

---

## 🎯 1. Resumen Ejecutivo

El proyecto demuestra una comprensión sólida de los ciclos de vida de datos (ETL) y la integración de modelos de Machine Learning en aplicaciones web. La arquitectura propuesta es ambiciosa y está bien estructurada, destacando un enfoque pragmático al combinar reglas de negocio, análisis léxico (VADER) y redes neuronales (PyTorch) como mecanismo de *fallback*.

Sin embargo, para que este proyecto pase de ser un **prototipo funcional** a un **sistema listo para producción**, es imperativo corregir vulnerabilidades críticas relacionadas con la gestión de dependencias, el manejo de rutas y la programación defensiva.

---

## 🟢 2. Análisis de Fortalezas (Lo que se hizo muy bien)

* **Arquitectura Multi-Capa (NLP):** La decisión de utilizar un pipeline escalonado (`Reglas` ➔ `VADER` ➔ `Traducción` ➔ `PyTorch`) demuestra un excelente criterio analítico. Optimiza recursos al no depender exclusivamente del modelo pesado para casos simples.
* **Modularidad y Separación de Responsabilidades (SoC):** La estructura del proyecto (`data/`, `src/`, `reports/`) y la división lógica de los scripts facilitan la lectura y el escalamiento del código.
* **Visualización y Entrega de Valor:** La implementación de una UI web con reportes interactivos (Chart.js) aporta un valor tangible al producto final y demuestra enfoque en el usuario.

---

## 🔴 3. Áreas Críticas de Mejora (Riesgos de Ejecución)

A continuación, los puntos que comprometen la estabilidad del sistema y requieren atención inmediata:

<details>
<summary><b>⚠️ Efectos Secundarios en Tiempo de Importación (Import Time)</b></summary>
<br>

* **El Problema:** En `src/nlp_pipeline.py`, la carga de `stopwords.words("spanish")` ocurre a nivel de módulo. Si el entorno no tiene los recursos descargados, la aplicación colapsará inmediatamente impidiendo que arranque la UI.
* **La Solución:** Encapsular la inicialización en una función (ej. `iniciar_nltk()`) que maneje un bloque `try/except` e incluya `nltk.download(...)` antes de asignar las stopwords.
</details>

<details>
<summary><b>⚠️ Inconsistencia en la Gestión del Estado (Rutas de Archivos)</b></summary>
<br>

* **El Problema:** Desincronización en el guardado. `src/app.py` usa `"historial_resenas.json"` (ruta relativa), mientras `src/predictor.py` usa `os.path.join(BASE_DIR, "historial_resenas.json")`. Esto creará historiales paralelos.
* **La Solución:** Definir una **Única Fuente de Verdad (SSOT)**. Centralizar la constante `HISTORIAL_PATH` usando rutas absolutas con `BASE_DIR` y exportarla para todos los módulos.
</details>

---

## 🛠️ 4. Oportunidades de Refactorización (Deuda Técnica)

* **Ausencia de Programación Defensiva en ETL:** En `src/preprocess.py`, se asume ciegamente la existencia de las columnas `review_id`, `sentiment` e `is_defective`. 
  * *Acción:* Implementar validaciones tempranas (`df.columns`) y lanzar excepciones descriptivas en lugar de fallar abruptamente con un `KeyError`.
* **Validación de Dependencias NLTK:** Se exige `punkt_tab` en la documentación, un recurso no estándar que genera confusión. 
  * *Acción:* Auditar y exigir estrictamente solo paquetes válidos (`punkt`, `stopwords`).
* **Optimización de Expresiones Regulares:** Las listas de señales en `predictor.py` contienen redundancias y duplicados, aumentando el costo computacional innecesariamente.

---

## 📝 5. Plan de Acción Inmediato (Next Steps)

Para considerar este *ticket* como cerrado, por favor abre un **Pull Request** cumpliendo las siguientes tareas:

- [ ] **Refactorizar NLTK:** Mover la carga de dependencias NLTK dentro de funciones seguras (`try/except`).
- [ ] **Unificar Rutas:** Centralizar la constante de la ruta del historial JSON (`HISTORIAL_PATH`) en un archivo de configuración transversal.
- [ ] **Validar Esquemas:** Añadir una capa de validación de columnas al inicio del script `preprocess.py`.
- [ ] **Limpieza de Código:** Eliminar las redundancias en las listas de señales/regex.

> *Nota: Por favor, etiqueta a tu líder técnico en el PR una vez que estas tareas estén completadas para proceder con la revisión de código.*