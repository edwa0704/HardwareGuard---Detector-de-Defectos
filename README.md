🛡️ HardwareGuard AI — Detector de Defectos de Hardware

>Motor NLP multicapa que analiza reseñas reales en tiempo real para detectar automáticamente lotes de hardware >defectuosos mediante análisis de sentimiento avanzado.

📋 Descripción del Proyecto
HardwareGuard AI es un sistema de Procesamiento de Lenguaje Natural (NLP) que analiza reseñas de productos de hardware para detectar patrones que indican defectos de fábrica. El sistema combina múltiples capas de análisis para lograr alta precisión incluso con reseñas ambiguas, sarcásticas o en diferentes idiomas.
Impacto empresarial: Previene que el departamento de compras adquiera lotes defectuosos de impresoras, laptops y otros equipos, reduciendo costos de garantía y devoluciones.

| Dato | | Valor |
|---|---|
| Dataset | | 5,000 reseñas reales de hardware |
| Modelo  | | Red Neuronal PyTorch |
| Precisión IA | | 94% sobre dataset real |
| Correctos | | 4,698 / 5,000 |
| Positivos detectados | | 3,426 |
| Defectos detectados | | 1,574 |
| Idiomas soportados | | ES 🇵🇪 · EN 🇺🇸 · PT 🇧🇷 · FR 🇫🇷 · IT 🇮🇹 · DE 🇩🇪 |

🧠 Motor de Análisis NLP — 6 Capas
El sistema evalúa cada reseña en cascada hasta obtener un resultado confiable:

>Reseña → [1] Negaciones → [2] Señales Hardware → [3] Diccionario ES nativo
>       → [4] VADER (EN) → [5] Modelo .pth → [6] Fallback DEFECTO

| Capa | | Descripción | |Ejemplo |
| 1️⃣ Negaciones | | Detecta frases negadas antes que las señales | | "no estoy contento" → DEFECTO |
| 2️⃣ Señales hardware | | Palabras técnicas de fallo o éxito | | "se calienta", "funciona perfecto" |
| 3️⃣ Diccionario ES | | Expresiones peruanas y latinoamericanas | | "plata botada", "una joya" |
| 4️⃣ VADER | | Análisis semántico en inglés | | Para texto no cubierto por capas anteriores |
| 5️⃣ Modelo .pth | | Red neuronal entrenada con 5,000 reseñas | | Último recurso inteligente |
| 6️⃣ Fallback | | Si nada es concluyente → DEFECTO | | Mejor alertar que ignorar un fallo |
