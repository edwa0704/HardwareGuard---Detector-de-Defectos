import os
import re
import json
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, hstack, csr_matrix
from flask import Flask, request, jsonify, render_template_string
 
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
 
# Librerías para análisis multiidioma
try:
    from deep_translator import GoogleTranslator
    from langdetect import detect, DetectorFactory
    from textblob import TextBlob
    DetectorFactory.seed = 42  # Reproducibilidad en detección de idioma
    MULTIIDIOMA_DISPONIBLE = True
    print("[HardwareGuard] Modo multiidioma activado (deep_translator + langdetect + textblob)")
except ImportError:
    MULTIIDIOMA_DISPONIBLE = False
    print("[HardwareGuard] Modo basico activado (instala deep-translator langdetect textblob para multiidioma)")
 
# ─────────────────────────────────────────────
# 0. CONFIGURACIÓN
# ─────────────────────────────────────────────
 
BASE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
UMBRAL   = 0.28
 
# ─────────────────────────────────────────────
# 1. PALABRAS CLAVE
# ─────────────────────────────────────────────
 
PALABRAS_DEFECTO = {
    "broken","defective","defect","stopped","failed","failure",
    "malfunction","malfunctioned","broke","dead","died",
    "returned","return","refund","replacement","replaced",
    "disappointed","disappointing","waste","useless","garbage",
    "terrible","horrible","awful","worst","poor","faulty","fault",
    "overheating","overheat","cracked","scratched","damaged",
    "not working","doesnt work","stopped working","wont turn",
    "wont start","dead on arrival","doa",
}
 
PALABRAS_POSITIVAS = {
    "excellent","perfect","amazing","fantastic","wonderful",
    "great","love","loved","awesome","outstanding",
    "recommend","recommended","happy","satisfied","works",
    "working","reliable","durable","quality","best",
    "easy","fast","quick","smooth","solid",
}
 
STOPWORDS = {
    'the','a','an','and','or','but','in','on','at','to','for',
    'of','with','is','was','are','were','be','been','have','has',
    'had','do','does','did','will','would','could','should','this',
    'that','these','those','it','its','i','my','me','we','our',
    'you','your','he','she','they','their','not','no','so','if',
    'as','by','from','about','up','out','very','just','also',
    'more','most','some','any','all','get','got','one','two',
}
 
 
# ─────────────────────────────────────────────
# DICCIONARIO ESPAÑOL → INGLÉS
# ─────────────────────────────────────────────
 
DICCIONARIO_ES_EN = {
    "defectuoso":"defective","defecto":"defect","defectos":"defects",
    "roto":"broken","rota":"broken","dañado":"damaged","dañada":"damaged",
    "falla":"failure","fallas":"failure","falló":"failed","fallando":"failing",
    "fallo":"fault","malo":"bad","mala":"bad","malísimo":"terrible",
    "terrible":"terrible","horrible":"horrible","pésimo":"awful","pésima":"awful",
    "inservible":"useless","inútil":"useless","basura":"garbage",
    "decepcionado":"disappointed","decepcionante":"disappointing",
    "decepción":"disappointment","devuelto":"returned","devolví":"returned",
    "devolver":"return","devolución":"refund","reembolso":"refund",
    "sobrecalienta":"overheating","recalentado":"overheating",
    "rayado":"scratched","rayada":"scratched","muerto":"dead","muerta":"dead",
    "quemado":"burned","quemada":"burned","lento":"slow","lenta":"slow",
    "frágil":"fragile","inestable":"unstable","no funciona":"not working",
    "no enciende":"wont start","no prende":"wont start",
    "dejó de":"stopped","se dañó":"broke","se rompió":"broke",
    "no sirve":"useless","no carga":"not charging","mala calidad":"poor quality",
    "muy malo":"very bad","muy mala":"very bad",
    "lote defectuoso":"defective batch","de fábrica":"factory defect",
    "excelente":"excellent","perfecto":"perfect","perfecta":"perfect",
    "increíble":"amazing","increible":"amazing","fantástico":"fantastic",
    "maravilloso":"wonderful","maravillosa":"wonderful",
    "bueno":"good","buena":"good","muy bueno":"very good","muy buena":"very good",
    "genial":"great","recomiendo":"recommend","recomendable":"recommended",
    "satisfecho":"satisfied","satisfecha":"satisfied",
    "contento":"happy","contenta":"happy","feliz":"happy",
    "funciona":"works","funcionando":"working","funciona bien":"works great",
    "durable":"durable","duradero":"durable","confiable":"reliable",
    "calidad":"quality","rápido":"fast","rapido":"fast","rápida":"fast",
    "fácil":"easy","facil":"easy","vale la pena":"worth it",
    "muy bien":"very good",
}
 
 
def traducir_es_en(texto):
    """
    Traduce palabras y frases clave del español al inglés
    usando el diccionario local sin APIs externas.
 
    Proceso:
        1. Detectar si el texto tiene palabras en español
        2. Reemplazar frases completas primero (más específicas)
        3. Reemplazar palabras individuales
        4. Retornar texto traducido para el modelo
 
    Args:
        texto (str): Reseña original en cualquier idioma
 
    Returns:
        tuple: (texto_traducido, es_español)
    """
    texto_lower = texto.lower()
    indicadores_es = {
        # Artículos y preposiciones
        'el','la','los','las','de','que','en','un','una','es',
        'se','no','al','con','por','su','para','pero','muy',
        # Palabras comunes en reseñas
        'producto','funciona','llegó','compré','está','este',
        'llego','compre','esta','fue','tiene','tenía','tenia',
        # Palabras positivas en español
        'excelente','perfecto','perfecta','increíble','increible',
        'fantástico','fantastico','maravilloso','genial','recomiendo',
        'satisfecho','satisfecha','contento','contenta','feliz',
        'bueno','buena','calidad','rápido','rapido','fácil','facil',
        # Palabras negativas en español
        'defectuoso','defectuosa','roto','rota','dañado','dañada',
        'malo','mala','pésimo','pésima','pesimo','pesima','terrible',
        'horrible','decepcionado','decepcionada','inservible','basura',
        'devuelto','devolví','falla','fallas','falló','fallo',
    }
    tokens = set(texto_lower.split())
    es_español = len(tokens & indicadores_es) >= 1
 
    if not es_español:
        return texto, False
 
    # Limpiar puntuación para que "defectuoso," se detecte como "defectuoso"
    import re as _re
    texto_traducido = _re.sub(r"[^\w\s]", " ", texto_lower)
    texto_traducido = _re.sub(r"\s+", " ", texto_traducido).strip()
 
    # Traducir frases de mayor a menor longitud
    frases = sorted(DICCIONARIO_ES_EN.keys(), key=len, reverse=True)
    for frase in frases:
        if frase in texto_traducido:
            texto_traducido = texto_traducido.replace(frase, DICCIONARIO_ES_EN[frase])
 
    return texto_traducido, True
 
# ─────────────────────────────────────────────
# 2. ARQUITECTURA DEL MODELO
# ─────────────────────────────────────────────
 
class HardwareGuardNet(nn.Module):
    def __init__(self, input_dim):
        super(HardwareGuardNet, self).__init__()
        self.red = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        return self.red(x)
 
 
# ─────────────────────────────────────────────
# 3. CARGAR MODELO Y VECTORIZADOR
# ─────────────────────────────────────────────
 
def cargar_modelo():
    """
    Carga el modelo entrenado desde disco.
    Intenta cargar V2, si no existe carga V1.
 
    Returns:
        tuple: (modelo, input_dim, umbral)
    """
    # Intentar cargar V2 primero
    for version in ["hardwareguard_model_v2.pth", "hardwareguard_model.pth"]:
        ruta = os.path.join(DATA_DIR, version)
        if os.path.exists(ruta):
            checkpoint = torch.load(ruta, map_location="cpu", weights_only=False)
            input_dim  = checkpoint["input_dim"]
            umbral     = checkpoint.get("umbral", UMBRAL)
 
            modelo = HardwareGuardNet(input_dim=input_dim)
 
            # Cargar pesos — ignorar capas incompatibles si es V2 con BatchNorm
            try:
                modelo.load_state_dict(checkpoint["model_state_dict"])
            except RuntimeError:
                # V2 tiene BatchNorm, cargar arquitectura compatible
                class HardwareGuardNetV2(nn.Module):
                    def __init__(self, input_dim):
                        super().__init__()
                        self.red = nn.Sequential(
                            nn.Linear(input_dim, 256),
                            nn.BatchNorm1d(256),
                            nn.ReLU(),
                            nn.Dropout(0.4),
                            nn.Linear(256, 128),
                            nn.BatchNorm1d(128),
                            nn.ReLU(),
                            nn.Dropout(0.4),
                            nn.Linear(128, 64),
                            nn.ReLU(),
                            nn.Linear(64, 1),
                            nn.Sigmoid()
                        )
                    def forward(self, x):
                        return self.red(x)
 
                modelo = HardwareGuardNetV2(input_dim=input_dim)
                modelo.load_state_dict(checkpoint["model_state_dict"])
 
            modelo.eval()
            print(f"[HardwareGuard] Modelo cargado: {version}")
            print(f"  Input dim : {input_dim}")
            print(f"  Umbral    : {umbral}")
            return modelo, input_dim, umbral
 
    raise FileNotFoundError(
        "No se encontró ningún modelo entrenado.\n"
        "Ejecuta primero: python src/entrenar_modelo.py"
    )
 
 
def cargar_vectorizador():
    """
    Reconstruye el vectorizador TF-IDF entrenando con el dataset original.
    Necesario para transformar texto nuevo con el mismo vocabulario.
 
    Returns:
        TfidfVectorizer: Vectorizador entrenado
    """
    ruta_clean = os.path.join(DATA_DIR, "reviews_clean.csv")
    if not os.path.exists(ruta_clean):
        raise FileNotFoundError(
            f"No se encontró: {ruta_clean}\n"
            "Ejecuta primero: python src/procesar_kaggle.py"
        )
 
    print("[HardwareGuard] Reconstruyendo vectorizador TF-IDF...")
    df = pd.read_csv(ruta_clean, low_memory=False)
 
    def limpiar(texto):
        if not isinstance(texto, str):
            return ""
        texto = texto.lower()
        texto = re.sub(r"[^a-záéíóúüñ\s]", " ", texto)
        texto = re.sub(r"\s+", " ", texto).strip()
        tokens = [t for t in texto.split() if t not in STOPWORDS and len(t) > 2]
        return " ".join(tokens)
 
    textos = df["review_text"].apply(limpiar).tolist()
 
    vec = TfidfVectorizer(
        max_features=8000, min_df=3, max_df=0.90,
        ngram_range=(1, 2), sublinear_tf=True
    )
    vec.fit(textos)
    print(f"[HardwareGuard] Vectorizador listo — vocabulario: {len(vec.vocabulary_):,} términos")
    return vec, limpiar
 
 
# ─────────────────────────────────────────────
# 4. FUNCIÓN DE PREDICCIÓN
# ─────────────────────────────────────────────
 
def detectar_idioma(texto):
    """
    Detecta el idioma de un texto usando langdetect.
    Si no está disponible retorna 'en' por defecto.
 
    Args:
        texto (str): Texto a analizar
 
    Returns:
        str: Código de idioma (ej: 'es', 'en', 'zh', 'pt')
    """
    if not MULTIIDIOMA_DISPONIBLE:
        return "en"
    try:
        return detect(texto)
    except Exception:
        return "en"
 
 
def traducir_a_ingles(texto, idioma_origen):
    """
    Traduce cualquier idioma al inglés usando GoogleTranslator.
    Si el texto ya es inglés o la traducción falla, retorna el original.
 
    Args:
        texto         : Texto a traducir
        idioma_origen : Código de idioma detectado
 
    Returns:
        str: Texto en inglés
    """
    if idioma_origen == "en" or not MULTIIDIOMA_DISPONIBLE:
        return texto
    try:
        traducido = GoogleTranslator(source="auto", target="en").translate(texto)
        return traducido if traducido else texto
    except Exception:
        return texto
 
 
def analizar_sentimiento_textblob(texto_en_ingles):
    """
    Analiza el sentimiento de un texto en inglés usando TextBlob.
 
    TextBlob devuelve:
        polarity  : -1.0 (muy negativo) → +1.0 (muy positivo)
        subjectivity: 0.0 (objetivo) → 1.0 (subjetivo)
 
    Ejemplos:
        "terrible product, completely broken"  → polarity: -0.85
        "excellent quality, works perfectly"   → polarity: +0.75
        "the product arrived"                  → polarity: 0.0
 
    Args:
        texto_en_ingles (str): Texto ya traducido al inglés
 
    Returns:
        tuple: (polarity, subjectivity)
    """
    if not MULTIIDIOMA_DISPONIBLE:
        return 0.0, 0.0
    try:
        blob = TextBlob(texto_en_ingles)
        return blob.sentiment.polarity, blob.sentiment.subjectivity
    except Exception:
        return 0.0, 0.0
 
 
def predecir(texto, modelo, vectorizador, limpiar_fn, input_dim):
    """
    Predice si una reseña indica defecto de fábrica.
 
    Pipeline completo:
        1. Detectar idioma (langdetect)
        2. Traducir a inglés (deep_translator)
        3. Analizar sentimiento (TextBlob)
        4. Vectorizar con TF-IDF
        5. Predicción con red neuronal PyTorch
        6. Combinar señales para resultado final
 
    Fórmula de combinación:
        prob_final = modelo(40%) + sentimiento_negativo(40%) + defect_score(20%)
 
    Args:
        texto        : Reseña del usuario en cualquier idioma
        modelo       : Red neuronal entrenada
        vectorizador : TF-IDF entrenado
        limpiar_fn   : Función de limpieza de texto
        input_dim    : Dimensión esperada por el modelo
 
    Returns:
        dict: {probabilidad, clasificacion, idioma, sentimiento, nivel_riesgo}
    """
    if not texto or len(texto.strip()) < 5:
        return {"error": "La reseña es muy corta. Escribe al menos 5 palabras."}
 
    # ── PASO 1: Detectar idioma ──────────────────
    idioma_codigo = detectar_idioma(texto)
    NOMBRES_IDIOMA = {
        "es":"Español","en":"Inglés","zh-cn":"Chino","zh-tw":"Chino",
        "pt":"Portugués","fr":"Francés","de":"Alemán","it":"Italiano",
        "ja":"Japonés","ko":"Coreano","ar":"Árabe","ru":"Ruso",
    }
    idioma_nombre = NOMBRES_IDIOMA.get(idioma_codigo, f"Otro ({idioma_codigo})")
 
    # ── PASO 2: Traducir a inglés ────────────────
    texto_en = traducir_a_ingles(texto, idioma_codigo)
    fue_traducido = idioma_codigo != "en"
 
    # ── PASO 3: Análisis de sentimiento TextBlob ─
    polaridad, subjetividad = analizar_sentimiento_textblob(texto_en)
    # Convertir polaridad a señal de defecto:
    # polaridad -1 (muy negativo) → sentimiento_defecto = 1.0
    # polaridad +1 (muy positivo) → sentimiento_defecto = 0.0
    sentimiento_defecto = (1.0 - polaridad) / 2.0  # rango 0-1
 
    # ── PASO 4: Vectorizar con TF-IDF ───────────
    texto_limpio = limpiar_fn(texto_en)
    X_tfidf = vectorizador.transform([texto_limpio])
 
    # ── PASO 5: Features de palabras clave ───────
    tokens = re.sub(r"[^\w\s]", " ", texto_en.lower()).split()
    n = max(len(tokens), 1)
 
    DEFECTO_EXT = PALABRAS_DEFECTO | {
        # Inglés adicional
        "wont","awful","broke","poor","worst","useless","garbage",
        "horrible","terrible","disappointing","disappointed","faulty",
        "damaged","defective","failed","failure","dead","returned","waste",
        "broken","stopped","malfunction","overheating","scratched",
        # Alemán (traducido por GoogleTranslator)
        "kaputt","defekt","beschädigt","enttäuscht","schrecklich",
        "furchtbar","schlecht","versagt","defektes","kaputten",
        # Francés
        "panne","cassé","déçu","horrible","terrible","défectueux",
        "mauvais","raté","brisé","défaillance",
        # Italiano
        "rotto","difettoso","deluso","orribile","terribile","guasto",
        "pessimo","scarso","smesso","fallito",
        # Portugués
        "horrível","defeituoso","decepcionado","quebrado","parou",
        "péssimo","ruim","estragou","defeito","falhou",
        # Español adicional
        "defectuoso","roto","dañado","pésimo","horrible",
        "decepcionado","inservible","basura","falló","avería",
    }
    POSITIVAS_EXT = PALABRAS_POSITIVAS | {
        "excellent","perfect","amazing","satisfied","happy","recommend",
        "great","good","works","working","reliable","fast","easy",
        "awesome","fantastic","wonderful","love","quality","smooth",
    }
 
    defect_score   = sum(1 for t in tokens if t in DEFECTO_EXT) / n
    positive_score = sum(1 for t in tokens if t in POSITIVAS_EXT) / n
    palabras_defecto_encontradas   = list(set(t for t in tokens if t in DEFECTO_EXT))[:5]
    palabras_positivas_encontradas = list(set(t for t in tokens if t in POSITIVAS_EXT))[:5]
 
    # ── PASO 6: Predicción red neuronal ──────────
    if input_dim == X_tfidf.shape[1]:
        X_final = X_tfidf.toarray().astype(np.float32)
    else:
        extra   = np.array([[defect_score, positive_score, 0.5]], dtype=np.float32)
        X_final = np.hstack([X_tfidf.toarray(), extra]).astype(np.float32)
 
    with torch.no_grad():
        prob_modelo = modelo(torch.tensor(X_final)).item()
 
    # ── PASO 7: Detectar palabras de contraste ───
    # "precio bueno PERO calidad pésima" → el "pero" indica que
    # la parte negativa es la que importa
    palabras_contraste_es = {"pero","aunque","sin embargo","lamentablemente",
                              "desafortunadamente","lastima","lastimosamente"}
    palabras_contraste_en = {"but","however","unfortunately","sadly","yet",
                              "despite","although","though","except"}
    # Contraste en otros idiomas
    palabras_contraste_de = {"aber","jedoch","leider","trotzdem","obwohl"}
    palabras_contraste_fr = {"mais","cependant","malheureusement","pourtant","malgré"}
    palabras_contraste_it = {"ma","però","tuttavia","purtroppo","nonostante"}
    palabras_contraste_pt = {"mas","porém","infelizmente","contudo","apesar"}
    texto_lower = texto.lower()
    texto_en_lower = texto_en.lower()
    hay_contraste = (
        any(p in texto_lower for p in palabras_contraste_es) or
        any(p in texto_en_lower for p in palabras_contraste_en) or
        any(p in texto_lower for p in palabras_contraste_de) or
        any(p in texto_lower for p in palabras_contraste_fr) or
        any(p in texto_lower for p in palabras_contraste_it) or
        any(p in texto_lower for p in palabras_contraste_pt)
    )
 
    # Si hay contraste Y palabras de defecto → aumentar señal
    bonus_contraste = 0.15 if (hay_contraste and defect_score > 0) else 0.0
 
    # ── PASO 8: Combinar señales ─────────────────
    # Fórmula: modelo + sentimiento TextBlob + palabras clave + contraste
    prob = (prob_modelo * 0.35 +
            sentimiento_defecto * 0.40 +
            defect_score * 0.15 +
            bonus_contraste)
 
    # Si hay palabras positivas SIN contraste, reducir probabilidad
    if not hay_contraste:
        prob = max(0.0, prob - positive_score * 0.15)
    prob = min(1.0, prob)
 
    clasificacion = "DEFECTO" if prob >= UMBRAL else "SIN_DEFECTO"
 
    # Nivel de riesgo
    if prob >= 0.65:
        nivel = "ALTO"
        color = "#E63946"
    elif prob >= 0.40:
        nivel = "MEDIO"
        color = "#F4A261"
    else:
        nivel = "BAJO"
        color = "#2A9D8F"
 
    # Etiqueta de sentimiento
    if polaridad >= 0.2:
        sentimiento_label = f"Positivo ({polaridad:+.2f})"
    elif polaridad <= -0.2:
        sentimiento_label = f"Negativo ({polaridad:+.2f})"
    else:
        sentimiento_label = f"Neutral ({polaridad:+.2f})"
 
    idioma_display = f"{idioma_nombre} → Inglés" if fue_traducido else "Inglés"
 
    return {
        "probabilidad"    : round(prob * 100, 1),
        "clasificacion"   : clasificacion,
        "idioma"          : idioma_display,
        "sentimiento"     : sentimiento_label,
        "nivel_riesgo"    : nivel,
        "color"           : color,
        "texto_traducido" : texto_en if fue_traducido else "",
        "defect_score"    : round(defect_score * 100, 1),
        "positive_score"  : round(positive_score * 100, 1),
        "palabras_defecto"  : palabras_defecto_encontradas,
        "palabras_positivas": palabras_positivas_encontradas,
    }
 
 
# ─────────────────────────────────────────────
# 5. INTERFAZ HTML
# ─────────────────────────────────────────────
 
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HardwareGuard — Predictor</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: #0f1923;
            color: #e2e8f0;
            min-height: 100vh;
        }
        header {
            background: linear-gradient(135deg, #1D3557 0%, #457B9D 100%);
            padding: 30px 40px;
            text-align: center;
        }
        header h1 { font-size: 2rem; color: white; margin-bottom: 6px; }
        header p  { color: #a8c4e0; font-size: 0.95rem; }
        .container {
            max-width: 860px;
            margin: 40px auto;
            padding: 0 20px;
        }
        .card {
            background: #1a2638;
            border-radius: 14px;
            padding: 32px;
            margin-bottom: 24px;
            border: 1px solid #2d3f55;
        }
        .card h2 {
            font-size: 1.1rem;
            color: #457B9D;
            margin-bottom: 18px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        textarea {
            width: 100%;
            min-height: 130px;
            background: #0f1923;
            border: 2px solid #2d3f55;
            border-radius: 10px;
            padding: 16px;
            color: #e2e8f0;
            font-size: 15px;
            font-family: inherit;
            resize: vertical;
            transition: border-color 0.2s;
        }
        textarea:focus {
            outline: none;
            border-color: #457B9D;
        }
        textarea::placeholder { color: #4a6280; }
        .ejemplos {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 14px;
        }
        .ejemplo-btn {
            background: #0f1923;
            border: 1px solid #2d3f55;
            color: #a8c4e0;
            padding: 7px 14px;
            border-radius: 20px;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.2s;
        }
        .ejemplo-btn:hover { border-color: #457B9D; color: white; }
        .ejemplo-btn.negativo { border-color: #E63946; color: #E63946; }
        .ejemplo-btn.negativo:hover { background: #E63946; color: white; }
        .btn-analizar {
            width: 100%;
            padding: 16px;
            background: linear-gradient(135deg, #E63946, #c1121f);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            margin-top: 16px;
            transition: opacity 0.2s;
            letter-spacing: 1px;
        }
        .btn-analizar:hover { opacity: 0.9; }
        .btn-analizar:disabled { opacity: 0.5; cursor: not-allowed; }
        .resultado { display: none; }
        .resultado-header {
            border-radius: 12px;
            padding: 28px;
            text-align: center;
            margin-bottom: 20px;
        }
        .resultado-header .icono { font-size: 3.5rem; margin-bottom: 10px; }
        .resultado-header .clasificacion {
            font-size: 1.6rem;
            font-weight: 800;
            margin-bottom: 6px;
        }
        .resultado-header .probabilidad {
            font-size: 3rem;
            font-weight: 900;
        }
        .resultado-header .nivel {
            font-size: 0.9rem;
            opacity: 0.8;
            margin-top: 6px;
            letter-spacing: 2px;
            text-transform: uppercase;
        }
        .metricas {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
            margin-bottom: 16px;
        }
        .metrica {
            background: #0f1923;
            border-radius: 10px;
            padding: 16px;
            text-align: center;
        }
        .metrica .valor {
            font-size: 1.6rem;
            font-weight: 800;
            color: #457B9D;
        }
        .metrica .label {
            font-size: 0.78rem;
            color: #4a6280;
            margin-top: 4px;
            text-transform: uppercase;
        }
        .palabras-clave {
            background: #0f1923;
            border-radius: 10px;
            padding: 16px;
            margin-bottom: 14px;
        }
        .palabras-clave h4 {
            font-size: 0.78rem;
            color: #4a6280;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }
        .tags { display: flex; flex-wrap: wrap; gap: 8px; }
        .tag {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }
        .tag-defecto  { background: #E6394620; color: #E63946; border: 1px solid #E6394640; }
        .tag-positivo { background: #2A9D8F20; color: #2A9D8F; border: 1px solid #2A9D8F40; }
        .spinner {
            display: none;
            text-align: center;
            padding: 30px;
            color: #457B9D;
        }
        .historial { }
        .historial-item {
            background: #0f1923;
            border-radius: 10px;
            padding: 14px 18px;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-left: 4px solid #2d3f55;
        }
        .historial-item.defecto { border-left-color: #E63946; }
        .historial-item.ok      { border-left-color: #2A9D8F; }
        .historial-texto { font-size: 13px; color: #a8c4e0; flex: 1; margin-right: 16px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .historial-badge {
            font-size: 11px;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 20px;
            white-space: nowrap;
        }
        .badge-defecto  { background: #E6394620; color: #E63946; }
        .badge-ok       { background: #2A9D8F20; color: #2A9D8F; }
        footer {
            text-align: center;
            padding: 30px;
            color: #2d3f55;
            font-size: 12px;
        }
    </style>
</head>
<body>
 
<header>
    <h1>🛡️ HardwareGuard</h1>
    <p>Detector de Defectos de Fábrica — Análisis de Sentimiento en Tiempo Real</p>
</header>
 
<div class="container">
 
    <!-- Input -->
    <div class="card">
        <h2>Analizar Reseña</h2>
        <textarea id="resena" placeholder="Escribe o pega aquí la reseña del producto en inglés o español...&#10;&#10;Ejemplo: 'The printer stopped working after 2 weeks, very disappointed with the quality.'"></textarea>
 
        <div class="ejemplos">
            <span style="font-size:12px;color:#4a6280;align-self:center;">Ejemplos rápidos:</span>
            <button class="ejemplo-btn negativo" onclick="setEjemplo('The product stopped working after 2 weeks. Completely defective, returned it immediately.')">💀 Defectuoso</button>
            <button class="ejemplo-btn negativo" onclick="setEjemplo('Terrible quality, broke after 3 days. Dead on arrival, total waste of money.')">🔴 Falla total</button>
            <button class="ejemplo-btn" onclick="setEjemplo('Excellent product, works perfectly since day one. Highly recommend it!')">🟢 Excelente</button>
            <button class="ejemplo-btn" onclick="setEjemplo('Amazing quality, fast delivery, very satisfied with this purchase.')">✅ Satisfecho</button>
            <button class="ejemplo-btn" onclick="setEjemplo('El producto llegó defectuoso, no enciende desde la caja. Muy decepcionante.')">🇪🇸 Español</button>
        </div>
 
        <button class="btn-analizar" id="btnAnalizar" onclick="analizar()">
            ANALIZAR RESEÑA
        </button>
    </div>
 
    <!-- Spinner -->
    <div class="spinner" id="spinner">
        <p>Analizando con HardwareGuard...</p>
    </div>
 
    <!-- Resultado -->
    <div class="card resultado" id="resultado">
        <h2>Resultado del Análisis</h2>
 
        <div class="resultado-header" id="resultadoHeader">
            <div class="icono" id="icono"></div>
            <div class="clasificacion" id="clasificacionTexto"></div>
            <div class="probabilidad" id="probabilidadTexto"></div>
            <div class="nivel" id="nivelTexto"></div>
        <div style="margin-top:8px;font-size:11px;opacity:0.6;" id="idiomaTexto"></div>
        <div style="margin-top:4px;font-size:11px;opacity:0.6;" id="sentimientoTexto"></div>
        <div style="margin-top:4px;font-size:11px;opacity:0.7;font-style:italic;" id="traduccionTexto"></div>
        </div>
 
        <div class="metricas">
            <div class="metrica">
                <div class="valor" id="defectScore"></div>
                <div class="label">Señal de Defecto</div>
            </div>
            <div class="metrica">
                <div class="valor" id="positiveScore"></div>
                <div class="label">Señal Positiva</div>
            </div>
        </div>
 
        <div class="palabras-clave" id="palabrasDefecto" style="display:none">
            <h4>Palabras de defecto detectadas</h4>
            <div class="tags" id="tagsDefecto"></div>
        </div>
 
        <div class="palabras-clave" id="palabrasPositivas" style="display:none">
            <h4>Palabras positivas detectadas</h4>
            <div class="tags" id="tagsPositivas"></div>
        </div>
    </div>
 
    <!-- Historial -->
    <div class="card" id="historialCard" style="display:none">
        <h2>Historial de Análisis</h2>
        <div id="historialLista"></div>
    </div>
 
</div>
 
<footer>
    HardwareGuard — Detector de Defectos vía NLP &nbsp;|&nbsp;
    Modelo: Red Neuronal PyTorch &nbsp;|&nbsp;
    Dataset: Amazon Consumer Reviews &nbsp;|&nbsp; Frank 2026
</footer>
 
<script>
    const historial = [];
 
    function setEjemplo(texto) {
        document.getElementById('resena').value = texto;
    }
 
    async function analizar() {
        const texto = document.getElementById('resena').value.trim();
        if (!texto) {
            alert('Por favor escribe una reseña primero.');
            return;
        }
 
        // Mostrar spinner
        document.getElementById('spinner').style.display = 'block';
        document.getElementById('resultado').style.display = 'none';
        document.getElementById('btnAnalizar').disabled = true;
 
        try {
            const resp = await fetch('/predecir', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ texto })
            });
            const data = await resp.json();
 
            if (data.error) {
                alert(data.error);
                return;
            }
 
            mostrarResultado(data, texto);
            agregarHistorial(data, texto);
 
        } catch (e) {
            alert('Error al conectar con el servidor.');
        } finally {
            document.getElementById('spinner').style.display = 'none';
            document.getElementById('btnAnalizar').disabled = false;
        }
    }
 
    function mostrarResultado(data, texto) {
        const esDefecto = data.clasificacion === 'DEFECTO';
        const color     = data.color;
 
        // Header
        const header = document.getElementById('resultadoHeader');
        header.style.background = esDefecto
            ? 'linear-gradient(135deg, #E6394615, #E6394630)'
            : 'linear-gradient(135deg, #2A9D8F15, #2A9D8F30)';
        header.style.border = `1px solid ${color}40`;
 
        document.getElementById('icono').textContent          = esDefecto ? '🚨' : '✅';
        document.getElementById('clasificacionTexto').textContent = esDefecto ? 'ALERTA DE DEFECTO' : 'SIN DEFECTO';
        document.getElementById('clasificacionTexto').style.color = color;
        document.getElementById('probabilidadTexto').textContent  = data.probabilidad + '%';
        document.getElementById('probabilidadTexto').style.color  = color;
        document.getElementById('nivelTexto').textContent         = 'Nivel de riesgo: ' + data.nivel_riesgo;
        document.getElementById('idiomaTexto').textContent        = 'Idioma detectado: ' + (data.idioma || 'Inglés');
        document.getElementById('sentimientoTexto').textContent   = 'Sentimiento TextBlob: ' + (data.sentimiento || '');
        if (data.texto_traducido) {
            document.getElementById('traduccionTexto').textContent = 'Traducción: ' + data.texto_traducido;
        } else {
            document.getElementById('traduccionTexto').textContent = '';
        }
 
        // Métricas
        document.getElementById('defectScore').textContent   = data.defect_score + '%';
        document.getElementById('positiveScore').textContent = data.positive_score + '%';
 
        // Palabras clave defecto
        if (data.palabras_defecto && data.palabras_defecto.length > 0) {
            document.getElementById('palabrasDefecto').style.display = 'block';
            document.getElementById('tagsDefecto').innerHTML =
                data.palabras_defecto.map(p =>
                    `<span class="tag tag-defecto">${p}</span>`
                ).join('');
        } else {
            document.getElementById('palabrasDefecto').style.display = 'none';
        }
 
        // Palabras positivas
        if (data.palabras_positivas && data.palabras_positivas.length > 0) {
            document.getElementById('palabrasPositivas').style.display = 'block';
            document.getElementById('tagsPositivas').innerHTML =
                data.palabras_positivas.map(p =>
                    `<span class="tag tag-positivo">${p}</span>`
                ).join('');
        } else {
            document.getElementById('palabrasPositivas').style.display = 'none';
        }
 
        document.getElementById('resultado').style.display = 'block';
    }
 
    function agregarHistorial(data, texto) {
        historial.unshift({ data, texto });
        if (historial.length > 8) historial.pop();
 
        const lista = document.getElementById('historialLista');
        lista.innerHTML = historial.map(h => {
            const esDefecto = h.data.clasificacion === 'DEFECTO';
            return `
                <div class="historial-item ${esDefecto ? 'defecto' : 'ok'}">
                    <span class="historial-texto">${h.texto}</span>
                    <span class="historial-badge ${esDefecto ? 'badge-defecto' : 'badge-ok'}">
                        ${esDefecto ? '🚨 ' + h.data.probabilidad + '%' : '✅ ' + h.data.probabilidad + '%'}
                    </span>
                </div>
            `;
        }).join('');
 
        document.getElementById('historialCard').style.display = 'block';
    }
 
    // Analizar con Enter + Ctrl
    document.addEventListener('keydown', e => {
        if (e.ctrlKey && e.key === 'Enter') analizar();
    });
</script>
 
</body>
</html>"""
 
 
# ─────────────────────────────────────────────
# 6. SERVIDOR FLASK
# ─────────────────────────────────────────────
 
app = Flask(__name__)
 
# Cargar modelo y vectorizador al iniciar
print("[HardwareGuard] Iniciando servidor...")
MODELO, INPUT_DIM, UMBRAL_MODELO = cargar_modelo()
VECTORIZADOR, LIMPIAR_FN = cargar_vectorizador()
print(f"[HardwareGuard] Servidor listo en http://localhost:5000\n")
 
 
@app.route("/")
def index():
    """Página principal del predictor."""
    return render_template_string(HTML_TEMPLATE)
 
 
@app.route("/predecir", methods=["POST"])
def predecir_endpoint():
    """
    Endpoint que recibe una reseña y devuelve la predicción.
 
    Request body: { "texto": "reseña aquí" }
    Response    : { "probabilidad": 78.5, "clasificacion": "DEFECTO", ... }
    """
    data  = request.get_json()
    texto = data.get("texto", "").strip()
 
    resultado = predecir(texto, MODELO, VECTORIZADOR, LIMPIAR_FN, INPUT_DIM)
    return jsonify(resultado)
 
 
# ─────────────────────────────────────────────
# 7. EJECUCIÓN PRINCIPAL
# ─────────────────────────────────────────────
 
if __name__ == "__main__":
    import webbrowser
    import threading
 
    def abrir_navegador():
        import time
        time.sleep(1.5)
        webbrowser.open("http://localhost:5000")
 
    threading.Thread(target=abrir_navegador, daemon=True).start()
 
    print("=" * 55)
    print("  HARDWAREGUARD — PREDICTOR WEB")
    print("  http://localhost:5000")
    print("  Presiona Ctrl+C para detener el servidor")
    print("=" * 55)
 
    app.run(debug=False, port=5000, host="0.0.0.0")