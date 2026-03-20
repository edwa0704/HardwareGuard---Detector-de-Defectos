import os
import re
import json
import torch
import numpy as np
import webbrowser
import threading
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from langdetect import detect
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
 
# ══════════════════════════════════════════════════════════
# 1. CONFIGURACIÓN DE RUTAS
# ══════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
HISTORIAL_FILE = os.path.join(BASE_DIR, "historial_resenas.json")
sia = SentimentIntensityAnalyzer()
 
# ══════════════════════════════════════════════════════════
# 2. MODELO .PTH
# ══════════════════════════════════════════════════════════
class Modelo(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.red = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128), torch.nn.ReLU(),
            torch.nn.Dropout(0.2), torch.nn.Linear(128, 64),
            torch.nn.ReLU(), torch.nn.Linear(64, 1), torch.nn.Sigmoid()
        )
    def forward(self, x): return self.red(x)
 
def cargar_modelo():
    try:
        ruta = os.path.join(DATA_DIR, "hardwareguard_model.pth")
        if not os.path.exists(ruta): return None
        checkpoint = torch.load(ruta, map_location="cpu")
        model = Modelo(checkpoint["input_dim"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model
    except: return None
 
def cargar_vectorizador():
    try:
        ruta_vocab = os.path.join(DATA_DIR, "vocab.json")
        if not os.path.exists(ruta_vocab): return None
        with open(ruta_vocab, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        vec = TfidfVectorizer(vocabulary=vocab)
        vec.fit(["init"])
        return vec
    except: return None
 
# ══════════════════════════════════════════════════════════
# 3. SEÑALES DE CONTEXTO
#    El sistema usa estas listas para entender QUÉ parte
#    de la reseña habla del PRODUCTO vs. del vendedor/envío
# ══════════════════════════════════════════════════════════
 
# Fragmentos que NO hablan del producto hardware
SEÑALES_NO_PRODUCTO = [
    # vendedor / atención — ES
    "vendedor", "vendedora", "atención", "atencion", "servicio", "soporte",
    "amable", "amabilidad", "amor", "cariño", "regalo", "regaló", "regala",
    "funda", "estuche", "empaque", "embalaje", "packaging",
    # envío / entrega — ES
    "envío", "envio", "entrega", "llegó", "llego", "despacho", "courier",
    "shipping", "delivery", "arrived", "llegada",
    # precio — ES/EN
    "precio", "costo", "coste", "barato", "económico", "oferta", "descuento",
    "price", "cost", "cheap", "discount", "deal",
    # devolución / garantía
    "devolución", "devolucion", "reembolso", "garantía", "garantia",
    "return", "refund", "warranty",
    # PT
    "vendedor", "entrega", "frete", "atendimento", "embalagem", "preço",
    # FR / IT
    "vendeur", "livraison", "prix", "service", "venditore", "spedizione", "prezzo",
]
 
# Señales de FALLO TÉCNICO del hardware
# IMPORTANTE: estas se evalúan sobre el texto YA TRADUCIDO al inglés
SEÑALES_DEFECTO = [
    # temperatura
    "overheat", "overheating", "gets hot", "too hot", "burning", "burns",
    "heats up", "heat up", "running hot",
    # apagado / reinicio / fallo de arranque
    "stopped working", "stop working", "stops working", "stopped working",
    "shuts off", "shuts down", "shut off", "shutdown", "turns off",
    "turned off", "won't turn on", "wont turn on", "does not turn on",
    "doesn't turn on", "restart", "reboots", "keeps restarting",
    "not working", "doesnt work", "doesn't work", "ceased to work",
    "no longer works", "stopped functioning", "failed after",
    "died after", "broke after", "broken after", "broke down",
    # pantalla / imagen
    "screen", "display", "cracked screen", "broken screen", "no display",
    "black screen", "flickering", "flickers", "pixelated", "won't turn on",
    # batería / carga
    "no charge", "won't charge", "doesn't charge", "drains fast",
    "battery dies", "bad battery", "battery dead", "battery drains",
    # velocidad / congelamiento
    "slow", "laggy", "lag", "freezes", "frozen", "freeze", "hangs",
    "unresponsive", "crashes", "keeps crashing",
    # fallos generales EN
    "defect", "defective", "broken", "damaged", "faulty", "fault",
    "malfunction", "malfunctioning", "useless", "unusable", "waste",
    "terrible", "horrible", "awful", "garbage", "trash", "junk",
    "piece of junk", "returned it", "had to return",
    # conectividad
    "no wifi", "no bluetooth", "disconnects", "won't connect",
    "connection issues", "signal",
    # ruido / físico
    "noise", "strange sound", "weird sound", "creak", "rattles",
    # cámara / audio
    "camera", "microphone", "speaker", "won't record",
    # ── señales en español (para texto no traducido o traducción fallida) ──
    "calienta", "caliente", "sobrecalienta", "quema",
    "apaga", "apagarse", "se apaga", "reinicia", "reinicio",
    "pantalla rota", "no enciende", "no prende", "no se ve",
    "no carga", "se descarga",
    "lento", "lenta", "cuelga", "congela", "tarda mucho",
    "defecto", "falla", "fallo", "roto", "dañado", "malogrado",
    "no funciona", "inútil", "inutilizable", "basura",
    "no conecta", "se desconecta",
    # cese de funcionamiento — frases compuestas ES
    "dejó de funcionar", "dejo de funcionar",
    "dejó de encender", "dejo de encender",
    "dejó de cargar", "dejo de cargar",
    "dejó de responder", "dejo de responder",
    "dejó de servir", "dejo de servir",
    "paró de funcionar", "paro de funcionar",
    "ya no funciona", "ya no enciende", "ya no carga",
    "ya no sirve", "ya no responde", "ya no prende",
    "duró poco", "duro poco", "duró nada", "duro nada",
    "se arruinó", "se arruino", "se malogró", "se malogro",
    "se dañó", "se daño", "se rompió", "se rompio",
    "se apagó solo", "se apago solo",
    "se murió", "se murio",
    "a los pocos días", "a la semana se", "al mes se",
    "duró una semana", "duró un mes", "duro una semana", "duro un mes",
    # ── señales en portugués ──
    "parou de funcionar", "parou funcionar", "nao funciona",
    "não funciona", "quebrado", "defeituoso", "travando", "trava",
    "aquece", "superaquece", "bateria ruim", "descarrega rapido",
    # ── FR / IT / DE ──
    "ne fonctionne plus", "en panne", "cassé", "défectueux",
    "non funziona", "rotto", "difettoso", "kaputt", "funktioniert nicht",
]
 
# Señales de BUEN FUNCIONAMIENTO del hardware
SEÑALES_POSITIVO = [
    # rendimiento
    "excelente", "excellent", "perfecto", "perfect", "funciona", "works",
    "funciona bien", "works great", "funciona perfecto", "works perfectly",
    "muy rápido", "super rapido", "veloz", "speedy", "fluido", "smooth",
    "carga rápido", "carga rapido", "responde rápido", "responde rapido",
    "potente", "powerful", "increíble", "increible", "incredible", "amazing",
    # calidad
    "buena calidad", "good quality", "calidad", "quality", "resistente",
    "durable", "sólido", "solido", "solid", "robusto",
    # recomendación / satisfacción
    "recomiendo", "recommend", "satisfecho", "feliz", "happy",
    "contento", "encantado", "love it", "me encanta", "genial", "great",
    "óptimo", "optimo", "optimal",
    # PT / FR / IT / DE
    "ótimo", "perfeito", "funciona bem", "parfait", "ottimo", "funziona",
    "schnell", "gut", "funktioniert",
    # palabras que el sistema anterior fallaba
    "vuela", "joya", "original", "increíble",
]
 
# ══════════════════════════════════════════════════════════
# 4. DICCIONARIO DE SENTIMIENTOS ES/EN NATIVO
#    Cubre expresiones coloquiales, peruanas y latinoamericanas
#    que VADER (entrenado en inglés) no entiende.
#    Score: -1.0 (muy negativo) a +1.0 (muy positivo)
# ══════════════════════════════════════════════════════════
SENTIMIENTOS_ES_EN = {
    # ── Muy negativos (-1.0) ──
    "porquería": -1.0, "porqueria": -1.0, "pésimo": -1.0, "pesimo": -1.0,
    "malísimo": -1.0, "malisimo": -1.0, "horrible": -1.0, "espantoso": -1.0,
    "una basura": -1.0, "es basura": -1.0, "pura basura": -1.0,
    "no sirve para nada": -1.0, "no vale nada": -1.0, "lo peor": -1.0,
    "pésima calidad": -1.0, "pesima calidad": -1.0, "fatal": -1.0,
    "decepcionante": -1.0, "una decepción": -1.0, "una decepcion": -1.0,
    "me arrepiento": -1.0, "me arrepentí": -1.0, "me arrepenti": -1.0,
    "estafa": -1.0, "timo": -1.0, "fraude": -1.0, "engaño": -1.0,
    "me tiene harto": -1.0, "me hartó": -1.0, "me harto": -1.0,
    "qué asco": -1.0, "que asco": -1.0, "da asco": -1.0, "asco total": -1.0,
    "no funciona para nada": -1.0, "inservible": -1.0,
    "dinero tirado": -1.0, "tiré mi dinero": -1.0, "tire mi dinero": -1.0,
    "botado el dinero": -1.0, "plata botada": -1.0, "plata a la basura": -1.0,
    "una porquería": -1.0, "pura porquería": -1.0,
    "terrible": -1.0, "malogramiento": -1.0,
    # EN equivalents
    "total garbage": -1.0, "absolute trash": -1.0, "complete junk": -1.0,
    "worst purchase": -1.0, "waste of money": -1.0, "money wasted": -1.0,
    "regret buying": -1.0, "deeply disappointed": -1.0,
 
    # ── Negativos (-0.7) ──
    "malo": -0.7, "mala": -0.7, "mal": -0.7, "feo": -0.7, "fea": -0.7,
    "no me gustó": -0.7, "no me gusto": -0.7, "no me gusta": -0.7,
    "decepcionado": -0.7, "decepcionada": -0.7, "fallido": -0.7,
    "no recomiendo": -0.7, "no lo recomiendo": -0.7,
    "mala compra": -0.7, "mala adquisición": -0.7,
    "no vale la pena": -0.7, "no vale": -0.7,
    "esperaba más": -0.7, "esperaba mejor": -0.7,
    "defraudado": -0.7, "defraudada": -0.7,
    "no cumple": -0.7, "no cumplió": -0.7, "no cumplio": -0.7,
    "mediocre": -0.7, "flojo": -0.7, "floja": -0.7,
    "not worth it": -0.7, "not recommended": -0.7, "disappointed": -0.7,
    "poor quality": -0.7, "bad quality": -0.7, "subpar": -0.7,
 
    # ── Levemente negativos (-0.4) ──
    "regular": -0.4, "más o menos": -0.4, "mas o menos": -0.4,
    "podría ser mejor": -0.4, "podria ser mejor": -0.4,
    "le falta": -0.4, "le faltan": -0.4, "mejorable": -0.4,
    "no es lo que esperaba": -0.4, "algo lento": -0.4,
    "medio lento": -0.4, "medio malo": -0.4,
    "so so": -0.4, "mediocre": -0.4, "average at best": -0.4,
    "could be better": -0.4, "not great": -0.4,
 
    # ── Muy positivos (+1.0) ──
    "joya": 1.0, "una joya": 1.0, "es una joya": 1.0,
    "excelente": 1.0, "espectacular": 1.0, "increíble": 1.0, "increible": 1.0,
    "lo máximo": 1.0, "lo maximo": 1.0, "de lujo": 1.0, "top": 1.0,
    "lo mejor": 1.0, "el mejor": 1.0, "la mejor": 1.0,
    "perfecto": 1.0, "perfecta": 1.0, "impecable": 1.0,
    "maravilloso": 1.0, "maravillosa": 1.0, "genial": 1.0,
    "súper": 1.0, "super": 1.0, "excelentísimo": 1.0,
    "de primera": 1.0, "primera calidad": 1.0, "top calidad": 1.0,
    "vale cada centavo": 1.0, "vale cada sol": 1.0, "valió la pena": 1.0,
    "valiо la pena": 1.0, "lo recomiendo": 1.0, "100% recomendado": 1.0,
    "cien por ciento": 1.0, "sin defectos": 1.0,
    "outstanding": 1.0, "excellent": 1.0, "absolutely love it": 1.0,
    "best purchase": 1.0, "worth every penny": 1.0, "highly recommend": 1.0,
    "flawless": 1.0, "perfect": 1.0, "exceptional": 1.0,
 
    # ── Positivos (+0.7) ──
    "bueno": 0.7, "buena": 0.7, "bien": 0.7, "funciona bien": 0.7,
    "me gustó": 0.7, "me gusto": 0.7, "me gusta": 0.7,
    "satisfecho": 0.7, "satisfecha": 0.7, "contento": 0.7, "contenta": 0.7,
    "recomendado": 0.7, "recomendable": 0.7,
    "cumple su función": 0.7, "cumple su funcion": 0.7,
    "cumple lo prometido": 0.7, "tal como se describe": 0.7,
    "buen producto": 0.7, "buena calidad": 0.7,
    "vale la pena": 0.7, "buen precio calidad": 0.7,
    "good": 0.7, "great": 0.7, "works well": 0.7, "happy with": 0.7,
    "pleased": 0.7, "satisfied": 0.7, "recommend": 0.7,
 
    # ── Levemente positivos (+0.4) ──
    "no está mal": 0.4, "no esta mal": 0.4, "está bien": 0.4, "esta bien": 0.4,
    "sirve": 0.4, "funciona": 0.4, "cumple": 0.4,
    "aceptable": 0.4, "correcto": 0.4, "correcta": 0.4,
    "decent": 0.4, "okay": 0.4, "ok": 0.4, "fine": 0.4, "works": 0.4,
}
 
def aplicar_negaciones(texto_lower):
    """
    Detecta frases negadas y las convierte en su opuesto semántico.
    Ej: "no estoy contento" → señal negativa
        "no funciona mal" → señal positiva
    Retorna un score de ajuste: -0.7 si hay negación de positivo,
    +0.7 si hay negación de negativo, 0 si no hay negación relevante.
    """
    # Patrones: negación + palabra positiva = negativo
    neg_de_positivo = [
        "no estoy contento", "no estoy satisfecho", "no me gusta",
        "no me gustó", "no me gusto", "no cumple", "no cumplió",
        "no cumplio", "no es bueno", "no es buena", "no funciona bien",
        "no vale la pena", "no lo recomiendo", "no recomiendo",
        "not happy", "not satisfied", "not good", "not worth",
        "not working well", "doesn't meet", "does not meet",
        "no meet expectations", "not as expected",
    ]
    # Patrones: negación + palabra negativa = positivo
    neg_de_negativo = [
        "no está mal", "no esta mal", "no es malo", "no es mala",
        "no tiene fallas", "no tiene defectos", "sin fallas", "sin defectos",
        "no falla", "no se cuelga", "no se calienta", "no da problemas",
        "not bad", "no issues", "no problems", "no complaints",
        "without issues", "without problems",
    ]
    for frase in neg_de_positivo:
        if frase in texto_lower:
            return -0.7
    for frase in neg_de_negativo:
        if frase in texto_lower:
            return 0.7
    return 0.0
 
def score_sentimiento_es(texto):
    """
    Calcula score de sentimiento usando el diccionario ES/EN nativo.
    Prioriza frases largas (más específicas) sobre palabras sueltas.
    Retorna valor entre -1.0 y +1.0, o None si no encuentra nada.
    """
    texto_l = texto.lower()
    # Ordenar por longitud descendente: frases primero, luego palabras
    candidatos = sorted(SENTIMIENTOS_ES_EN.keys(), key=len, reverse=True)
    scores = []
    for frase in candidatos:
        if frase in texto_l:
            scores.append(SENTIMIENTOS_ES_EN[frase])
    if not scores:
        return None
    return sum(scores) / len(scores)  # promedio ponderado
 
# ══════════════════════════════════════════════════════════
# 4. MOTOR DE ANÁLISIS
# ══════════════════════════════════════════════════════════
def contar_señales(texto, lista):
    """Cuenta señales de una lista presentes en el texto."""
    return sum(1 for s in lista if s in texto)
 
def separar_fragmentos_producto(texto_lower):
    """
    Divide la reseña por conectores adversativos y copulativos.
    Clasifica cada fragmento: ¿habla del producto o del vendedor/envío?
    Devuelve los fragmentos que SÍ hablan del producto.
    """
    conectores = [
        r'\bpero\b', r'\bsin embargo\b', r'\bno obstante\b', r'\baunque\b',
        r'\bauque\b', r'\bmas\b', r'\bporém\b', r'\bhowever\b', r'\bbut\b',
        r'\byet\b', r'\bthough\b', r'\bmais\b', r'\bpourtant\b',
        r'\bma\b', r'\baber\b', r'\by\b', r'\band\b', r'\be\b',
    ]
    patron = '|'.join(conectores)
    partes = re.split(patron, texto_lower)
 
    fragmentos_producto = []
    for parte in partes:
        parte = parte.strip()
        if not parte:
            continue
        np_score = contar_señales(parte, SEÑALES_NO_PRODUCTO)
        hw_score = contar_señales(parte, SEÑALES_DEFECTO) + contar_señales(parte, SEÑALES_POSITIVO)
        # Solo ignorar si claramente habla de no-producto y nada de hardware
        if np_score > hw_score and hw_score == 0:
            continue
        fragmentos_producto.append(parte)
 
    # Si filtramos todo, usar el texto completo (nunca dejar vacío)
    return fragmentos_producto if fragmentos_producto else partes
 
def traducir_a_ingles(texto, lang):
    """Traduce a inglés para que VADER analice correctamente."""
    try:
        if lang != "en":
            return str(TextBlob(texto).translate(to="en"))
    except:
        pass
    return texto
 
def detectar_sarcasmo(texto_lower):
    """Reservado para detección futura de ironía."""
    return False
 
 
def predecir(texto, model, vec):
    raw = texto.strip()
    if not raw:
        return "⚠️ Texto vacío", 0.0
 
    texto_lower = raw.lower()
 
    # ── A. Detectar idioma ──────────────────────────────────────
    try:
        lang = detect(raw)
    except:
        lang = "es"
    # Normalizar: si no es inglés, tratar como español para el análisis nativo
    es_ingles = (lang == "en")
    flag = {"es": "🇵🇪", "en": "🇺🇸", "pt": "🇧🇷",
            "fr": "🇫🇷", "it": "🇮🇹", "de": "🇩🇪"}.get(lang, "🌐")
 
    # ── B. Separar fragmentos del producto ──────────────────────
    fragmentos_orig = separar_fragmentos_producto(texto_lower)
    texto_producto  = " ".join(fragmentos_orig)
 
    # ── C. Traducir y preparar versión en inglés ───────────────
    texto_en_completo = traducir_a_ingles(texto_lower, lang).lower()
    fragmentos_en     = separar_fragmentos_producto(texto_en_completo)
    texto_producto_en = " ".join(fragmentos_en)
 
    # ── C1. Negaciones PRIMERO (antes de buscar señales) ────────
    #    "no estoy contento" no debe activar POSITIVO por "contento"
    score_neg = aplicar_negaciones(texto_producto)
    if score_neg <= -0.5:
        res = f"{flag} DEFECTO 🚨"
        aprender_de_resena(texto_lower, texto_en_completo, res)
        return res, 100.0
    if score_neg >= 0.5:
        res = f"{flag} POSITIVO ✅"
        aprender_de_resena(texto_lower, texto_en_completo, res)
        return res, 0.0
 
    # ── C2. Señales directas de hardware (ES + EN) ──────────────
    n_defectos = max(
        contar_señales_con_aprendizaje(texto_producto,    SEÑALES_DEFECTO,  "defecto"),
        contar_señales_con_aprendizaje(texto_producto_en, SEÑALES_DEFECTO,  "defecto"),
    )
    n_positivos = max(
        contar_señales_con_aprendizaje(texto_producto,    SEÑALES_POSITIVO, "positivo"),
        contar_señales_con_aprendizaje(texto_producto_en, SEÑALES_POSITIVO, "positivo"),
    )
 
    # REGLA: cualquier señal de defecto gana sobre elogios
    if n_defectos > 0:
        res = f"{flag} DEFECTO 🚨"
        aprender_de_resena(texto_lower, texto_en_completo, res)
        return res, 100.0
 
    if n_positivos > 0:
        res = f"{flag} POSITIVO ✅"
        aprender_de_resena(texto_lower, texto_en_completo, res)
        return res, 0.0
 
    # ── D. Sin señales directas → análisis por sentimiento ─────
    #    Para español: diccionario ES nativo (entiende coloquial, peruano)
    #    Para inglés:  VADER directo (sin traducción, más preciso)
 
    if not es_ingles:
        # D1. Diccionario ES nativo sobre texto original
        score_es = score_sentimiento_es(texto_producto)
        if score_es is not None:
            score_final = score_es
            if score_final <= -0.3:
                res = f"{flag} DEFECTO 🚨"
                aprender_de_resena(texto_lower, texto_en_completo, res)
                return res, 100.0
            if score_final >= 0.3:
                res = f"{flag} POSITIVO ✅"
                aprender_de_resena(texto_lower, texto_en_completo, res)
                return res, 0.0
 
        # D2. Si el diccionario ES no fue concluyente → VADER sobre traducción
        score_vader = sia.polarity_scores(texto_producto_en)["compound"]
    else:
        # D3. Texto en inglés → VADER directo, sin traducción
        score_vader = sia.polarity_scores(texto_producto)["compound"]
 
    if score_vader <= -0.05:
        res = f"{flag} DEFECTO 🚨"
        aprender_de_resena(texto_lower, texto_en_completo, res)
        return res, 100.0
    if score_vader >= 0.05:
        res = f"{flag} POSITIVO ✅"
        aprender_de_resena(texto_lower, texto_en_completo, res)
        return res, 0.0
 
    # ── E. Último recurso: modelo .pth entrenado ────────────────
    if model and vec:
        try:
            X = vec.transform([raw]).toarray().astype(np.float32)
            with torch.no_grad():
                p = model(torch.tensor(X)).item()
            if p >= 0.5:
                res = f"{flag} DEFECTO 🚨"
            else:
                res = f"{flag} POSITIVO ✅"
            aprender_de_resena(texto_lower, texto_en_completo, res)
            return res, (100.0 if p >= 0.5 else 0.0)
        except:
            pass
 
    # ── F. Fallback: ante la duda, reportar DEFECTO ─────────────
    res = f"{flag} DEFECTO 🚨"
    aprender_de_resena(texto_lower, texto_en_completo, res)
    return res, 100.0
 
# ══════════════════════════════════════════════════════════
# 5. APRENDIZAJE AUTOMÁTICO DE VOCABULARIO
#    Cuando el sistema analiza una reseña, extrae palabras
#    nuevas y las guarda con su etiqueta (defecto/positivo)
#    para reconocerlas automáticamente en el futuro.
# ══════════════════════════════════════════════════════════
VOCAB_APRENDIDO_FILE = os.path.join(BASE_DIR, "vocab_aprendido.json")
 
def cargar_vocab_aprendido():
    """Carga el vocabulario aprendido desde disco."""
    if not os.path.exists(VOCAB_APRENDIDO_FILE):
        return {"defecto": [], "positivo": []}
    with open(VOCAB_APRENDIDO_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except:
            return {"defecto": [], "positivo": []}
 
def guardar_vocab_aprendido(vocab):
    """Guarda el vocabulario aprendido en disco."""
    with open(VOCAB_APRENDIDO_FILE, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=4, ensure_ascii=False)
 
def extraer_palabras_clave(texto):
    """
    Extrae palabras y bigramas (pares de palabras) relevantes de un texto.
    Ignora stopwords y palabras muy cortas.
    """
    stopwords = {
        "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del",
        "al", "a", "en", "con", "por", "para", "que", "es", "son", "fue",
        "muy", "pero", "sin", "embargo", "the", "a", "an", "is", "are",
        "was", "it", "its", "and", "or", "but", "of", "to", "in", "for",
        "on", "with", "at", "by", "this", "that", "se", "me", "te", "le",
        "no", "si", "ya", "mi", "tu", "su", "o", "e", "y", "i", "he",
        "she", "we", "my", "so", "as", "be", "do", "did", "has", "had",
        "not", "have", "after", "before", "when", "also", "just", "more",
    }
    palabras = re.findall(r"[a-záéíóúüñça-z']+", texto.lower())
    palabras = [p for p in palabras if len(p) > 3 and p not in stopwords]
 
    # Palabras simples + bigramas (pares)
    tokens = list(set(palabras))
    bigramas = [f"{palabras[i]} {palabras[i+1]}" for i in range(len(palabras)-1)]
    return tokens + bigramas
 
def aprender_de_resena(texto_original, texto_en, resultado):
    """
    Después de cada análisis, extrae palabras nuevas del texto
    y las agrega al vocabulario aprendido si no existen ya.
    Solo aprende si el resultado fue decidido con alta confianza.
    """
    vocab = cargar_vocab_aprendido()
    conocidas_defecto = set(SEÑALES_DEFECTO + vocab["defecto"])
    conocidas_positivo = set(SEÑALES_POSITIVO + vocab["positivo"])
    todas_conocidas = conocidas_defecto | conocidas_positivo
 
    # Extraer palabras de ambas versiones (original + inglés)
    palabras = extraer_palabras_clave(texto_original)
    palabras += extraer_palabras_clave(texto_en)
    palabras_nuevas = [p for p in palabras if p not in todas_conocidas]
 
    if not palabras_nuevas:
        return vocab  # nada nuevo que aprender
 
    es_defecto = "🚨" in resultado
 
    nuevas_agregadas = []
    for palabra in palabras_nuevas:
        # Verificar con VADER si la palabra por sí sola tiene carga semántica
        score = sia.polarity_scores(palabra)["compound"]
 
        if es_defecto and score <= -0.01:
            # Palabra con carga negativa en contexto de defecto → aprender como defecto
            if palabra not in vocab["defecto"]:
                vocab["defecto"].append(palabra)
                nuevas_agregadas.append(f"defecto: '{palabra}'")
        elif not es_defecto and score >= 0.01:
            # Palabra con carga positiva en contexto de positivo → aprender como positivo
            if palabra not in vocab["positivo"]:
                vocab["positivo"].append(palabra)
                nuevas_agregadas.append(f"positivo: '{palabra}'")
 
    if nuevas_agregadas:
        guardar_vocab_aprendido(vocab)
 
    return vocab
 
def contar_señales_con_aprendizaje(texto, lista_base, vocab_key):
    """Busca señales en lista base + vocabulario aprendido."""
    vocab = cargar_vocab_aprendido()
    lista_completa = lista_base + vocab.get(vocab_key, [])
    return sum(1 for s in lista_completa if s in texto)
 
# ══════════════════════════════════════════════════════════
# 6. SERVIDOR FLASK
# ══════════════════════════════════════════════════════════
app = Flask(__name__)
modelo, vectorizador = cargar_modelo(), cargar_vectorizador()
 
def gestionar_historial(nueva_entrada=None, limpiar=False):
    if limpiar:
        if os.path.exists(HISTORIAL_FILE):
            os.remove(HISTORIAL_FILE)
        return []
    datos = []
    if os.path.exists(HISTORIAL_FILE):
        with open(HISTORIAL_FILE, "r", encoding="utf-8") as f:
            try: datos = json.load(f)
            except: datos = []
    if nueva_entrada:
        datos.insert(0, nueva_entrada)
        with open(HISTORIAL_FILE, "w", encoding="utf-8") as f:
            json.dump(datos, f, indent=4, ensure_ascii=False)
    return datos
 
# ══════════════════════════════════════════════════════════
# 6. INTERFAZ WEB
# ══════════════════════════════════════════════════════════
HTML = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HardwareGuard | SENATI AI</title>
    <link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg:           #05070f;
            --surface:      #0d1117;
            --card:         #111827;
            --border:       #1c2333;
            --green:        #00ff88;
            --red:          #ff3b5c;
            --blue:         #4d9eff;
            --text:         #e2e8f0;
            --muted:        #64748b;
        }
        * { margin:0; padding:0; box-sizing:border-box; }
 
        body {
            background: var(--bg);
            color: var(--text);
            font-family: 'Syne', sans-serif;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 40px 20px;
            background-image:
                radial-gradient(ellipse at 15% 15%, rgba(0,255,136,.05) 0%, transparent 55%),
                radial-gradient(ellipse at 85% 85%, rgba(77,158,255,.04) 0%, transparent 55%);
        }
 
        /* ── HEADER ── */
        .header { text-align:center; margin-bottom:36px; }
        .eyebrow {
            font-family:'Space Mono',monospace;
            font-size:10px; letter-spacing:5px;
            color:var(--blue); text-transform:uppercase; margin-bottom:10px;
        }
        h1 {
            font-size:clamp(26px,5vw,44px); font-weight:800; line-height:1.1;
            background:linear-gradient(135deg,#fff 40%,var(--green));
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;
            background-clip:text;
        }
        .sub {
            font-family:'Space Mono',monospace;
            font-size:12px; color:var(--muted); margin-top:8px;
        }
 
        /* ── CARD ── */
        .card {
            width:100%; max-width:820px;
            background:var(--card);
            border:1px solid var(--border);
            border-radius:16px; padding:26px;
            margin-bottom:20px;
        }
 
        .label {
            font-family:'Space Mono',monospace;
            font-size:10px; letter-spacing:3px;
            text-transform:uppercase; color:var(--blue); margin-bottom:10px;
        }
 
        textarea {
            width:100%; height:95px;
            background:var(--surface);
            border:1px solid var(--border);
            border-radius:10px;
            color:var(--text);
            font-family:'Space Mono',monospace;
            font-size:13px; padding:14px;
            resize:vertical; outline:none;
            transition:border-color .2s; line-height:1.6;
        }
        textarea:focus   { border-color:var(--blue); }
        textarea::placeholder { color:var(--muted); }
 
        .btn {
            width:100%; padding:13px; margin-top:12px;
            background:transparent;
            border:1.5px solid var(--green);
            border-radius:10px; color:var(--green);
            font-family:'Space Mono',monospace;
            font-size:12px; font-weight:700;
            letter-spacing:2px; text-transform:uppercase;
            cursor:pointer; transition:all .2s;
        }
        .btn:hover   { background:var(--green); color:var(--bg); }
        .btn:active  { transform:scale(.99); }
        .btn.loading { opacity:.5; pointer-events:none; }
 
        .btn-clear {
            border-color:var(--muted); color:var(--muted);
            margin-top:8px; font-size:10px;
        }
        .btn-clear:hover { background:var(--muted); color:var(--bg); }
 
        /* ── TABLE ── */
        .tbl-label {
            font-family:'Space Mono',monospace;
            font-size:10px; letter-spacing:3px;
            color:var(--muted); text-transform:uppercase; margin-bottom:14px;
        }
        table { width:100%; border-collapse:collapse; }
        thead th {
            font-family:'Space Mono',monospace;
            font-size:9px; letter-spacing:2px; text-transform:uppercase;
            color:var(--muted); padding:8px 12px;
            border-bottom:1px solid var(--border); text-align:left;
        }
        tbody tr { border-bottom:1px solid rgba(255,255,255,.04); transition:background .15s; }
        tbody tr:hover { background:rgba(255,255,255,.02); }
        td { padding:13px 12px; font-size:13px; vertical-align:middle; }
 
        .td-hora  { font-family:'Space Mono',monospace; font-size:11px; color:var(--muted); white-space:nowrap; }
        .td-texto { font-family:'Space Mono',monospace; font-size:12px; color:#94a3b8; line-height:1.5; max-width:380px; word-break:break-word; }
        .td-prob  { font-family:'Space Mono',monospace; font-size:12px; color:var(--muted); }
 
        .badge {
            display:inline-flex; align-items:center; gap:6px;
            padding:5px 12px; border-radius:6px;
            font-family:'Space Mono',monospace;
            font-size:10px; font-weight:700; letter-spacing:1px; white-space:nowrap;
        }
        .badge-defect { background:rgba(255,59,92,.12); border:1px solid rgba(255,59,92,.3); color:#ff3b5c; }
        .badge-ok     { background:rgba(0,255,136,.08);  border:1px solid rgba(0,255,136,.25); color:#00ff88; }
 
        .empty {
            text-align:center; padding:40px;
            font-family:'Space Mono',monospace; font-size:12px; color:var(--muted);
        }
        .chip-label {
            font-family:'Space Mono',monospace; font-size:9px;
            letter-spacing:2px; text-transform:uppercase;
            color:var(--muted); margin-bottom:8px;
        }
        .chip-wrap { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:6px; }
        .chip {
            padding:3px 10px; border-radius:20px;
            font-family:'Space Mono',monospace; font-size:10px;
        }
        .chip-defect { background:rgba(255,59,92,.12); border:1px solid rgba(255,59,92,.25); color:#ff3b5c; }
        .chip-ok     { background:rgba(0,255,136,.08);  border:1px solid rgba(0,255,136,.2);  color:#00ff88; }
        .btn-row { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:8px; }
        .btn-tr { border-color:#00ff88; color:#00ff88; font-size:10px; margin-top:0; }
        .btn-tr:hover { background:#00ff88; color:var(--bg); }
        .btn-ds { border-color:#4d9eff; color:#4d9eff; font-size:10px; margin-top:0; }
        .btn-ds:hover { background:#4d9eff; color:var(--bg); }
    </style>
</head>
<body>
 
<div class="header">
    <div class="eyebrow">HardwareGuard · SENATI AI</div>
    <h1>Detector de Defectos<br>en Tiempo Real</h1>
    <div class="sub">// ES · EN · PT · FR · IT · DE &nbsp;|&nbsp; VADER NLP + Modelo IA</div>
</div>
 
<div class="card">
    <div class="label">// reseña del cliente</div>
    <textarea id="txt"
        placeholder="Ej: &quot;El vendedor fue muy amable, pero la tablet se calienta y se apaga sola&quot; — cualquier idioma...">
    </textarea>
    <button class="btn" id="btnA" onclick="analizar()">▶ Analizar Producto</button>
    <div class="btn-row">
        <button class="btn btn-tr" onclick="window.open('/grafico-tiempo-real','_blank')">📊 Gráfico Tiempo Real</button>
        <button class="btn btn-ds" onclick="window.open('/reporte','_blank')">📁 Gráfico Dataset</button>
    </div>
    <button class="btn btn-clear" onclick="limpiar()">✕ Limpiar Historial</button>
</div>
 
<div class="card">
    <div class="tbl-label">// historial de análisis</div>
    <div id="wrap"><div class="empty">Sin reseñas analizadas aún.</div></div>
</div>
 
<div class="card" id="vocabCard">
    <div class="tbl-label">// vocabulario aprendido automáticamente</div>
    <div id="vocabWrap"><div class="empty">Aún no se han aprendido palabras nuevas.</div></div>
</div>
 
<script>
function esc(t){ return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }
 
function dibujar(data){
    const w = document.getElementById('wrap');
    if(!data||!data.length){ w.innerHTML='<div class="empty">Sin reseñas analizadas aún.</div>'; return; }
    const filas = data.map(d=>{
        const def = d.resultado.includes('🚨');
        const bc  = def ? 'badge-defect':'badge-ok';
        const lbl = def ? 'DEFECTO':'POSITIVO';
        const ico = def ? '🚨':'✅';
        const flag = d.resultado.split(' ')[0];
        return `<tr>
            <td class="td-hora">${d.hora}</td>
            <td class="td-texto">${esc(d.texto)}</td>
            <td><span class="badge ${bc}">${flag} ${ico} ${lbl}</span></td>
            <td class="td-prob">${d.probabilidad}%</td>
        </tr>`;
    }).join('');
    w.innerHTML=`<table>
        <thead><tr><th>Hora</th><th>Reseña</th><th>Veredicto</th><th>Fallo %</th></tr></thead>
        <tbody>${filas}</tbody>
    </table>`;
}
 
function analizar(){
    const btn=document.getElementById('btnA');
    const txt=document.getElementById('txt').value.trim();
    if(!txt) return;
    btn.classList.add('loading'); btn.textContent='⏳ Analizando...';
    fetch('/predecir',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({texto:txt})})
    .then(r=>r.json()).then(d=>{
        dibujar(d.historial);
        document.getElementById('txt').value='';
        btn.classList.remove('loading'); btn.textContent='▶ Analizar Producto';
    }).catch(()=>{ btn.classList.remove('loading'); btn.textContent='▶ Analizar Producto'; });
}
 
function limpiar(){
    if(!confirm('¿Limpiar todo el historial?')) return;
    fetch('/limpiar',{method:'POST'}).then(r=>r.json()).then(d=>dibujar(d.historial));
}
 
document.addEventListener('keydown',e=>{ if(e.ctrlKey&&e.key==='Enter') analizar(); });
 
function dibujarVocab(data){
    const w = document.getElementById('vocabWrap');
    const def = data.defecto || [];
    const pos = data.positivo || [];
    if(!def.length && !pos.length){
        w.innerHTML='<div class="empty">Aún no se han aprendido palabras nuevas.</div>';
        return;
    }
    const makeChips = (arr, cls) => arr.map(p=>`<span class="chip ${cls}">${esc(p)}</span>`).join('');
    w.innerHTML = `
        <div style="margin-bottom:12px">
            <div class="chip-label">DEFECTO detectado (${def.length})</div>
            <div class="chip-wrap">${makeChips(def,'chip-defect')}</div>
        </div>
        <div>
            <div class="chip-label">POSITIVO detectado (${pos.length})</div>
            <div class="chip-wrap">${makeChips(pos,'chip-ok')}</div>
        </div>`;
}
 
function cargarVocab(){ fetch('/vocab').then(r=>r.json()).then(d=>dibujarVocab(d)); }
 
window.onload=()=>{
    fetch('/historial').then(r=>r.json()).then(d=>dibujar(d));
    cargarVocab();
};
 
// Refrescar vocab después de cada análisis
const _analizar = analizar;
analizar = function(){
    const orig = document.getElementById('btnA').onclick;
    _analizar();
    setTimeout(cargarVocab, 1500);
};
</script>
</body>
</html>
"""
 
# ══════════════════════════════════════════════════════════
# 7. RUTAS FLASK  (idénticas a tu versión original)
# ══════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════
# ANÁLISIS DEL DATASET
# ══════════════════════════════════════════════════════════
import csv, os as _os
 
def analizar_dataset():
    """
    Analiza el dataset CSV con el motor mejorado y devuelve estadísticas.
    Compara predicción del sistema vs etiqueta real del dataset.
    """
    rutas = [
        _os.path.join(BASE_DIR, "..", "data", "hardware_reviews_clean.csv"),
        _os.path.join(BASE_DIR, "..", "data", "hardware_reviews_nltk.csv"),
    ]
    ruta_csv = next((r for r in rutas if _os.path.exists(r)), None)
    if not ruta_csv:
        return None
 
    stats = {
        "total": 0, "positivos": 0, "defectos": 0,
        "correctos": 0, "incorrectos": 0,
        "por_categoria": {}, "por_rating": {},
        "tiempo_real": {"positivos": 0, "defectos": 0},
        "muestra": []  # primeras 200 para mostrar en tabla
    }
 
    with open(ruta_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            texto = row.get("review_text", row.get("reviewText", "")).strip()
            if not texto:
                continue
 
            # Etiqueta real del dataset
            label_real = row.get("is_defective", "0").strip()
            es_defecto_real = label_real == "1"
 
            # Predicción con nuestro motor mejorado
            res, _ = predecir(texto, modelo, vectorizador)
            es_defecto_pred = "🚨" in res
 
            # Estadísticas
            stats["total"] += 1
            if es_defecto_pred:
                stats["defectos"] += 1
            else:
                stats["positivos"] += 1
 
            if es_defecto_pred == es_defecto_real:
                stats["correctos"] += 1
            else:
                stats["incorrectos"] += 1
 
            # Por categoría
            cat = row.get("category", "Otro")
            if cat not in stats["por_categoria"]:
                stats["por_categoria"][cat] = {"defectos": 0, "positivos": 0}
            if es_defecto_pred:
                stats["por_categoria"][cat]["defectos"] += 1
            else:
                stats["por_categoria"][cat]["positivos"] += 1
 
            # Por rating
            rating = row.get("rating", "?")
            if rating not in stats["por_rating"]:
                stats["por_rating"][rating] = {"defectos": 0, "positivos": 0}
            if es_defecto_pred:
                stats["por_rating"][rating]["defectos"] += 1
            else:
                stats["por_rating"][rating]["positivos"] += 1
 
            # Muestra para tabla (primeras 200)
            if i < 200:
                stats["muestra"].append({
                    "id": row.get("review_id", str(i)),
                    "producto": row.get("product", ""),
                    "categoria": cat,
                    "rating": rating,
                    "texto": texto[:120] + ("..." if len(texto) > 120 else ""),
                    "prediccion": "DEFECTO" if es_defecto_pred else "POSITIVO",
                    "real": "DEFECTO" if es_defecto_real else "POSITIVO",
                    "correcto": es_defecto_pred == es_defecto_real,
                    "flag": res.split(" ")[0],
                })
 
    # Agregar reseñas en tiempo real al stats
    historial = gestionar_historial()
    for h in historial:
        if "🚨" in h.get("resultado", ""):
            stats["tiempo_real"]["defectos"] += 1
        else:
            stats["tiempo_real"]["positivos"] += 1
 
    stats["precision"] = round(stats["correctos"] / stats["total"] * 100, 1) if stats["total"] > 0 else 0
    return stats
 
HTML_REPORTE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HardwareGuard | Reporte del Dataset</title>
    <link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
    <style>
        :root {
            --bg:#05070f; --surface:#0d1117; --card:#111827;
            --border:#1c2333; --green:#00ff88; --red:#ff3b5c;
            --blue:#4d9eff; --yellow:#fbbf24; --text:#e2e8f0; --muted:#64748b;
        }
        * { margin:0; padding:0; box-sizing:border-box; }
        body {
            background:var(--bg); color:var(--text);
            font-family:'Syne',sans-serif; padding:30px 20px;
            background-image:
                radial-gradient(ellipse at 10% 10%, rgba(0,255,136,.04) 0%, transparent 55%),
                radial-gradient(ellipse at 90% 90%, rgba(77,158,255,.04) 0%, transparent 55%);
        }
        .header { text-align:center; margin-bottom:36px; }
        .eyebrow { font-family:'Space Mono',monospace; font-size:10px; letter-spacing:5px; color:var(--blue); text-transform:uppercase; margin-bottom:10px; }
        h1 { font-size:clamp(22px,4vw,38px); font-weight:800;
            background:linear-gradient(135deg,#fff 40%,var(--blue));
            -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
        .sub { font-family:'Space Mono',monospace; font-size:11px; color:var(--muted); margin-top:8px; }
        .back-btn {
            display:inline-flex; align-items:center; gap:8px;
            font-family:'Space Mono',monospace; font-size:11px;
            color:var(--blue); text-decoration:none; letter-spacing:1px;
            border:1px solid var(--blue); padding:7px 16px; border-radius:8px;
            margin-bottom:28px; transition:all .2s;
        }
        .back-btn:hover { background:var(--blue); color:var(--bg); }
 
        /* KPI cards */
        .kpi-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:14px; margin-bottom:24px; }
        .kpi {
            background:var(--card); border:1px solid var(--border);
            border-radius:12px; padding:20px 18px; text-align:center;
        }
        .kpi-val { font-size:32px; font-weight:800; line-height:1; margin-bottom:6px; }
        .kpi-label { font-family:'Space Mono',monospace; font-size:9px; letter-spacing:2px; text-transform:uppercase; color:var(--muted); }
        .kpi.green .kpi-val { color:var(--green); }
        .kpi.red   .kpi-val { color:var(--red); }
        .kpi.blue  .kpi-val { color:var(--blue); }
        .kpi.yellow .kpi-val { color:var(--yellow); }
 
        /* Chart grid */
        .chart-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:18px; margin-bottom:24px; }
        .chart-card {
            background:var(--card); border:1px solid var(--border);
            border-radius:14px; padding:22px;
        }
        .chart-title { font-family:'Space Mono',monospace; font-size:10px; letter-spacing:2px; text-transform:uppercase; color:var(--blue); margin-bottom:16px; }
        .chart-wrap { position:relative; height:240px; }
 
        /* Table */
        .table-card { background:var(--card); border:1px solid var(--border); border-radius:14px; padding:22px; margin-bottom:24px; }
        .table-title { font-family:'Space Mono',monospace; font-size:10px; letter-spacing:2px; text-transform:uppercase; color:var(--muted); margin-bottom:14px; }
        .tbl-wrap { overflow-x:auto; }
        table { width:100%; border-collapse:collapse; min-width:700px; }
        thead th { font-family:'Space Mono',monospace; font-size:9px; letter-spacing:2px; text-transform:uppercase; color:var(--muted); padding:8px 12px; border-bottom:1px solid var(--border); text-align:left; }
        tbody tr { border-bottom:1px solid rgba(255,255,255,.04); transition:background .15s; }
        tbody tr:hover { background:rgba(255,255,255,.025); }
        td { padding:11px 12px; font-size:12px; vertical-align:middle; }
        .td-mono { font-family:'Space Mono',monospace; font-size:11px; color:#94a3b8; }
        .badge { display:inline-flex; align-items:center; gap:5px; padding:3px 10px; border-radius:5px; font-family:'Space Mono',monospace; font-size:10px; font-weight:700; white-space:nowrap; }
        .badge-defect { background:rgba(255,59,92,.12); border:1px solid rgba(255,59,92,.3); color:#ff3b5c; }
        .badge-ok     { background:rgba(0,255,136,.08);  border:1px solid rgba(0,255,136,.25); color:#00ff88; }
        .badge-wrong  { background:rgba(251,191,36,.1);  border:1px solid rgba(251,191,36,.3); color:#fbbf24; }
 
        .loading { text-align:center; padding:60px; font-family:'Space Mono',monospace; color:var(--muted); font-size:13px; }
        .progress-wrap { background:var(--surface); border-radius:8px; height:6px; margin-top:12px; overflow:hidden; }
        .progress-bar { height:100%; background:var(--blue); border-radius:8px; transition:width .3s; animation:pulse 1.5s ease-in-out infinite; }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.5} }
    </style>
</head>
<body>
    <a class="back-btn" href="javascript:window.close()">← Cerrar Reporte</a>
 
    <div class="header">
        <div class="eyebrow">HardwareGuard · SENATI AI</div>
        <h1>Reporte del Dataset</h1>
        <div class="sub" id="sub">// cargando análisis...</div>
    </div>
 
    <div id="content">
        <div class="loading">
            <div>⏳ Analizando dataset con motor IA...</div>
            <div style="font-size:11px;margin-top:8px;color:#4d9eff">Esto puede tomar unos segundos</div>
            <div class="progress-wrap" style="max-width:300px;margin:16px auto 0">
                <div class="progress-bar" style="width:60%"></div>
            </div>
        </div>
    </div>
 
<script>
function esc(t){ return String(t).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
 
const CHART_DEFAULTS = {
    color: '#e2e8f0',
    plugins: { legend: { labels: { color:'#94a3b8', font:{ family:'Space Mono', size:11 } } } },
    scales: {
        x: { ticks:{ color:'#64748b', font:{family:'Space Mono',size:10} }, grid:{ color:'rgba(255,255,255,.05)' } },
        y: { ticks:{ color:'#64748b', font:{family:'Space Mono',size:10} }, grid:{ color:'rgba(255,255,255,.06)' } }
    }
};
 
function render(data) {
    const s = data.stats;
    const precisionColor = s.precision >= 75 ? '#00ff88' : s.precision >= 60 ? '#fbbf24' : '#ff3b5c';
    document.getElementById('sub').textContent =
        `// ${s.total.toLocaleString()} reseñas del dataset · ${Object.keys(data.historial_tr).length > 0 ? Object.values(data.historial_tr).reduce((a,b)=>a+b,0) : 0} en tiempo real`;
 
    document.getElementById('content').innerHTML = `
    <!-- KPIs -->
    <div class="kpi-grid">
        <div class="kpi blue"><div class="kpi-val">${s.total.toLocaleString()}</div><div class="kpi-label">Total Reseñas</div></div>
        <div class="kpi green"><div class="kpi-val">${s.positivos.toLocaleString()}</div><div class="kpi-label">Positivos</div></div>
        <div class="kpi red"><div class="kpi-val">${s.defectos.toLocaleString()}</div><div class="kpi-label">Defectos</div></div>
        <div class="kpi ${s.precision>=75?'green':s.precision>=60?'yellow':'red'}">
            <div class="kpi-val" style="color:${precisionColor}">${s.precision}%</div>
            <div class="kpi-label">Precisión IA</div>
        </div>
        <div class="kpi green"><div class="kpi-val">${s.correctos.toLocaleString()}</div><div class="kpi-label">Correctos</div></div>
        <div class="kpi red"><div class="kpi-val">${s.incorrectos.toLocaleString()}</div><div class="kpi-label">Incorrectos</div></div>
    </div>
 
    <!-- Charts -->
    <div class="chart-grid">
        <div class="chart-card">
            <div class="chart-title">// distribución general</div>
            <div class="chart-wrap"><canvas id="cDist"></canvas></div>
        </div>
        <div class="chart-card">
            <div class="chart-title">// defectos por categoría</div>
            <div class="chart-wrap"><canvas id="cCat"></canvas></div>
        </div>
        <div class="chart-card">
            <div class="chart-title">// predicción vs rating</div>
            <div class="chart-wrap"><canvas id="cRating"></canvas></div>
        </div>
        <div class="chart-card">
            <div class="chart-title">// dataset vs tiempo real</div>
            <div class="chart-wrap"><canvas id="cTR"></canvas></div>
        </div>
    </div>
 
    <!-- Sample table -->
    <div class="table-card">
        <div class="table-title">// muestra de análisis (primeras 200 reseñas)</div>
        <div class="tbl-wrap">
        <table>
            <thead><tr>
                <th>ID</th><th>Producto</th><th>Cat</th><th>★</th>
                <th>Reseña</th><th>IA predice</th><th>Real</th><th>Match</th>
            </tr></thead>
            <tbody id="tBody"></tbody>
        </table>
        </div>
    </div>`;
 
    // Fill table
    const tbody = document.getElementById('tBody');
    tbody.innerHTML = s.muestra.map(r => `<tr>
        <td class="td-mono">${esc(r.id)}</td>
        <td style="font-size:11px;max-width:140px">${esc(r.producto)}</td>
        <td class="td-mono">${esc(r.categoria)}</td>
        <td class="td-mono">${esc(r.rating)}</td>
        <td style="font-size:11px;color:#94a3b8;max-width:260px">${esc(r.texto)}</td>
        <td><span class="badge ${r.prediccion==='DEFECTO'?'badge-defect':'badge-ok'}">${r.flag} ${r.prediccion==='DEFECTO'?'🚨':'✅'} ${r.prediccion}</span></td>
        <td><span class="badge ${r.real==='DEFECTO'?'badge-defect':'badge-ok'}">${r.real}</span></td>
        <td><span class="badge ${r.correcto?'badge-ok':'badge-wrong'}">${r.correcto?'✓':'✗'}</span></td>
    </tr>`).join('');
 
    // Chart 1: Doughnut distribución
    new Chart(document.getElementById('cDist'), {
        type: 'doughnut',
        data: {
            labels: ['POSITIVO ✅', 'DEFECTO 🚨'],
            datasets:[{ data:[s.positivos, s.defectos],
                backgroundColor:['rgba(0,255,136,.25)','rgba(255,59,92,.25)'],
                borderColor:['#00ff88','#ff3b5c'], borderWidth:2 }]
        },
        options:{ responsive:true, maintainAspectRatio:false,
            plugins:{ legend:{ position:'bottom', labels:{ color:'#94a3b8', font:{family:'Space Mono',size:11}, padding:16 } } } }
    });
 
    // Chart 2: Bar por categoría
    const cats = Object.keys(s.por_categoria);
    new Chart(document.getElementById('cCat'), {
        type:'bar',
        data:{
            labels: cats,
            datasets:[
                { label:'POSITIVO', data: cats.map(c=>s.por_categoria[c].positivos),
                  backgroundColor:'rgba(0,255,136,.2)', borderColor:'#00ff88', borderWidth:1.5 },
                { label:'DEFECTO',  data: cats.map(c=>s.por_categoria[c].defectos),
                  backgroundColor:'rgba(255,59,92,.2)',  borderColor:'#ff3b5c', borderWidth:1.5 }
            ]
        },
        options:{ responsive:true, maintainAspectRatio:false, ...CHART_DEFAULTS,
            plugins:{ legend:{ labels:{ color:'#94a3b8', font:{family:'Space Mono',size:10} } } } }
    });
 
    // Chart 3: Rating vs predicción
    const ratings = ['1','2','3','4','5'];
    new Chart(document.getElementById('cRating'), {
        type:'bar',
        data:{
            labels: ratings.map(r=>'★'+r),
            datasets:[
                { label:'POSITIVO', data: ratings.map(r=>(s.por_rating[r]||{positivos:0}).positivos),
                  backgroundColor:'rgba(0,255,136,.2)', borderColor:'#00ff88', borderWidth:1.5 },
                { label:'DEFECTO',  data: ratings.map(r=>(s.por_rating[r]||{defectos:0}).defectos),
                  backgroundColor:'rgba(255,59,92,.2)',  borderColor:'#ff3b5c', borderWidth:1.5 }
            ]
        },
        options:{ responsive:true, maintainAspectRatio:false, ...CHART_DEFAULTS,
            plugins:{ legend:{ labels:{ color:'#94a3b8', font:{family:'Space Mono',size:10} } } } }
    });
 
    // Chart 4: Dataset vs Tiempo Real
    const tr = s.tiempo_real;
    new Chart(document.getElementById('cTR'), {
        type:'bar',
        data:{
            labels:['Dataset (IA)', 'Tiempo Real'],
            datasets:[
                { label:'POSITIVO', data:[s.positivos, tr.positivos],
                  backgroundColor:'rgba(0,255,136,.2)', borderColor:'#00ff88', borderWidth:1.5 },
                { label:'DEFECTO',  data:[s.defectos, tr.defectos],
                  backgroundColor:'rgba(255,59,92,.2)',  borderColor:'#ff3b5c', borderWidth:1.5 }
            ]
        },
        options:{ responsive:true, maintainAspectRatio:false, ...CHART_DEFAULTS,
            plugins:{ legend:{ labels:{ color:'#94a3b8', font:{family:'Space Mono',size:10} } } } }
    });
}
 
// Load data
var base = window.location.protocol + '//' + window.location.hostname + ':5000';
fetch(base + '/api/reporte')
    .then(r=>r.json())
    .then(data=>render(data))
    .catch(()=>{
        document.getElementById('content').innerHTML =
            '<div class="loading" style="color:#ff3b5c">❌ Error al cargar el reporte. Verifica que el dataset esté en data/hardware_reviews_clean.csv</div>';
    });
</script>
</body>
</html>
"""
 
@app.route("/")
def home(): return render_template_string(HTML)
 
HTML_TIEMPO_REAL = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HardwareGuard | Grafico Tiempo Real</title>
    <link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
    <style>
        :root { --bg:#05070f; --card:#111827; --border:#1c2333; --green:#00ff88; --red:#ff3b5c; --blue:#4d9eff; --text:#e2e8f0; --muted:#64748b; }
        * { margin:0; padding:0; box-sizing:border-box; }
        body { background:var(--bg); color:var(--text); font-family:'Syne',sans-serif; padding:30px 20px;
            background-image: radial-gradient(ellipse at 15% 15%, rgba(0,255,136,.05) 0%, transparent 55%); }
        .topbar { display:flex; gap:10px; margin-bottom:28px; }
        .btn-top { font-family:'Space Mono',monospace; font-size:10px; letter-spacing:1px; text-transform:uppercase;
            background:transparent; border:1px solid var(--muted); color:var(--muted); padding:7px 16px;
            border-radius:8px; cursor:pointer; transition:all .2s; }
        .btn-top:hover { border-color:var(--green); color:var(--green); }
        .header { text-align:center; margin-bottom:32px; }
        .eyebrow { font-family:'Space Mono',monospace; font-size:10px; letter-spacing:5px; color:var(--green); text-transform:uppercase; margin-bottom:10px; }
        h1 { font-size:clamp(22px,4vw,36px); font-weight:800;
            background:linear-gradient(135deg,#fff 40%,var(--green));
            -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
        .sub { font-family:'Space Mono',monospace; font-size:11px; color:var(--muted); margin-top:8px; }
        .kpi-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:14px; margin-bottom:24px; }
        .kpi { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:18px; text-align:center; }
        .kpi-val { font-size:30px; font-weight:800; line-height:1; margin-bottom:5px; }
        .kpi-label { font-family:'Space Mono',monospace; font-size:9px; letter-spacing:2px; text-transform:uppercase; color:var(--muted); }
        .kv-green { color:var(--green); } .kv-red { color:var(--red); } .kv-blue { color:var(--blue); }
        .chart-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:18px; margin-bottom:24px; }
        .chart-card { background:var(--card); border:1px solid var(--border); border-radius:14px; padding:22px; }
        .chart-full { background:var(--card); border:1px solid var(--border); border-radius:14px; padding:22px; margin-bottom:24px; }
        .chart-title { font-family:'Space Mono',monospace; font-size:10px; letter-spacing:2px; text-transform:uppercase; color:var(--green); margin-bottom:16px; }
        .chart-wrap { position:relative; height:240px; }
        .table-card { background:var(--card); border:1px solid var(--border); border-radius:14px; padding:22px; }
        .tbl-title { font-family:'Space Mono',monospace; font-size:10px; letter-spacing:2px; text-transform:uppercase; color:var(--muted); margin-bottom:14px; }
        table { width:100%; border-collapse:collapse; }
        thead th { font-family:'Space Mono',monospace; font-size:9px; letter-spacing:2px; text-transform:uppercase;
            color:var(--muted); padding:8px 12px; border-bottom:1px solid var(--border); text-align:left; }
        tbody tr { border-bottom:1px solid rgba(255,255,255,.04); }
        tbody tr:hover { background:rgba(255,255,255,.02); }
        td { padding:12px; font-size:12px; vertical-align:middle; }
        .td-mono { font-family:'Space Mono',monospace; font-size:11px; color:#94a3b8; white-space:nowrap; }
        .badge { display:inline-flex; align-items:center; gap:5px; padding:4px 10px; border-radius:5px;
            font-family:'Space Mono',monospace; font-size:10px; font-weight:700; white-space:nowrap; }
        .bd { background:rgba(255,59,92,.12); border:1px solid rgba(255,59,92,.3); color:#ff3b5c; }
        .bp { background:rgba(0,255,136,.08); border:1px solid rgba(0,255,136,.25); color:#00ff88; }
        .empty { text-align:center; padding:60px; font-family:'Space Mono',monospace; font-size:13px; color:var(--muted); }
    </style>
</head>
<body>
    <div class="topbar">
        <button class="btn-top" onclick="window.close()">← Cerrar</button>
        <button class="btn-top" onclick="cargar()">↺ Actualizar</button>
    </div>
    <div class="header">
        <div class="eyebrow">HardwareGuard · Tiempo Real</div>
        <h1>Grafico de Resenas Analizadas en Vivo</h1>
        <div class="sub" id="sub">// cargando...</div>
    </div>
    <div id="kpis"></div>
    <div id="charts"></div>
    <div id="evol"></div>
    <div id="tabla"></div>
 
<script>
function esc(t){ return String(t).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
 
var chartDona=null, chartIdioma=null, chartEvol=null;
 
function destruir(c){ if(c){ c.destroy(); } return null; }
 
function render(data){
    if(!data || data.length===0){
        document.getElementById('sub').textContent='// sin resenas analizadas aun';
        document.getElementById('kpis').innerHTML='<div class="empty">Sin resenas aun.<br><small>Analiza algunas resenas en la pantalla principal.</small></div>';
        document.getElementById('charts').innerHTML='';
        document.getElementById('evol').innerHTML='';
        document.getElementById('tabla').innerHTML='';
        return;
    }
 
    var total    = data.length;
    var defectos = data.filter(function(d){ return d.resultado && d.resultado.indexOf('🚨')>=0; }).length;
    var positivos= total - defectos;
    var pct      = total>0 ? ((defectos/total)*100).toFixed(1) : '0.0';
 
    document.getElementById('sub').textContent='// '+total+' resenas analizadas en tiempo real';
 
    // KPIs
    document.getElementById('kpis').innerHTML=
        '<div class="kpi-grid">'+
        '<div class="kpi"><div class="kpi-val kv-blue">'+total+'</div><div class="kpi-label">Total</div></div>'+
        '<div class="kpi"><div class="kpi-val kv-green">'+positivos+'</div><div class="kpi-label">Positivos</div></div>'+
        '<div class="kpi"><div class="kpi-val kv-red">'+defectos+'</div><div class="kpi-label">Defectos</div></div>'+
        '<div class="kpi"><div class="kpi-val '+(parseFloat(pct)>50?'kv-red':'kv-green')+'">'+pct+'%</div><div class="kpi-label">% Defectos</div></div>'+
        '</div>';
 
    // Idiomas
    var idiomas={};
    data.forEach(function(d){
        var r=d.resultado||'';
        var flag=r.split(' ')[0];
        var n={'🇵🇪':'Espanol','🇺🇸':'Ingles','🇧🇷':'Portugues','🇫🇷':'Frances','🇮🇹':'Italiano','🇩🇪':'Aleman','🌐':'Otro'}[flag]||'Otro';
        idiomas[n]=(idiomas[n]||0)+1;
    });
 
    // Charts grid
    document.getElementById('charts').innerHTML=
        '<div class="chart-grid">'+
        '<div class="chart-card"><div class="chart-title">// distribucion</div><div class="chart-wrap"><canvas id="cDona"></canvas></div></div>'+
        '<div class="chart-card"><div class="chart-title">// por idioma</div><div class="chart-wrap"><canvas id="cIdioma"></canvas></div>'+
        '</div>';
 
    chartDona   = destruir(chartDona);
    chartIdioma = destruir(chartIdioma);
 
    chartDona = new Chart(document.getElementById('cDona'),{
        type:'doughnut',
        data:{labels:['POSITIVO','DEFECTO'],datasets:[{data:[positivos,defectos],
            backgroundColor:['rgba(0,255,136,.25)','rgba(255,59,92,.25)'],
            borderColor:['#00ff88','#ff3b5c'],borderWidth:2}]},
        options:{responsive:true,maintainAspectRatio:false,
            plugins:{legend:{position:'bottom',labels:{color:'#94a3b8',font:{family:'Space Mono',size:11},padding:16}}}}
    });
 
    var cols=['#4d9eff','#00ff88','#fbbf24','#ff3b5c','#a78bfa','#34d399'];
    chartIdioma = new Chart(document.getElementById('cIdioma'),{
        type:'bar',
        data:{labels:Object.keys(idiomas),datasets:[{label:'Resenas',data:Object.values(idiomas),
            backgroundColor:cols.map(function(c){return c+'44';}),
            borderColor:cols,borderWidth:1.5}]},
        options:{responsive:true,maintainAspectRatio:false,
            plugins:{legend:{display:false}},
            scales:{x:{ticks:{color:'#64748b',font:{family:'Space Mono',size:10}},grid:{color:'rgba(255,255,255,.05)'}},
                    y:{ticks:{color:'#64748b',font:{family:'Space Mono',size:10},stepSize:1},grid:{color:'rgba(255,255,255,.06)'}}}}
    });
 
    // Evolucion
    var ultimos=[].concat(data).reverse().slice(0,20);
    var evLabels=ultimos.map(function(d,i){return d.hora||('#'+(i+1));});
    var evData  =ultimos.map(function(d){return (d.resultado&&d.resultado.indexOf('🚨')>=0)?1:0;});
    var evColors=evData.map(function(v){return v?'rgba(255,59,92,.5)':'rgba(0,255,136,.4)';});
    var evBorder=evData.map(function(v){return v?'#ff3b5c':'#00ff88';});
 
    document.getElementById('evol').innerHTML=
        '<div class="chart-full"><div class="chart-title">// ultimas 20 resenas (rojo=defecto · verde=positivo)</div>'+
        '<div class="chart-wrap"><canvas id="cEvol"></canvas></div></div>';
 
    chartEvol=destruir(chartEvol);
    chartEvol=new Chart(document.getElementById('cEvol'),{
        type:'bar',
        data:{labels:evLabels,datasets:[{data:evData,backgroundColor:evColors,borderColor:evBorder,borderWidth:1.5}]},
        options:{responsive:true,maintainAspectRatio:false,
            plugins:{legend:{display:false}},
            scales:{x:{ticks:{color:'#64748b',font:{family:'Space Mono',size:9},maxRotation:45},grid:{color:'rgba(255,255,255,.05)'}},
                    y:{min:0,max:1,ticks:{color:'#64748b',font:{family:'Space Mono',size:10},
                        callback:function(v){return v===1?'DEFECTO':'POSITIVO';}},
                        grid:{color:'rgba(255,255,255,.06)'}}}}
    });
 
    // Tabla historial
    var filas=data.map(function(d){
        var def=d.resultado&&d.resultado.indexOf('🚨')>=0;
        var flag=(d.resultado||'').split(' ')[0];
        return '<tr>'+
            '<td class="td-mono">'+esc(d.hora||'')+'</td>'+
            '<td style="font-size:11px;color:#94a3b8;max-width:400px">'+esc(d.texto||'')+'</td>'+
            '<td><span class="badge '+(def?'bd':'bp')+'">'+flag+' '+(def?'🚨 DEFECTO':'✅ POSITIVO')+'</span></td>'+
            '</tr>';
    }).join('');
 
    document.getElementById('tabla').innerHTML=
        '<div class="table-card">'+
        '<div class="tbl-title">// historial completo ordenado por hora</div>'+
        '<table><thead><tr><th>Hora</th><th>Resena</th><th>Resultado</th></tr></thead>'+
        '<tbody>'+filas+'</tbody></table></div>';
}
 
function cargar(){
    var base = window.location.protocol + '//' + window.location.hostname + ':5000';
    fetch(base + '/historial')
        .then(function(r){
            if(!r.ok){ throw new Error('HTTP ' + r.status); }
            return r.json();
        })
        .then(function(d){ render(d); })
        .catch(function(err){
            document.getElementById('sub').textContent='// error: ' + err.message;
            document.getElementById('kpis').innerHTML='<div class="empty" style="color:#ff3b5c">'+
                'Error al cargar datos.<br><small>Asegurate de que predictor.py este corriendo en el puerto 5000</small></div>';
        });
}
 
cargar();
setInterval(cargar, 10000);
</script>
</body>
</html>
"""
 
 
@app.route("/grafico-tiempo-real")
def grafico_tr(): return render_template_string(HTML_TIEMPO_REAL)
 
@app.route("/reporte")
def reporte(): return render_template_string(HTML_REPORTE)
 
@app.route("/api/reporte")
def api_reporte():
    import json as _json
    stats = analizar_dataset()
    if not stats:
        return jsonify({"error": "Dataset no encontrado"}), 404
    historial = gestionar_historial()
    tr = {"positivos": sum(1 for h in historial if "🚨" not in h.get("resultado","")),
          "defectos":  sum(1 for h in historial if "🚨" in h.get("resultado",""))}
    stats["tiempo_real"] = tr
    return jsonify({"stats": stats, "historial_tr": tr})
 
@app.route("/historial")
def api_historial(): return jsonify(gestionar_historial())
 
@app.route("/vocab")
def api_vocab():
    """Devuelve el vocabulario aprendido automáticamente."""
    return jsonify(cargar_vocab_aprendido())
 
@app.route("/limpiar", methods=["POST"])
def api_limpiar(): return jsonify({"historial": gestionar_historial(limpiar=True)})
 
@app.route("/predecir", methods=["POST"])
def api_predecir():
    data = request.get_json()
    texto = data.get("texto", "")
    res, prob = predecir(texto, modelo, vectorizador)
    entrada = {
        "hora":        datetime.now().strftime("%H:%M:%S"),
        "texto":       texto,
        "resultado":   res,
        "probabilidad": prob
    }
    return jsonify({"historial": gestionar_historial(entrada)})
 
if __name__ == "__main__":
    threading.Thread(
        target=lambda: (
            __import__('time').sleep(1),
            webbrowser.open("http://127.0.0.1:5000")
        ),
        daemon=True
    ).start()
    app.run(port=5000, debug=False)