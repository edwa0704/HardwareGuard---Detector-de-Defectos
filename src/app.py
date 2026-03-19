import os
import re
import json
import webbrowser
import threading
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string

# ─────────────────────────────
# 1. GESTIÓN DE DATOS (JSON)
# ─────────────────────────────
# UBICACIÓN: Se guardará en la raíz de tu carpeta del proyecto
HISTORIAL_PATH = "historial_resenas.json"

def obtener_datos():
    if not os.path.exists(HISTORIAL_PATH):
        return []
    with open(HISTORIAL_PATH, "r", encoding="utf-8") as f:
        try: return json.load(f)
        except: return []

def guardar_dato(nueva_entrada):
    datos = obtener_datos()
    datos.insert(0, nueva_entrada) # Lo más nuevo primero
    with open(HISTORIAL_PATH, "w", encoding="utf-8") as f:
        json.dump(datos, f, indent=4, ensure_ascii=False)
    return datos

# ─────────────────────────────
# 2. LÓGICA DE ANÁLISIS (CONTEXTO)
# ─────────────────────────────
def analizar_sentimiento(texto):
    t = texto.lower()
    # Palabras clave para forzar el contexto
    negativas = ["roto", "falla", "calienta", "ruido", "lento", "malo", "defecto", "devolucion"]
    positivas = ["bueno", "excelente", "perfecto", "rapido", "recomiendo", "original"]
    
    score = 50 # Empezamos en ambiguo
    
    for p in negativas:
        if p in t: score += 20
    for p in positivas:
        if p in t: score -= 20
    
    # Limitar entre 0 y 100
    score = max(0, min(100, score))
    
    if score > 60: return "DEFECTO 🚨", score
    elif score < 40: return "POSITIVO ✅", score
    else: return "AMBIGUO ⚠️", score

# ─────────────────────────────
# 3. INTERFAZ WEB (TODO EN UNO)
# ─────────────────────────────
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>HardwareGuard | Panel de Reseñas</title>
    <style>
        body { background: #0f172a; color: white; font-family: 'Segoe UI', sans-serif; padding: 30px; }
        .container { max-width: 1000px; margin: auto; }
        .input-box { background: #1e293b; padding: 20px; border-radius: 12px; margin-bottom: 20px; }
        textarea { width: 100%; height: 60px; border-radius: 8px; padding: 10px; font-size: 16px; border: none; }
        .btn-analizar { background: #22c55e; color: white; border: none; padding: 12px; width: 100%; border-radius: 8px; cursor: pointer; font-weight: bold; margin-top: 10px; }
        .btn-borrar { background: #ef4444; color: white; border: none; padding: 5px 10px; border-radius: 5px; cursor: pointer; font-size: 12px; float: right; }
        
        table { width: 100%; border-collapse: collapse; margin-top: 20px; background: #1e293b; border-radius: 10px; overflow: hidden; }
        th, td { padding: 15px; text-align: left; border-bottom: 1px solid #334155; }
        th { background: #334155; color: #94a3b8; }
        .rojo { color: #f87171; font-weight: bold; }
        .verde { color: #4ade80; font-weight: bold; }
        .naranja { color: #fbbf24; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 HardwareGuard: Detector de Defectos</h1>
        
        <div class="input-box">
            <textarea id="texto" placeholder="Pegue la reseña del cliente aquí..."></textarea>
            <button class="btn-analizar" onclick="analizar()">Analizar y Guardar en Historial</button>
        </div>

        <h3>Historial Guardado <button class="btn-borrar" onclick="borrarTodo()">Borrar Todo</button></h3>
        <table>
            <thead>
                <tr>
                    <th>Hora</th>
                    <th>Reseña</th>
                    <th>Estado</th>
                    <th>Confianza</th>
                </tr>
            </thead>
            <tbody id="tabla-cuerpo"></tbody>
        </table>
    </div>

    <script>
    // Al cargar la página, traer historial
    fetch('/get_historial').then(r => r.json()).then(data => actualizarTabla(data));

    function analizar() {
        const texto = document.getElementById("texto").value;
        if(!texto) return;

        fetch("/predecir", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({texto: texto})
        })
        .then(res => res.json())
        .then(data => {
            actualizarTabla(data);
            document.getElementById("texto").value = "";
        });
    }

    function borrarTodo() {
        if(confirm("¿Seguro que quieres borrar todo el historial?")) {
            fetch("/borrar", {method: "POST"}).then(() => actualizarTabla([]));
        }
    }

    function actualizarTabla(datos) {
        let html = "";
        datos.forEach(d => {
            let clase = d.resultado.includes("🚨") ? "rojo" : (d.resultado.includes("✅") ? "verde" : "naranja");
            html += `<tr>
                <td><small>${d.fecha}</small></td>
                <td>${d.texto}</td>
                <td class="${clase}">${d.resultado}</td>
                <td>${d.probabilidad}%</td>
            </tr>`;
        });
        document.getElementById("tabla-cuerpo").innerHTML = html;
    }
    </script>
</body>
</html>
"""

# ─────────────────────────────
# 4. RUTAS DE FLASK
# ─────────────────────────────
app = Flask(__name__)

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/get_historial")
def get_historial():
    return jsonify(obtener_datos())

@app.route("/borrar", methods=["POST"])
def borrar():
    if os.path.exists(HISTORIAL_PATH):
        os.remove(HISTORIAL_PATH)
    return jsonify([])

@app.route("/predecir", methods=["POST"])
def predecir():
    raw_data = request.get_json()
    texto = raw_data.get("texto", "")
    
    res, prob = analizar_sentimiento(texto)
    
    nueva_entrada = {
        "fecha": datetime.now().strftime("%H:%M:%S"),
        "texto": texto,
        "resultado": res,
        "probabilidad": prob
    }
    
    historial_actualizado = guardar_dato(nueva_entrada)
    return jsonify(historial_actualizado)

if __name__ == "__main__":
    def abrir_browser():
        import time
        time.sleep(1.2)
        webbrowser.open("http://127.0.0.1:5000")

    threading.Thread(target=abrir_browser).start()
    app.run(port=5000, debug=False)