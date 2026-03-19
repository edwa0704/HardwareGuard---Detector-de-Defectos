import os, json, csv, sys
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
 
# ══════════════════════════════════════════════════════════
# 1. RUTAS
# ══════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
CSV_PATH = os.path.join(DATA_DIR, "hardware_reviews_clean.csv")
MODEL_OUT = os.path.join(DATA_DIR, "hardwareguard_model.pth")
VOCAB_OUT  = os.path.join(DATA_DIR, "vocab.json")
 
sia = SentimentIntensityAnalyzer()
 
# ══════════════════════════════════════════════════════════
# 2. SEÑALES DEL MOTOR MEJORADO (mismas que predictor.py)
#    Usadas para etiquetar los 796 casos neutrales
# ══════════════════════════════════════════════════════════
SEÑALES_DEFECTO = [
    "overheat","overheating","gets hot","too hot","burning","burns","heats up",
    "stopped working","stop working","stops working","shuts off","shuts down",
    "shut off","shutdown","turns off","turned off","won't turn on","wont turn on",
    "not working","doesnt work","doesn't work","ceased to work","no longer works",
    "stopped functioning","failed after","died after","broke after","broken after",
    "broke down","screen","cracked screen","broken screen","no display","black screen",
    "flickering","flickers","pixelated","no charge","won't charge","doesn't charge",
    "drains fast","battery dies","bad battery","battery dead","battery drains",
    "slow","laggy","lag","freezes","frozen","freeze","hangs","unresponsive",
    "crashes","keeps crashing","defect","defective","broken","damaged","faulty",
    "fault","malfunction","malfunctioning","useless","unusable","waste",
    "terrible","horrible","awful","garbage","trash","junk","piece of junk",
    "no wifi","no bluetooth","disconnects","won't connect","connection issues",
    "calienta","caliente","sobrecalienta","quema","apaga","apagarse","se apaga",
    "reinicia","reinicio","pantalla rota","no enciende","no prende","no se ve",
    "no carga","se descarga","lento","lenta","cuelga","congela","tarda mucho",
    "defecto","falla","fallo","roto","dañado","malogrado","no funciona",
    "inútil","inutilizable","basura","no conecta","se desconecta",
    "dejó de funcionar","dejo de funcionar","dejó de encender","dejo de encender",
    "dejó de cargar","dejo de cargar","ya no funciona","ya no enciende",
    "ya no carga","ya no sirve","duró poco","duro poco","se arruinó","se arruino",
    "se malogró","se malogro","se dañó","se daño","se rompió","se rompio",
    "parou de funcionar","nao funciona","não funciona","quebrado","defeituoso",
    "travando","trava","porquería","porqueria","pésimo","pesimo","malísimo",
    "malisimo","horrible","espantoso","una basura","es basura","fatal",
    "decepcionante","me arrepiento","estafa","fraude","me tiene harto",
    "qué asco","que asco","da asco","inservible","dinero tirado","plata botada",
    "no vale nada","lo peor","mediocre",
]
 
SEÑALES_POSITIVO = [
    "excelente","excellent","perfecto","perfect","funciona","works","funciona bien",
    "works great","rápido","rapido","fast","veloz","speedy","fluido","smooth",
    "potente","powerful","increíble","increible","incredible","amazing",
    "buena calidad","good quality","calidad","quality","resistente","durable",
    "sólido","solido","solid","robusto","recomiendo","recommend",
    "cumple lo prometido","tal como se describe","buen producto","buena calidad",
    "vale la pena","outstanding","absolutely love it","best purchase",
    "worth every penny","highly recommend","flawless","exceptional",
    "bueno","buena","bien","funciona bien","me gustó","me gusto","me gusta",
    "recomendado","recomendable","cumple su función","cumple su funcion",
    "buen precio calidad","good","great","works well","happy with","pleased",
    "joya","una joya","espectacular","lo máximo","lo maximo","de lujo","top",
    "lo mejor","el mejor","la mejor","impecable","maravilloso","maravillosa",
    "genial","vale cada centavo","vale cada sol","valió la pena",
    "lo recomiendo","100% recomendado","sin defectos","no está mal","no esta mal",
    "sin fallas","no falla","no se cuelga","no se calienta","no da problemas",
]
 
SENTIMIENTOS_ES = {
    "porquería":-1.0,"porqueria":-1.0,"pésimo":-1.0,"pesimo":-1.0,
    "malísimo":-1.0,"malisimo":-1.0,"horrible":-1.0,"fatal":-1.0,
    "decepcionante":-1.0,"me arrepiento":-1.0,"estafa":-1.0,
    "me tiene harto":-1.0,"qué asco":-1.0,"que asco":-1.0,
    "plata botada":-1.0,"dinero tirado":-1.0,"no vale nada":-1.0,
    "lo peor":-1.0,"inservible":-1.0,"waste of money":-0.9,
    "malo":-0.7,"mala":-0.7,"no me gustó":-0.7,"no me gusto":-0.7,
    "no me gusta":-0.7,"no recomiendo":-0.7,"mala compra":-0.7,
    "no vale la pena":-0.7,"esperaba más":-0.7,"defraudado":-0.7,
    "not worth it":-0.7,"disappointed":-0.7,"poor quality":-0.7,
    "regular":-0.4,"más o menos":-0.4,"mas o menos":-0.4,
    "podría ser mejor":-0.4,"mejorable":-0.4,"not great":-0.4,
    "joya":1.0,"una joya":1.0,"excelente":1.0,"espectacular":1.0,
    "increíble":1.0,"increible":1.0,"lo máximo":1.0,"lo maximo":1.0,
    "perfecto":1.0,"perfecta":1.0,"impecable":1.0,"maravilloso":1.0,
    "genial":1.0,"vale cada centavo":1.0,"vale cada sol":1.0,
    "valió la pena":1.0,"100% recomendado":1.0,"sin defectos":1.0,
    "outstanding":1.0,"flawless":1.0,"best purchase":1.0,
    "bueno":0.7,"buena":0.7,"bien":0.7,"me gustó":0.7,"me gusto":0.7,
    "recomendado":0.7,"cumple lo prometido":0.7,"buen producto":0.7,
    "vale la pena":0.7,"good":0.7,"great":0.7,"works well":0.7,
    "no está mal":0.4,"no esta mal":0.4,"sirve":0.4,"funciona":0.4,
    "cumple":0.4,"aceptable":0.4,"decent":0.4,"okay":0.4,"works":0.4,
}
 
NEG_DE_POSITIVO = [
    "no estoy contento","no estoy satisfecho","no me gusta","no me gustó",
    "no me gusto","no cumple","no cumplió","no cumplio","no es bueno",
    "no es buena","no funciona bien","no vale la pena","no lo recomiendo",
    "no recomiendo","not happy","not satisfied","not good","not worth",
]
NEG_DE_NEGATIVO = [
    "no está mal","no esta mal","no es malo","no es mala","no tiene fallas",
    "no tiene defectos","sin fallas","sin defectos","no falla","no se cuelga",
    "no se calienta","no da problemas","not bad","no issues","no problems",
]
 
# ══════════════════════════════════════════════════════════
# 3. MOTOR DE ETIQUETADO (para neutrales)
# ══════════════════════════════════════════════════════════
def etiquetar_con_motor(texto):
    """
    Aplica el motor mejorado para determinar si una reseña es defecto (1) o no (0).
    Retorna: (etiqueta: int, confianza: str)
    """
    t = texto.lower().strip()
 
    # A. Negaciones primero
    for frase in NEG_DE_POSITIVO:
        if frase in t:
            return 1, "negacion"
    for frase in NEG_DE_NEGATIVO:
        if frase in t:
            return 0, "negacion"
 
    # B. Señales directas de hardware
    n_def = sum(1 for s in SEÑALES_DEFECTO  if s in t)
    n_pos = sum(1 for s in SEÑALES_POSITIVO if s in t)
    if n_def > 0:
        return 1, "señal_directa"
    if n_pos > 0:
        return 0, "señal_directa"
 
    # C. Diccionario ES nativo
    candidatos = sorted(SENTIMIENTOS_ES.keys(), key=len, reverse=True)
    scores = [SENTIMIENTOS_ES[f] for f in candidatos if f in t]
    if scores:
        score_es = sum(scores) / len(scores)
        if score_es <= -0.3:
            return 1, "diccionario_es"
        if score_es >= 0.3:
            return 0, "diccionario_es"
 
    # D. VADER
    score_vader = sia.polarity_scores(t)['compound']
    if score_vader <= -0.05:
        return 1, "vader"
    if score_vader >= 0.05:
        return 0, "vader"
 
    # E. Si no hay señal clara → usar rating si está disponible
    return None, "sin_señal"
 
# ══════════════════════════════════════════════════════════
# 4. ARQUITECTURA DEL MODELO
# ══════════════════════════════════════════════════════════
class Modelo(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.red = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256), torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 128), torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),  torch.nn.ReLU(),
            torch.nn.Linear(64, 1),    torch.nn.Sigmoid()
        )
    def forward(self, x): return self.red(x)
 
# ══════════════════════════════════════════════════════════
# 5. CARGA Y PREPARACIÓN DEL DATASET
# ══════════════════════════════════════════════════════════
def cargar_dataset():
    print(f"\n📂 Cargando dataset: {CSV_PATH}")
    if not os.path.exists(CSV_PATH):
        print(f"❌ No se encontró: {CSV_PATH}")
        sys.exit(1)
 
    textos, etiquetas = [], []
    stats = {"positivos":0, "negativos":0, "neutrales_etiquetados":0,
             "neutrales_descartados":0, "por_motor":{}}
 
    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texto = row.get("review_text", "").strip()
            if not texto:
                continue
 
            sentiment = row.get("sentiment", "").strip()
            is_defective = row.get("is_defective", "0").strip()
            rating = int(row.get("rating", "3"))
 
            if sentiment in ("positive", "negative"):
                # Usar etiqueta real del dataset
                etiqueta = int(is_defective)
                textos.append(texto)
                etiquetas.append(etiqueta)
                if etiqueta == 1:
                    stats["negativos"] += 1
                else:
                    stats["positivos"] += 1
 
            elif sentiment == "neutral":
                # Usar motor mejorado para etiquetar
                etiqueta, metodo = etiquetar_con_motor(texto)
 
                if etiqueta is None:
                    # Si el motor no puede decidir, usar rating como desempate
                    if rating <= 2:
                        etiqueta, metodo = 1, "rating_bajo"
                    elif rating >= 4:
                        etiqueta, metodo = 0, "rating_alto"
                    else:
                        stats["neutrales_descartados"] += 1
                        continue  # rating 3 sin señal → descartar
 
                textos.append(texto)
                etiquetas.append(etiqueta)
                stats["neutrales_etiquetados"] += 1
                stats["por_motor"][metodo] = stats["por_motor"].get(metodo, 0) + 1
 
    total = len(textos)
    print(f"✅ Dataset preparado:")
    print(f"   Positivos (dataset real): {stats['positivos']}")
    print(f"   Negativos (dataset real): {stats['negativos']}")
    print(f"   Neutrales etiquetados:    {stats['neutrales_etiquetados']}")
    print(f"   Neutrales descartados:    {stats['neutrales_descartados']}")
    print(f"   Método motor: {stats['por_motor']}")
    print(f"   TOTAL para entrenamiento: {total}")
    print(f"   Balance: {sum(etiquetas)} defectos / {total-sum(etiquetas)} positivos")
    return textos, etiquetas
 
# ══════════════════════════════════════════════════════════
# 6. VECTORIZACIÓN
# ══════════════════════════════════════════════════════════
def vectorizar(textos):
    print("\n🔤 Vectorizando texto (TF-IDF)...")
    vec = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),      # unigramas + bigramas
        min_df=2,                 # al menos 2 apariciones
        sublinear_tf=True,        # suavizado logarítmico
        strip_accents="unicode",
    )
    X = vec.fit_transform(textos).toarray().astype(np.float32)
    print(f"   Vocabulario: {len(vec.vocabulary_)} términos")
    print(f"   Matriz: {X.shape}")
    return X, vec
 
# ══════════════════════════════════════════════════════════
# 7. ENTRENAMIENTO
# ══════════════════════════════════════════════════════════
def entrenar(X, y):
    print("\n🧠 Iniciando entrenamiento...")
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")
 
    # Tensores
    X_tr = torch.tensor(X_train)
    y_tr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_te = torch.tensor(X_test)
    y_te = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1)
 
    # Modelo
    model = Modelo(X_train.shape[1])
 
    # Pesos para desbalance de clases
    n_pos = sum(y_train)
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos]) if n_pos > 0 else torch.tensor([1.0])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
 
    # Usamos salida pre-sigmoid para BCEWithLogitsLoss
    # Ajustamos el modelo para devolver logits en entrenamiento
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
 
    EPOCHS = 60
    BATCH  = 128
    mejor_acc = 0
    mejor_estado = None
 
    print(f"\n{'Época':>6} {'Loss':>10} {'Acc Train':>10} {'Acc Test':>10}")
    print("-" * 42)
 
    for epoch in range(EPOCHS):
        model.train()
        # Mini-batches
        idx = torch.randperm(len(X_tr))
        epoch_loss = 0
        for i in range(0, len(X_tr), BATCH):
            batch_idx = idx[i:i+BATCH]
            xb, yb = X_tr[batch_idx], y_tr[batch_idx]
 
            optimizer.zero_grad()
            # Para BCEWithLogitsLoss necesitamos logits, no sigmoid
            # Temporalmente usamos la red sin sigmoid final
            out = model.red[:-1](xb)  # sin Sigmoid
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
 
        # Evaluación
        model.eval()
        with torch.no_grad():
            pred_tr = (model(X_tr) >= 0.5).float()
            pred_te = (model(X_te) >= 0.5).float()
            acc_tr  = (pred_tr == y_tr).float().mean().item()
            acc_te  = (pred_te == y_te).float().mean().item()
 
        scheduler.step(epoch_loss)
 
        if (epoch + 1) % 10 == 0:
            print(f"{epoch+1:>6} {epoch_loss:>10.4f} {acc_tr*100:>9.1f}% {acc_te*100:>9.1f}%")
 
        # Guardar mejor modelo
        if acc_te > mejor_acc:
            mejor_acc = acc_te
            mejor_estado = {k: v.clone() for k, v in model.state_dict().items()}
 
    # Restaurar mejor modelo
    model.load_state_dict(mejor_estado)
    print(f"\n✅ Mejor accuracy en test: {mejor_acc*100:.1f}%")
 
    # Reporte final
    model.eval()
    with torch.no_grad():
        preds = (model(X_te) >= 0.5).numpy().astype(int).flatten()
    y_real = y_te.numpy().astype(int).flatten()
 
    print("\n📊 Reporte de clasificación:")
    print(classification_report(y_real, preds, target_names=["POSITIVO", "DEFECTO"]))
 
    print("📊 Matriz de confusión:")
    cm = confusion_matrix(y_real, preds)
    print(f"   VP (DEFECTO correcto):   {cm[1][1]}")
    print(f"   VN (POSITIVO correcto):  {cm[0][0]}")
    print(f"   FP (falso defecto):      {cm[0][1]}")
    print(f"   FN (defecto no detectado): {cm[1][0]}")
 
    return model
 
# ══════════════════════════════════════════════════════════
# 8. GUARDAR MODELO Y VOCABULARIO
# ══════════════════════════════════════════════════════════
def guardar(model, vec):
    print(f"\n💾 Guardando modelo en: {MODEL_OUT}")
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim":        next(model.parameters()).shape[1],
    }, MODEL_OUT)
 
    print(f"💾 Guardando vocabulario en: {VOCAB_OUT}")
    with open(VOCAB_OUT, "w", encoding="utf-8") as f:
        # Convertir int64 de numpy a int nativo de Python
        vocab_serializable = {k: int(v) for k, v in vec.vocabulary_.items()}
        json.dump(vocab_serializable, f, ensure_ascii=False)
 
    print("✅ Archivos guardados correctamente.")
 
# ══════════════════════════════════════════════════════════
# 9. MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 50)
    print("  HardwareGuard — Re-entrenamiento del Modelo")
    print("=" * 50)
 
    textos, etiquetas = cargar_dataset()
    X, vec            = vectorizar(textos)
    model             = entrenar(X, np.array(etiquetas, dtype=np.float32))
    guardar(model, vec)
 
    print("\n🎉 ¡Entrenamiento completado!")
    print("   Reinicia predictor.py para usar el nuevo modelo.")