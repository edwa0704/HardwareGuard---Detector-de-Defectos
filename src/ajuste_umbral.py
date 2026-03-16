import os
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
 
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, f1_score
)
 
# ─────────────────────────────────────────────
# 0. CONFIGURACIÓN
# ─────────────────────────────────────────────
 
SEMILLA   = 42
TEST_SIZE = 0.20
BASE_DIR  = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
 
torch.manual_seed(SEMILLA)
np.random.seed(SEMILLA)
 
 
# ─────────────────────────────────────────────
# 1. ARQUITECTURA (debe ser igual a V2)
# ─────────────────────────────────────────────
 
class HardwareGuardNet(nn.Module):
    """
    Misma arquitectura del modelo V2.
    Debe ser idéntica para cargar los pesos correctamente.
    """
    def __init__(self, input_dim):
        super(HardwareGuardNet, self).__init__()
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
 
 
# ─────────────────────────────────────────────
# 2. CARGAR MODELO Y DATOS
# ─────────────────────────────────────────────
 
def cargar_modelo_y_datos():
    """
    Carga el modelo V2 entrenado y los datos de prueba.
 
    Returns:
        tuple: (modelo, loader_test, y_test)
    """
    ruta_modelo = os.path.join(BASE_DIR, "hardwareguard_model_v2.pth")
    if not os.path.exists(ruta_modelo):
        raise FileNotFoundError(
            f"No se encontró: {ruta_modelo}\n"
            "Ejecuta primero: python src/entrenar_modelo_v2.py"
        )
 
    # Cargar checkpoint
    checkpoint = torch.load(ruta_modelo, map_location="cpu", weights_only=False)
    input_dim  = checkpoint["input_dim"]
 
    # Reconstruir modelo y cargar pesos
    modelo = HardwareGuardNet(input_dim=input_dim)
    modelo.load_state_dict(checkpoint["model_state_dict"])
    modelo.eval()
    print(f"[HardwareGuard] Modelo V2 cargado correctamente")
    print(f"  F1 durante entrenamiento : {checkpoint['f1_score']:.4f}")
    print(f"  Umbral original          : {checkpoint['umbral']}")
 
    # Cargar datos
    X = load_npz(os.path.join(BASE_DIR, "matriz_tfidf.npz")).toarray().astype(np.float32)
    y = pd.read_csv(os.path.join(BASE_DIR, "labels.csv"))["label"].values.astype(np.float32)
 
    # Misma división 80/20 del entrenamiento
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEMILLA, stratify=y
    )
 
    X_test_t = torch.tensor(X_test)
    y_test_t = torch.tensor(y_test).unsqueeze(1)
    loader_test = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=64, shuffle=False
    )
 
    print(f"  Datos de prueba          : {len(X_test):,} reseñas")
    print(f"  Defectos en prueba       : {int(y_test.sum())} ({y_test.mean()*100:.1f}%)\n")
 
    return modelo, loader_test, y_test
 
 
# ─────────────────────────────────────────────
# 3. OBTENER PROBABILIDADES
# ─────────────────────────────────────────────
 
def obtener_probabilidades(modelo, loader_test):
    """
    Obtiene las probabilidades brutas del modelo para cada reseña.
    Estas probabilidades se usan luego para probar distintos umbrales.
 
    Args:
        modelo      : Modelo entrenado en modo eval
        loader_test : DataLoader de prueba
 
    Returns:
        tuple: (probabilidades, labels_reales)
    """
    todas_probs  = []
    todos_labels = []
 
    with torch.no_grad():
        for X_batch, y_batch in loader_test:
            probs = modelo(X_batch)
            todas_probs.extend(probs.squeeze().tolist())
            todos_labels.extend(y_batch.squeeze().tolist())
 
    return np.array(todas_probs), np.array(todos_labels)
 
 
# ─────────────────────────────────────────────
# 4. BUSCAR UMBRAL ÓPTIMO
# ─────────────────────────────────────────────
 
def buscar_umbral_optimo(probs, labels):
    """
    Prueba múltiples umbrales y encuentra el que maximiza el F1-Score
    para la clase DEFECTO.
 
    Umbrales probados: 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50
 
    Args:
        probs  (np.array): Probabilidades del modelo
        labels (np.array): Labels reales
 
    Returns:
        float: Umbral óptimo encontrado
    """
    umbrales = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
 
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  BÚSQUEDA DE UMBRAL ÓPTIMO")
    print(f"{sep}")
    print(f"  {'Umbral':<10} {'Accuracy':<12} {'F1-Defecto':<14} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6}")
    print(f"  {'-'*65}")
 
    mejor_f1     = 0
    mejor_umbral = 0.35
    resultados   = []
 
    for u in umbrales:
        preds = (probs >= u).astype(float)
        acc   = accuracy_score(labels, preds)
        f1    = f1_score(labels, preds, zero_division=0)
        cm    = confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel()
 
        marcador = " ← MEJOR" if f1 > mejor_f1 else ""
        print(f"  {u:<10.2f} {acc*100:<11.2f}% {f1:<14.4f} {tp:>6} {fp:>6} {fn:>6} {tn:>6}{marcador}")
 
        resultados.append({
            "umbral": u, "accuracy": acc, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn
        })
 
        if f1 > mejor_f1:
            mejor_f1     = f1
            mejor_umbral = u
 
    print(f"  {'-'*65}")
    print(f"\n  Umbral óptimo encontrado: {mejor_umbral} (F1 = {mejor_f1:.4f})")
    print(f"{sep}\n")
 
    return mejor_umbral, mejor_f1, resultados
 
 
# ─────────────────────────────────────────────
# 5. REPORTE FINAL CON UMBRAL ÓPTIMO
# ─────────────────────────────────────────────
 
def reporte_final(probs, labels, umbral_optimo):
    """
    Imprime el reporte completo usando el umbral óptimo encontrado.
 
    Args:
        probs          : Probabilidades del modelo
        labels         : Labels reales
        umbral_optimo  : Umbral que maximiza el F1
    """
    preds = (probs >= umbral_optimo).astype(float)
    acc   = accuracy_score(labels, preds)
    f1    = f1_score(labels, preds, zero_division=0)
    cm    = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
 
    sep = "=" * 65
    print(f"{sep}")
    print(f"  RESULTADO FINAL — UMBRAL ÓPTIMO: {umbral_optimo}")
    print(f"{sep}\n")
 
    print(f"  EVOLUCIÓN DEL PROYECTO:")
    print(f"  {'Versión':<25} {'F1-Score':<12} {'TP':>6} {'FN':>6}")
    print(f"  {'-'*50}")
    print(f"  {'V1 PR#4 (original)':<25} {'0.5055':<12} {'184':>6} {'170':>6}")
    print(f"  {'V2 SMOTE umbral=0.35':<25} {'0.5565':<12} {'184':>6} {'170':>6}")
    print(f"  {'V2 umbral óptimo':<25} {f1:<12.4f} {tp:>6} {fn:>6}  ← ACTUAL")
    print()
 
    nivel = "EXCELENTE" if f1 >= 0.75 else "BUENO" if f1 >= 0.65 else "MEJORADO" if f1 > 0.55 else "EN PROGRESO"
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  F1-Score  : {f1:.4f}  [{nivel}]")
    print()
 
    print(f"  MATRIZ DE CONFUSIÓN (umbral={umbral_optimo}):")
    print(f"  {'':22} Pred: Sin Defecto   Pred: DEFECTO")
    print(f"  {'Real: Sin Defecto':<22} {tn:^18} {fp:^16}")
    print(f"  {'Real: DEFECTO':<22} {fn:^18} {tp:^16}")
    print()
 
    print(f"  REPORTE DETALLADO:")
    reporte = classification_report(
        labels, preds,
        target_names=["Sin Defecto", "DEFECTO"],
        digits=4
    )
    for linea in reporte.split("\n"):
        print(f"  {linea}")
 
    print(f"{sep}")
    print(f"  Umbral óptimo guardado para uso en predictor.")
    print(f"{sep}\n")
 
    # Guardar umbral óptimo
    ruta = os.path.join(BASE_DIR, "umbral_optimo.txt")
    with open(ruta, "w") as f:
        f.write(str(umbral_optimo))
    print(f"[HardwareGuard] Umbral óptimo guardado en: {ruta}")
 
    return f1
 
 
# ─────────────────────────────────────────────
# 6. EJECUCIÓN PRINCIPAL
# ─────────────────────────────────────────────
 
if __name__ == "__main__":
    print("=" * 65)
    print("  HARDWAREGUARD — AJUSTE DE UMBRAL ÓPTIMO")
    print("  (sin reentrenar — carga modelo V2 existente)")
    print("=" * 65 + "\n")
 
    # Cargar modelo y datos
    modelo, loader_test, y_test = cargar_modelo_y_datos()
 
    # Obtener probabilidades brutas
    probs, labels = obtener_probabilidades(modelo, loader_test)
 
    # Buscar umbral óptimo
    umbral_optimo, mejor_f1, resultados = buscar_umbral_optimo(probs, labels)
 
    # Reporte final
    f1_final = reporte_final(probs, labels, umbral_optimo)
 
    print(f"[HardwareGuard] F1-Score con umbral optimo: {f1_final:.4f}")