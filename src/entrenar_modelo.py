import os
import json
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
 
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, f1_score
)
 
# ─────────────────────────────────────────────
# 0. CONFIGURACIÓN
# ─────────────────────────────────────────────
 
# Semilla para reproducibilidad
SEMILLA = 42
torch.manual_seed(SEMILLA)
np.random.seed(SEMILLA)
 
# Hiperparámetros del entrenamiento
EPOCHS      = 20       # Número de pasadas completas por los datos
BATCH_SIZE  = 64       # Reseñas procesadas por cada paso
LR          = 0.001    # Learning rate — qué tan rápido aprende la red
TEST_SIZE   = 0.20     # 20% para prueba, 80% para entrenamiento
 
 
# ─────────────────────────────────────────────
# 1. CARGAR DATOS
# ─────────────────────────────────────────────
 
def cargar_datos():
    """
    Carga la matriz TF-IDF y los labels generados en el PR #3.
 
    Returns:
        tuple: (X, y)
            X : Matriz TF-IDF como array numpy (5000 × 231)
            y : Vector de labels binarios (0 o 1)
 
    Raises:
        FileNotFoundError: Si no existen los archivos del PR #3
    """
    base_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
 
    ruta_matriz = os.path.join(base_dir, "matriz_tfidf.npz")
    ruta_labels = os.path.join(base_dir, "labels.csv")
 
    for ruta in [ruta_matriz, ruta_labels]:
        if not os.path.exists(ruta):
            raise FileNotFoundError(
                f"No se encontró: {ruta}\n"
                "Ejecuta primero: python src/vectorizacion.py"
            )
 
    # Cargar matriz TF-IDF (formato sparse → dense)
    X = load_npz(ruta_matriz).toarray().astype(np.float32)
 
    # Cargar labels
    df_labels = pd.read_csv(ruta_labels)
    y = df_labels["label"].values.astype(np.float32)
 
    print(f"[HardwareGuard] Datos cargados:")
    print(f"  Matriz X  : {X.shape[0]:,} reseñas × {X.shape[1]} features")
    print(f"  Labels y  : {len(y):,} etiquetas")
    print(f"  Defectos  : {int(y.sum()):,} ({y.mean()*100:.1f}%)")
    print(f"  Sin defecto: {int((y==0).sum()):,} ({(y==0).mean()*100:.1f}%)")
 
    return X, y
 
 
# ─────────────────────────────────────────────
# 2. DIVIDIR DATOS 80/20
# ─────────────────────────────────────────────
 
def dividir_datos(X, y):
    """
    Divide los datos en 80% entrenamiento y 20% prueba.
    Usa estratificación para mantener la proporción de clases.
 
    Args:
        X (np.array): Matriz de features TF-IDF
        y (np.array): Vector de labels
 
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=SEMILLA,
        stratify=y          # Mantiene proporción 0/1 en ambos conjuntos
    )
 
    sep = "-" * 50
    print(f"\n[HardwareGuard] División 80/20:")
    print(f"  {sep}")
    print(f"  Entrenamiento : {len(X_train):,} reseñas (80%)")
    print(f"    → Defectos  : {int(y_train.sum()):,} ({y_train.mean()*100:.1f}%)")
    print(f"  Prueba        : {len(X_test):,} reseñas (20%)")
    print(f"    → Defectos  : {int(y_test.sum()):,} ({y_test.mean()*100:.1f}%)")
    print(f"  {sep}")
 
    return X_train, X_test, y_train, y_test
 
 
def crear_dataloaders(X_train, X_test, y_train, y_test):
    """
    Convierte los arrays numpy en tensores PyTorch y crea DataLoaders.
    Los DataLoaders permiten iterar el dataset en mini-batches.
 
    Args:
        X_train, X_test : Arrays numpy de features
        y_train, y_test : Arrays numpy de labels
 
    Returns:
        tuple: (loader_train, loader_test)
    """
    # Convertir a tensores PyTorch
    X_train_t = torch.tensor(X_train)
    X_test_t  = torch.tensor(X_test)
    y_train_t = torch.tensor(y_train).unsqueeze(1)  # Shape: [N, 1]
    y_test_t  = torch.tensor(y_test).unsqueeze(1)
 
    # Crear datasets
    dataset_train = TensorDataset(X_train_t, y_train_t)
    dataset_test  = TensorDataset(X_test_t, y_test_t)
 
    # Crear DataLoaders
    loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    loader_test  = DataLoader(dataset_test,  batch_size=BATCH_SIZE, shuffle=False)
 
    return loader_train, loader_test
 
 
# ─────────────────────────────────────────────
# 3. ARQUITECTURA DE LA RED NEURONAL
# ─────────────────────────────────────────────
 
class HardwareGuardNet(nn.Module):
    """
    Red Neuronal Clasificadora para detección de defectos de hardware.
 
    Arquitectura:
        - Capa de entrada : input_dim neuronas (tamaño del vocabulario TF-IDF)
        - Capa oculta 1   : 128 neuronas + ReLU + Dropout(0.3)
        - Capa oculta 2   : 64 neuronas  + ReLU + Dropout(0.3)
        - Capa de salida  : 1 neurona + Sigmoid → probabilidad [0,1]
 
    ¿Por qué estas capas?
        - ReLU     : Introduce no-linealidad (sin ella la red no aprendería patrones complejos)
        - Dropout  : Durante entrenamiento "apaga" 30% de neuronas al azar → evita memorizar datos
        - Sigmoid  : Convierte el valor final a probabilidad entre 0 y 1
 
    Args:
        input_dim (int): Número de features (columnas de la matriz TF-IDF)
    """
 
    def __init__(self, input_dim):
        super(HardwareGuardNet, self).__init__()
 
        self.red = nn.Sequential(
            # Capa 1: input_dim → 128
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
 
            # Capa 2: 128 → 64
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
 
            # Capa de salida: 64 → 1
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        """
        Propagación hacia adelante (Forward Pass).
        Recibe un batch de vectores TF-IDF y retorna probabilidades.
 
        Args:
            x (Tensor): Batch de vectores TF-IDF [batch_size × input_dim]
 
        Returns:
            Tensor: Probabilidades de defecto [batch_size × 1]
        """
        return self.red(x)
 
 
# ─────────────────────────────────────────────
# 4. ENTRENAMIENTO
# ─────────────────────────────────────────────
 
def entrenar_modelo(modelo, loader_train, loader_test):
    """
    Entrena la red neuronal durante N epochs.
 
    Proceso por cada epoch:
        1. Por cada batch: forward pass → calcular pérdida → backprop → actualizar pesos
        2. Evaluar accuracy en datos de prueba
        3. Imprimir progreso
 
    Función de pérdida: BCELoss (Binary Cross Entropy)
        Mide qué tan lejos está la predicción de la realidad.
        Si predijo 0.9 pero era 0 → pérdida alta
        Si predijo 0.9 y era 1   → pérdida baja
 
    Optimizador: Adam
        Algoritmo que ajusta los pesos de la red para minimizar la pérdida.
 
    Args:
        modelo      : Red neuronal HardwareGuardNet
        loader_train: DataLoader con datos de entrenamiento
        loader_test : DataLoader con datos de prueba
 
    Returns:
        tuple: (historial_perdida, historial_accuracy)
    """
    criterio   = nn.BCELoss()                          # Función de pérdida binaria
    optimizador = optim.Adam(modelo.parameters(), lr=LR)  # Optimizador Adam
 
    historial_perdida   = []
    historial_accuracy  = []
 
    print(f"\n[HardwareGuard] Iniciando entrenamiento...")
    print(f"  Epochs     : {EPOCHS}")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  Learning rate: {LR}")
    print(f"\n  {'Epoch':<8} {'Pérdida':<12} {'Accuracy Test':<15} {'Estado'}")
    print(f"  {'-'*50}")
 
    for epoch in range(1, EPOCHS + 1):
        # ── Fase de entrenamiento ────────────────
        modelo.train()
        perdida_epoch = 0.0
 
        for X_batch, y_batch in loader_train:
            # Forward pass: calcular predicciones
            predicciones = modelo(X_batch)
 
            # Calcular pérdida
            perdida = criterio(predicciones, y_batch)
 
            # Backward pass: calcular gradientes
            optimizador.zero_grad()
            perdida.backward()
 
            # Actualizar pesos
            optimizador.step()
 
            perdida_epoch += perdida.item()
 
        perdida_promedio = perdida_epoch / len(loader_train)
 
        # ── Fase de evaluación ───────────────────
        modelo.eval()
        todas_preds = []
        todos_labels = []
 
        with torch.no_grad():
            for X_batch, y_batch in loader_test:
                probs = modelo(X_batch)
                preds = (probs >= 0.5).float()
                todas_preds.extend(preds.squeeze().tolist())
                todos_labels.extend(y_batch.squeeze().tolist())
 
        accuracy = accuracy_score(todos_labels, todas_preds)
        historial_perdida.append(perdida_promedio)
        historial_accuracy.append(accuracy)
 
        # Indicador visual de progreso
        estado = "🔥" if accuracy >= 0.80 else "📈" if accuracy >= 0.70 else "⏳"
        print(f"  {epoch:<8} {perdida_promedio:<12.4f} {accuracy*100:<14.2f}% {estado}")
 
    print(f"  {'-'*50}")
    print(f"[HardwareGuard] Entrenamiento completado ✅\n")
 
    return historial_perdida, historial_accuracy
 
 
# ─────────────────────────────────────────────
# 5. EVALUACIÓN Y MATRIZ DE CONFUSIÓN
# ─────────────────────────────────────────────
 
def evaluar_modelo(modelo, loader_test):
    """
    Evalúa el modelo en datos de prueba e imprime métricas detalladas.
 
    Métricas:
        - Accuracy   : (TP + TN) / Total
        - Precisión  : TP / (TP + FP) — de los que alertamos, cuántos eran reales
        - Recall     : TP / (TP + FN) — de todos los defectos, cuántos detectamos
        - F1-Score   : Media armónica de Precisión y Recall
 
    Matriz de Confusión:
        Filas    = Valores reales
        Columnas = Predicciones del modelo
 
              Pred 0   Pred 1
        Real 0  [TN]    [FP]    ← Sin defecto
        Real 1  [FN]    [TP]    ← Con defecto
 
    Args:
        modelo      : Red neuronal entrenada
        loader_test : DataLoader con datos de prueba
    """
    modelo.eval()
    todas_preds  = []
    todos_labels = []
    todas_probs  = []
 
    with torch.no_grad():
        for X_batch, y_batch in loader_test:
            probs = modelo(X_batch)
            preds = (probs >= 0.5).float()
            todas_preds.extend(preds.squeeze().tolist())
            todos_labels.extend(y_batch.squeeze().tolist())
            todas_probs.extend(probs.squeeze().tolist())
 
    # Calcular métricas
    acc = accuracy_score(todos_labels, todas_preds)
    f1  = f1_score(todos_labels, todas_preds)
    cm  = confusion_matrix(todos_labels, todas_preds)
    tn, fp, fn, tp = cm.ravel()
 
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  🧠 HARDWAREGUARD — REPORTE DE EVALUACIÓN DEL MODELO")
    print(f"{sep}\n")
 
    # ── Accuracy general ─────────────────────
    nivel = "EXCELENTE 🏆" if acc >= 0.85 else "BUENO ✅" if acc >= 0.75 else "MEJORABLE 📈"
    print(f"  Accuracy global : {acc*100:.2f}%  [{nivel}]")
    print(f"  F1-Score        : {f1:.4f}")
    print()
 
    # ── Matriz de confusión ──────────────────
    print(f"  MATRIZ DE CONFUSIÓN:")
    print(f"  {'':20} Pred: Sin Defecto   Pred: DEFECTO")
    print(f"  {'Real: Sin Defecto':<20} {tn:^18} {fp:^16}")
    print(f"  {'Real: DEFECTO':<20} {fn:^18} {tp:^16}")
    print()
 
    # ── Interpretación ───────────────────────
    print(f"  INTERPRETACIÓN:")
    print(f"  ✅ Verdaderos Negativos (TN): {tn:,}  — Sin defecto, clasificado correcto")
    print(f"  ✅ Verdaderos Positivos (TP): {tp:,}  — Defecto detectado correctamente")
    print(f"  ⚠️  Falsos Positivos    (FP): {fp:,}  — Falsa alarma (sin defecto pero dijo sí)")
    print(f"  🚨 Falsos Negativos    (FN): {fn:,}  — DEFECTO NO DETECTADO (peligroso)")
    print()
 
    # ── Reporte completo sklearn ─────────────
    print(f"  REPORTE DETALLADO:")
    reporte = classification_report(
        todos_labels, todas_preds,
        target_names=["Sin Defecto", "DEFECTO"],
        digits=4
    )
    for linea in reporte.split("\n"):
        print(f"  {linea}")
 
    print(f"{sep}")
    print(f"  ✅ Modelo evaluado exitosamente")
    print(f"  📌 Falsos Negativos = {fn} → defectos que pasaron desapercibidos")
    print(f"     Objetivo: minimizar FN para no perder lotes defectuosos.")
    print(f"{sep}\n")
 
    return acc, cm
 
 
# ─────────────────────────────────────────────
# 6. GUARDAR MODELO
# ─────────────────────────────────────────────
 
def guardar_modelo(modelo, input_dim, acc):
    """
    Guarda los pesos del modelo entrenado en disco.
    Permite reutilizar el modelo sin reentrenar.
 
    Args:
        modelo    : Red neuronal entrenada
        input_dim : Dimensión de entrada del modelo
        acc       : Accuracy obtenida (para el nombre del archivo)
    """
    base_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
    ruta_modelo = os.path.join(base_dir, "hardwareguard_model.pth")
 
    torch.save({
        "model_state_dict" : modelo.state_dict(),
        "input_dim"        : input_dim,
        "accuracy"         : acc,
        "arquitectura"     : "Linear(input→128) ReLU Dropout Linear(128→64) ReLU Dropout Linear(64→1) Sigmoid",
    }, ruta_modelo)
 
    print(f"[HardwareGuard] 💾 Modelo guardado en: {ruta_modelo}")
 
 
# ─────────────────────────────────────────────
# 7. EJECUCIÓN PRINCIPAL
# ─────────────────────────────────────────────
 
if __name__ == "__main__":
    print("=" * 65)
    print("  🛡️  HARDWAREGUARD — ENTRENAMIENTO DEL MODELO NLP")
    print("=" * 65)
 
    # Cargar datos
    X, y = cargar_datos()
    input_dim = X.shape[1]
 
    # Dividir 80/20
    X_train, X_test, y_train, y_test = dividir_datos(X, y)
 
    # Crear DataLoaders
    loader_train, loader_test = crear_dataloaders(X_train, X_test, y_train, y_test)
 
    # Construir red neuronal
    modelo = HardwareGuardNet(input_dim=input_dim)
    print(f"\n[HardwareGuard] Arquitectura de la red:")
    print(f"  Entrada  : {input_dim} neuronas (features TF-IDF)")
    print(f"  Oculta 1 : 128 neuronas + ReLU + Dropout(0.3)")
    print(f"  Oculta 2 : 64 neuronas  + ReLU + Dropout(0.3)")
    print(f"  Salida   : 1 neurona + Sigmoid")
    total_params = sum(p.numel() for p in modelo.parameters())
    print(f"  Total parámetros entrenables: {total_params:,}")
 
    # Entrenar modelo
    historial_perdida, historial_accuracy = entrenar_modelo(modelo, loader_train, loader_test)
 
    # Evaluar modelo
    acc, cm = evaluar_modelo(modelo, loader_test)
 
    # Guardar modelo
    guardar_modelo(modelo, input_dim, acc)
 
    print(f"[HardwareGuard] 🎯 PR #4 completado — Accuracy final: {acc*100:.2f}%")