import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# For quantum encoding and circuit
import pennylane as qml
import numpy as np
import time
# For training the QSVM
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def find_csv_files(base_dir="."):
    csv_files = []

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(".csv"):
                path = os.path.join(root, file)
                csv_files.append(path)

    return csv_files

# Load separate files

def load_attack_dataset():

    csv_files = find_csv_files()

    normal_dfs = []
    attack_dfs = []

    for file in csv_files:

        try:
            df = pd.read_csv(file, nrows=500)

        except Exception:
            continue

        name = file.lower()

        if "normal" in name or "sensor" in name:
            df["Attack_type"] = "Normal"
            normal_dfs.append(df)

        elif "attack" in name:
            df["Attack_type"] = "Attack"
            attack_dfs.append(df)

    if len(normal_dfs) == 0 or len(attack_dfs) == 0:
        raise RuntimeError("Dataset must contain both Normal and Attack files")

    df_normal = pd.concat(normal_dfs, ignore_index=True)
    df_attack = pd.concat(attack_dfs, ignore_index=True)

    df = pd.concat([df_normal, df_attack], ignore_index=True)

    df["is_attack"] = (df["Attack_type"] != "Normal").astype(int)

    return df

# feature engineering for improved QSVM accuracy
def prepare_features(df):

    numerical_cols = df.select_dtypes(include="number").columns.tolist()
    numerical_cols.remove("is_attack")

    X = df[numerical_cols].fillna(0)
    y = df["is_attack"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # QSVM works best with small dimension
    from sklearn.decomposition import PCA
    pca = PCA(n_components=4)

    X_reduced = pca.fit_transform(X_scaled)

    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    return X_train, X_test, y_train.values, y_test.values

# Quantum Feature Map

n_qubits = 4

dev = qml.device("lightning.qubit", wires=n_qubits)

@qml.qnode(dev)
def feature_map(x):

    qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")

    qml.BasicEntanglerLayers(
        weights=np.ones((1, n_qubits)),
        wires=range(n_qubits)
    )

    return qml.state()

# kernel approximation
def nystrom_kernel(A, m=500):

    N = len(A)

    idx = np.random.choice(N, m, replace=False)

    landmarks = A[idx]

    # compute C
    C = np.abs(A @ landmarks.conj().T)**2

    # compute W
    W = np.abs(landmarks @ landmarks.conj().T)**2

    W_inv = np.linalg.pinv(W)

    K_approx = C @ W_inv @ C.T

    return K_approx

def compute_embeddings(X):

    print("Encoding quantum states")

    embeddings = np.empty((len(X), 2**n_qubits), dtype=np.complex128) # preallocated memory

    for i in range(len(X)):
        embeddings[i] = feature_map(X[i])

    return embeddings

def compute_kernel_matrix(A, B):

   return np.abs(A @ B.conj().T)**2

sample = 10000

# Train the QSVM
def train_qsvm(X_train, X_test, y_train):

    print("Computing quantum embeddings")

    embeddings = compute_embeddings(np.vstack([X_train[:sample], X_test[:sample]]))

    train_embed = embeddings[:len(X_train[:sample])]
    test_embed = embeddings[len(X_train[:sample]):len(X_train[:sample])+sample]

    print("Using Nyström kernel approximation")
    
    start_qsvm_train = time.time()
    K_train = nystrom_kernel(train_embed[:sample], m=100)
    end_qsvm_train = time.time()

    print(f"Quantum SVM training complete in {end_qsvm_train - start_qsvm_train:.2f} seconds.")

    # approximate test kernel
    start_qsvm_test = time.time()
    K_test = np.abs(test_embed @ train_embed.conj().T)**2
    end_qsvm_test = time.time()
    print(f"Quantum SVM testing complete in {end_qsvm_test - start_qsvm_test:.2f} seconds.")

    svm = SVC(kernel="precomputed", C=12)

    svm.fit(K_train, y_train)

    return svm, K_test

def evaluate(model, K_test, y_test):

    y_pred = model.predict(K_test)

    acc = accuracy_score(y_test[:sample], y_pred[:sample])

    print("Accuracy:", acc)

    print(classification_report(
        y_test[:sample], y_pred[:sample],
        target_names=["Normal","Attack"]
    ))

# Run
df = load_attack_dataset()

X_train, X_test, y_train, y_test = prepare_features(df)


model, K_test = train_qsvm(X_train[:sample], X_test[:sample], y_train[:sample])

evaluate(model, K_test[:sample], y_test[:sample])