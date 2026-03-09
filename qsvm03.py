import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# For quantum encoding and circuit
import pennylane as qml
import numpy as np

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
            df = pd.read_csv(file, nrows=1000)

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

dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def feature_map(x):

    for i in range(n_qubits):
        qml.Hadamard(wires=i)
        qml.RZ(x[i], wires=i)

    for i in range(n_qubits-1):
        qml.CNOT(wires=[i, i+1])

    for i in range(n_qubits):
        qml.RY(x[i], wires=i)

    return qml.state()

# fidelity kernel
def quantum_kernel(x1, x2):

    state1 = feature_map(x1)
    state2 = feature_map(x2)

    return np.abs(np.vdot(state1, state2))**2

def compute_embeddings(X):

    embeddings = []

    for i, x in enumerate(X):
        if i % 200 == 0:
            print("Encoding", i)

        embeddings.append(feature_map(x))

    return np.array(embeddings)

def compute_kernel_matrix(A, B):

    K = np.zeros((len(A), len(B)))

    for i in range(len(A)):
        for j in range(len(B)):

            K[i,j] = np.abs(np.vdot(A[i], B[j]))**2

    return K

sample = 2000

# Train the QSVM
def train_qsvm(X_train, X_test, y_train):

    print("Computing quantum embeddings")

    train_embed = compute_embeddings(X_train)
    test_embed = compute_embeddings(X_test)

    print("Building kernel matrices")

    K_train = compute_kernel_matrix(train_embed, train_embed)
    print("Done training. Start testing...")

    K_test = compute_kernel_matrix(test_embed, train_embed)

    svm = SVC(kernel="precomputed", C=10)

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

evaluate(model, K_test, y_test)