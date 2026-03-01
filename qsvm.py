import pennylane as qml
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import time

print("QUANTUM SVM WITH PENNYLANE")

# Load your data
X_train_scaled = np.load('X_train_scaled.npy')
X_test_scaled = np.load('X_test_scaled.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Reduce to 4 features
pca = PCA(n_components=4)
X_train = pca.fit_transform(X_train_scaled)[:200]  # Use 200 samples
X_test = pca.transform(X_test_scaled)[:50]
y_train = y_train[:200]
y_test = y_test[:50]

print(f"Training on {len(X_train)} samples")
print(f"Testing on {len(X_test)} samples")

# Define quantum device
dev = qml.device('default.qubit', wires=4)

@qml.qnode(dev)
def quantum_kernel(x1, x2):
    """Simple quantum kernel"""
    # Encode first point
    for i in range(4):
        qml.RY(x1[i], wires=i)
    
    # Encode second point
    for i in range(4):
        qml.RY(x2[i], wires=i)
    
    # Measure overlap (probability of |0...0> state)
    return qml.probs(wires=range(4))

print("Computing kernel matrix (this may take a minute)...")
start = time.time()

# Compute kernel matrix
K_train = np.zeros((len(X_train), len(X_train)))
for i in range(len(X_train)):
    for j in range(len(X_train)):
        K_train[i, j] = quantum_kernel(X_train[i], X_train[j])[0]

# Train SVM with precomputed kernel
svm = SVC(kernel='precomputed')
svm.fit(K_train, y_train)

# Compute test kernel matrix
K_test = np.zeros((len(X_test), len(X_train)))
for i in range(len(X_test)):
    for j in range(len(X_train)):
        K_test[i, j] = quantum_kernel(X_test[i], X_train[j])[0]

# Predict
y_pred = svm.predict(K_test)
accuracy = accuracy_score(y_test, y_pred)
elapsed = time.time() - start

print(f"✅ Quantum SVM Accuracy: {accuracy:.2%}")
print(f"⏱️  Time: {elapsed:.2f} seconds")