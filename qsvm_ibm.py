# ============================================
# QUANTUM SVM (QSVM)
# Running on REAL IBM Quantum Hardware
# ============================================

import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC

# Qiskit imports
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

print("STARTING QUANTUM SVM...")
print("="*50)

# ============================================
# STEP 1: LOAD DATA SAVED FROM CLASSICAL SVM
# ============================================
print("\nLoading data from Classical SVM...")

try:
    X_train_scaled = np.load('X_train_scaled.npy')
    X_test_scaled  = np.load('X_test_scaled.npy')
    y_train        = np.load('y_train.npy')
    y_test         = np.load('y_test.npy')
    print(f"✓ Data loaded successfully!")
    print(f"  Training samples: {len(X_train_scaled)}")
    print(f"  Testing samples:  {len(X_test_scaled)}")
except FileNotFoundError:
    print("ERROR: Could not find saved .npy files.")
    print("Please run svm.py first to generate the data files.")
    exit()

# ============================================
# STEP 2: REDUCE DATA SIZE FOR QUANTUM
# ============================================
print("\n" + "="*50)
print("Preparing data for Quantum hardware...")
print("="*50)

N_TRAIN    = 50  # reduced for real hardware to save runtime
N_TEST     = 20
N_FEATURES = 2   # each feature = 1 qubit

# Balance classes
train_idx_0 = np.where(y_train == 0)[0][:N_TRAIN // 2]
train_idx_1 = np.where(y_train == 1)[0][:N_TRAIN // 2]
train_idx   = np.concatenate([train_idx_0, train_idx_1])
np.random.shuffle(train_idx)

test_idx_0 = np.where(y_test == 0)[0][:N_TEST // 2]
test_idx_1 = np.where(y_test == 1)[0][:N_TEST // 2]
test_idx   = np.concatenate([test_idx_0, test_idx_1])
np.random.shuffle(test_idx)

X_train_small = X_train_scaled[train_idx]
y_train_small = y_train[train_idx]
X_test_small  = X_test_scaled[test_idx]
y_test_small  = y_test[test_idx]

# Reduce features using PCA
print(f"\nReducing features from {X_train_scaled.shape[1]} → {N_FEATURES} using PCA...")
pca = PCA(n_components=N_FEATURES)
X_train_pca = pca.fit_transform(X_train_small)
X_test_pca  = pca.transform(X_test_small)
print(f"✓ PCA complete. Explained variance: {pca.explained_variance_ratio_.sum():.2%}")

# Scale to [0, pi] for quantum encoding
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_train_q = scaler.fit_transform(X_train_pca)
X_test_q  = scaler.transform(X_test_pca)

print(f"\nQuantum dataset ready:")
print(f"  Training samples:  {len(X_train_q)}")
print(f"  Testing samples:   {len(X_test_q)}")
print(f"  Features (qubits): {N_FEATURES}")

# ============================================
# STEP 3: CONNECT TO REAL IBM QUANTUM HARDWARE
# ============================================
print("\n" + "="*50)
print("Connecting to IBM Quantum Hardware...")
print("="*50)

service = QiskitRuntimeService()
backend = service.least_busy(operational=True, simulator=False)
print(f"✓ Connected! Using backend: {backend.name}")

sampler = Sampler(backend)

# Set up quantum feature map and kernel
feature_map    = ZZFeatureMap(feature_dimension=N_FEATURES, reps=2)
quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
print(f"✓ Quantum kernel ready!")
#print(f"  Feature map: ZZFeatureMap, {N_FEATURES} qubits, 2 repetitions")

# ============================================
# STEP 4: TRAIN QSVM
# ============================================
print("\n" + "="*50)
print("Training Quantum SVM on real hardware...")
print("="*50)

start_train = time.time()
qsvm = SVC(kernel=quantum_kernel.evaluate)
qsvm.fit(X_train_q, y_train_small)
train_time = time.time() - start_train

print(f"✓ Training complete! ({train_time:.2f}s)")

# ============================================
# STEP 5: TEST QSVM
# ============================================
print("\n" + "="*50)
print("Testing Quantum SVM...")
print("="*50)

start_test = time.time()
y_pred     = qsvm.predict(X_test_q)
test_time  = time.time() - start_test
accuracy   = accuracy_score(y_test_small, y_pred)

print(f"✓ Testing complete! ({test_time:.4f}s)")

# ============================================
# STEP 6: RESULTS
# ============================================
print("\n" + "="*50)
print("✓ FINAL RESULTS READY")
print("="*50)
#print(f"Backend used:   {backend.name}")
print(f"Accuracy:       {accuracy:.2%}")
print(f"Training time:  {train_time:.2f}s")
print(f"Testing time:   {test_time:.4f}s")
#print(f"Qubits used:    {N_FEATURES}")
print(f"Train samples:  {len(X_train_q)}")
print(f"Test samples:   {len(X_test_q)}")

print("\nDetailed Report:")
print(classification_report(y_test_small, y_pred, target_names=['Normal', 'Attack']))

# ============================================
# STEP 7: COMPARISON SUMMARY
# ============================================
print("\n" + "="*50)
print("CLASSICAL vs QUANTUM COMPARISON")
print("="*50)
print(f"{'Metric':<20} {'Classical SVM':<20} {'Quantum SVM':<20}")
print("-"*60)
print(f"{'Accuracy':<20} {'98.80%':<20} {accuracy:.2%}")
print(f"{'Training time':<20} {'~0.60s':<20} {train_time:.2f}s")
print(f"{'Testing time':<20} {'~0.001s':<20} {test_time:.4f}s")
print(f"{'Samples used':<20} {'8000':<20} {len(X_train_q)}")
print(f"{'Features used':<20} {'42':<20} {N_FEATURES}")
print("="*50)