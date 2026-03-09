import numpy as np
import pandas as pd

import numpy as np
import pandas as pd # Import pandas as it's used for DataFrame operations
# ============================================
# CLASSICAL SVM (Data Preparation Only)
# ============================================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# The following imports are commented out as they are for classical SVM training
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, classification_report
import time
import os
import glob
import warnings
warnings.filterwarnings('ignore')

print("STARTING CLASSICAL SVM DATA PREPARATION...")
print("="*50)

# ============================================
# STEP 1: FIND ALL CSV FILES
# ============================================
all_csv = []
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.csv'):
            all_csv.append(os.path.join(root, file))

# ============================================
# STEP 2: LOOK FOR COMBINED DATASET FIRST (EASIEST)
# ============================================
combined_file = None
for file in all_csv:
    if 'DNN' in file or 'ML' in file or 'Selected' in file:
        try:
            with open(file, 'r') as f:
                if 'Attack_type' in f.readline():
                    combined_file = file
                    break
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue


df = None # Initialize df outside conditions

if combined_file:
    df = pd.read_csv(combined_file, nrows=10000)

    if 'Attack_type' in df.columns:
        classes = df['Attack_type'].unique()
        if 'Normal' in classes and len(classes) > 1:
            df['is_attack'] = (df['Attack_type'] != 'Normal').astype(int)
        else:
            combined_file = None # Invalidate if not suitable

# ============================================
# STEP 3: IF NO COMBINED FILE, USE SEPARATE FILES 
# ============================================
if not combined_file and df is None:
    normal_files = []
    for file in all_csv:
        if 'Normal' in file or 'sensor' in file.lower():
            normal_files.append(file)

    attack_files = []
    for file in all_csv:
        if 'attack' in file.lower() and 'Normal' not in file:
            attack_files.append(file)

    if normal_files and attack_files:
        normal_file = normal_files[0]
        attack_file = attack_files[0]

        df_normal = pd.read_csv(normal_file, nrows=5000)
        df_attack = pd.read_csv(attack_file, nrows=5000)

        df_normal['Attack_type'] = 'Normal'
        df_attack['Attack_type'] = 'Attack'

        df = pd.concat([df_normal, df_attack], ignore_index=True)
    
# ============================================
# STEP 4: PREPARE DATA
# ============================================
if df is None: # Exit if no data could be loaded or generated
    print("Could not load or generate any data. Exiting.")
    exit()

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

if 'is_attack' not in df.columns:
    df['is_attack'] = (df['Attack_type'] != 'Normal').astype(int)

normal_count = sum(df['is_attack'] == 0)
attack_count = sum(df['is_attack'] == 1)
print(f"\nNormal samples: {normal_count}")
print(f"Attack samples: {attack_count}")

if normal_count == 0 or attack_count == 0:
    print("Need both classes! Generating synthetic data ensures this, but if loading real data, check your files.")

# Select numerical features
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numerical_cols = [c for c in numerical_cols if c not in ['is_attack']]

# ============================================
# DROP HIGHLY CORRELATED (LEAKY) FEATURES
# ============================================
if not numerical_cols: # Ensure there are numerical columns before proceeding
    print("No numerical features found after initial selection. Exiting.")
    exit()

X_full = df[numerical_cols].fillna(0)

# Only attempt correlation if there's enough data and features to correlate
if len(X_full) > 1 and len(numerical_cols) > 0:
    try:
        correlations = X_full.corrwith(df['is_attack']).abs()
        leaky_features = correlations[correlations > 0.95].index.tolist()
        if leaky_features:
            numerical_cols = [c for c in numerical_cols if c not in leaky_features]
    except Exception as e:
        print(f"Warning: Could not calculate correlations to drop leaky features: {e}")
        print("Proceeding without dropping highly correlated features.")

X = df[numerical_cols].fillna(0)
y = df['is_attack']

print(f"\nUsing {len(numerical_cols)} numerical features")
print(f"Sample features: {numerical_cols[:5]}")

# Check if training set contains both Normal and Attack labels
print("Testtttt")
unique, counts = np.unique(y, return_counts=True)
if len(unique) < 2:
    raise RuntimeError(
        f"Training labels contain a single class: {dict(zip(unique, counts))}."
        "Make sure your data includes both Normal and Attack samples."
    )
    exit()

# ============================================
# STEP 5: SPLIT AND SCALE
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining: {len(X_train)} samples")
print(f"Testing: {len(X_test)} samples")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# STEP 6: SAVE FOR QUANTUM
# ============================================
np.save('X_train_scaled.npy', X_train_scaled)
np.save('X_test_scaled.npy', X_test_scaled)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

print("\nData preparation complete and saved to .npy files.")
print("="*50)

"""## Quantum SVM (QSVM) Training and Evaluation

Now, let's use the prepared data to train and evaluate a Quantum SVM. We will use the `quantum_kernel_element` function defined earlier to construct the kernel matrices for training and testing.
"""

# ============================================
# STEP 1: LOAD PREPARED DATA
# ============================================
print("Loading prepared data...")
X_train_scaled = np.load('X_train_scaled.npy')
X_test_scaled = np.load('X_test_scaled.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test_scaled shape: {X_test_scaled.shape}")
print(f"y_test shape: {y_test.shape}")

import pennylane as qml # Added pennylane import
import math # Added math import

dev = qml.device("default.mixed", wires=4) # Define the device

@qml.qnode(dev) #Define a QNode with measurements
def circuit(phi):
  for j in range(4):
    qml.RY(math.pi*phi[j], wires = j) #encoding data

    # Q Circuit - Quantum convolution
    qml.CRot(math.pi/4, 0, math.pi/4, wires=[0,1])
    qml.CRot(math.pi/4, 0, math.pi/4, wires=[1,2]) # control q and target qubit
    qml.CRot(math.pi/4, 0, math.pi/4, wires=[2,3])

    # Measurement = "updated features"
    return [qml.expval(qml.PauliZ(j)) for j in range(4)]

def quantum_kernel_element(x1, x2):
    # Run the circuit for each data point to get their feature vectors
    f_x1 = circuit(x1)
    f_x2 = circuit(x2)

    # Convert lists to numpy arrays for dot product
    f_x1_np = np.array(f_x1)
    f_x2_np = np.array(f_x2)

    # Calculate the dot product of the feature vectors
    return np.dot(f_x1_np, f_x2_np)

print("Quantum kernel element function and circuit defined.")

# ============================================
# STEP 1B: ENSURE BOTH CLASSES IN TRAIN SUBSET
# ============================================

subset_size = 65

# Find indices of each class
normal_idx = np.where(y_train == 0)[0]
attack_idx = np.where(y_train == 1)[0]

# Ensure we have samples of each
if len(normal_idx) == 0 or len(attack_idx) == 0:
    raise RuntimeError("Training set does not contain both classes.")

# Take half from each class
half = subset_size // 2

selected_normal = normal_idx[:half]
selected_attack = attack_idx[:half]

selected_idx = np.concatenate([selected_normal, selected_attack])
print("Selected idx")
print(selected_idx)

# Shuffle indices
np.random.shuffle(selected_idx)

X_train_subset = X_train_scaled[selected_idx]
y_train_subset = y_train[selected_idx]

print("Subset class distribution:", np.unique(y_train_subset, return_counts=True))

# ============================================
# STEP 2: DEFINE KERNEL MATRIX COMPUTATION FUNCTION
# ============================================
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC

def compute_kernel_matrix(data1, data2):
    num_samples1 = data1.shape[0]
    num_samples2 = data2.shape[0]
    kernel_matrix = np.zeros((num_samples1, num_samples2))
    for i in range(num_samples1):
        if i % 100 == 0:
                print(f"Computing kernel row {i}/{num_samples1}")
        for j in range(num_samples2):
            kernel_matrix[i, j] = quantum_kernel_element(data1[i], data2[j])
        print(f"Done with {i}")
    return kernel_matrix

print("Kernel matrix computation function defined.")

# ============================================
# STEP 3: COMPUTE QUANTUM KERNEL MATRICES
# ============================================
print("Computing quantum kernel matrices...")

start_kernel_train = time.time()
quantum_kernel_train = compute_kernel_matrix(X_train_subset, X_train_subset)
end_kernel_train = time.time()
print(f"Training kernel matrix computed in {end_kernel_train - start_kernel_train:.2f} seconds.")

start_kernel_test = time.time()
quantum_kernel_test = compute_kernel_matrix(X_test_scaled[:65], X_train_subset)
end_kernel_test = time.time()
print(f"Test kernel matrix computed in {end_kernel_test - start_kernel_test:.2f} seconds.")

print("Quantum Kernel Train Matrix shape:", quantum_kernel_train.shape)
print("Quantum Kernel Test Matrix shape:", quantum_kernel_test.shape)

# Ensure the kernel matrices are symmetric for SVC (training kernel)
# and handle potential floating point inaccuracies
quantum_kernel_train = (quantum_kernel_train + quantum_kernel_train.T) / 2

# ============================================
# STEP 4: TRAIN QSVM
# ============================================
print("\n" + "="*50)
print("Training Quantum SVM...")

start_qsvm_train = time.time()
# Use SVC with precomputed kernel
qsvm = SVC(kernel='precomputed', C=1.0, random_state=42)
qsvm.fit(quantum_kernel_train, y_train_subset)
end_qsvm_train = time.time()

print(f"Quantum SVM training complete in {end_qsvm_train - start_qsvm_train:.2f} seconds.")

# ============================================
# STEP 5: EVALUATE QSVM
# ============================================
print("\n" + "="*50)
print("Evaluating Quantum SVM...")

start_qsvm_test = time.time()
y_pred_qsvm = qsvm.predict(quantum_kernel_test)
end_qsvm_test = time.time()

accuracy_qsvm = accuracy_score(y_test[:65], y_pred_qsvm)

print(f"Quantum SVM testing complete in {end_qsvm_test - start_qsvm_test:.4f} seconds.")

# ============================================
# STEP 6: RESULTS
# ============================================
print("\nFINAL QUANTUM SVM RESULTS:")
print("="*50)
print(f"Accuracy: {accuracy_qsvm:.2%}")
print("\nDetailed Report:")
# classification_report is imported from sklearn.metrics in cell dd8298ac
print(classification_report(y_test[:65], y_pred_qsvm[:65], labels=[0, 1], target_names=['Normal', 'Attack']))