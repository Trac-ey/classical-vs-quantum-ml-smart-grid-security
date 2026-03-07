# ============================================
# CLASSICAL SVM 
# ============================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import time
import os
import glob
import warnings
warnings.filterwarnings('ignore')

print("STARTING CLASSICAL SVM...")
print("="*50)

# ============================================
# STEP 1: FIND ALL CSV FILES
# ============================================
#print("\nScanning your folders...")

# Find all CSV files
all_csv = []
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.csv'):
            all_csv.append(os.path.join(root, file))

#print(f"Found {len(all_csv)} CSV files total")

# ============================================
# STEP 2: LOOK FOR COMBINED DATASET FIRST (EASIEST)
# ============================================
combined_file = None
for file in all_csv:
    if 'DNN' in file or 'ML' in file or 'Selected' in file:
        if 'Attack_type' in open(file, 'r').readline():
            combined_file = file
            #print(f"\nFound combined dataset: {file}")
            break

if combined_file:
    #print("\nLoading combined dataset...")
    df = pd.read_csv(combined_file, nrows=10000)
    #print(f"Loaded {len(df)} samples")
    
    # Check if it has both classes
    if 'Attack_type' in df.columns:
        classes = df['Attack_type'].unique()
        if 'Normal' in classes and len(classes) > 1:
            #print("Combined file has both Normal and Attacks!")
            df['is_attack'] = (df['Attack_type'] != 'Normal').astype(int)
        else:
            #print("Combined file doesn't have both classes, will use separate files")
            combined_file = None

# ============================================
# STEP 3: IF NO COMBINED FILE, USE SEPARATE FILES
# ============================================
if not combined_file:
    #print("\nUsing separate Normal and Attack files...")
    
    # Find Normal files
    normal_files = []
    for file in all_csv:
        if 'Normal' in file or 'sensor' in file.lower():
            normal_files.append(file)
    
    # Find Attack files
    attack_files = []
    for file in all_csv:
        if 'attack' in file.lower() and 'Normal' not in file:
            attack_files.append(file)
    
    print(f"Found {len(normal_files)} Normal files")
    print(f"Found {len(attack_files)} Attack files")
    
    if not normal_files or not attack_files:
        print("Could not find both Normal and Attack files")
        exit()
    
    # Use first Normal file and first Attack file
    normal_file = normal_files[0]
    attack_file = attack_files[0]
    
    #print(f"\nUsing Normal file: {normal_file}")
    #print(f"Using Attack file: {attack_file}")
    
    # Load 5000 from each
    df_normal = pd.read_csv(normal_file, nrows=5000)
    df_attack = pd.read_csv(attack_file, nrows=5000)
    
    #print(f"Loaded {len(df_normal)} Normal samples")
    #print(f"Loaded {len(df_attack)} Attack samples")
    
    # Add labels
    df_normal['Attack_type'] = 'Normal'
    df_attack['Attack_type'] = 'Attack'
    
    # Combine
    df = pd.concat([df_normal, df_attack], ignore_index=True)

# ============================================
# STEP 4: PREPARE DATA
# ============================================
#print("\nPreparing data...")

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Create binary labels if not already done
if 'is_attack' not in df.columns:
    df['is_attack'] = (df['Attack_type'] != 'Normal').astype(int)

# Show distribution
normal_count = sum(df['is_attack'] == 0)
attack_count = sum(df['is_attack'] == 1)
print(f"\nNormal samples: {normal_count}")
print(f"Attack samples: {attack_count}")

if normal_count == 0 or attack_count == 0:
    print("Need both classes!")
    exit()

# Select numerical features:1
# numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# numerical_cols = [c for c in numerical_cols if c not in ['is_attack']]
# X = df[numerical_cols].fillna(0)
# y = df['is_attack']

# print(f"\nUsing {len(numerical_cols)} numerical features")
# print(f"Sample features: {numerical_cols[:5]}")


#2
# Select numerical features:2
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numerical_cols = [c for c in numerical_cols if c not in ['is_attack']]

# More aggressive leaky feature removal - lower threshold
X_full = df[numerical_cols].fillna(0)
correlations = X_full.corrwith(df['is_attack']).abs()
print("\nTop correlated features:")
print(correlations.sort_values(ascending=False).head(10))

# Drop top 5 most correlated features
always_drop = ['attack_label'] # new 
top_leaky = correlations.sort_values(ascending=False).head(5).index.tolist()
cols_to_drop=list(set(always_drop + top_leaky)) # added this to ensure attack_label is always dropped, even if not in top 5
#print(f"\nDropping top 5 correlated features: {top_leaky}")
numerical_cols = [c for c in numerical_cols if c not in cols_to_drop]

X = df[numerical_cols].fillna(0)
X=X.loc[:, X.std()>0] # Drop zero variance features
y = df['is_attack']

# ============================================
# STEP 5: SPLIT AND SCALE
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining: {len(X_train)} samples")
print(f"Testing: {len(X_test)} samples")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# STEP 6: TRAIN SVM
# ============================================
print("\n" + "="*50)
print("Training Classical SVM...")
print("="*50)

start = time.time()
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train_scaled, y_train)
train_time = time.time() - start

# ============================================
# STEP 7: EVALUATE
# ============================================
start_test = time.time()
y_pred = svm.predict(X_test_scaled)
test_time = time.time() - start_test
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.2%}")
print(f"Training time: {train_time:.2f} seconds")
print(f"Testing time: {test_time:.4f} seconds")
print(f"Features used: {X_train_scaled.shape[1]}")

# ============================================
# STEP 8: SAVE FOR QUANTUM
# ============================================
#print("\nSaving for Quantum SVM...")
np.save('X_train_scaled.npy', X_train_scaled)
np.save('X_test_scaled.npy', X_test_scaled)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)
#print("Data saved!")

# ============================================
# STEP 9: RESULTS
# ============================================
print("\n" + "="*50)
print("CLASSICAL SVM COMPLETE")
print("="*50)
print(f"Accuracy: {accuracy:.2%}")
print(f"Training time: {train_time:.2f}s")
print(f"Testing time: {test_time:.4f}s")
print(f"Features: {X_train_scaled.shape[1]}")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))