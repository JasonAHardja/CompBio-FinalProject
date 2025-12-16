import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)
from scipy.sparse import csr_matrix
import joblib

# CONFIGURATION
CSV_PATH = "/Users/jasonhardjawidjaja/Desktop/CompBio-FinalProject/DDI_data.csv"
XLSX_PATH = "/Users/jasonhardjawidjaja/Desktop/CompBio-FinalProject/DDI_types.xlsx"
OUT_DIR = "/Users/jasonhardjawidjaja/Desktop/CompBio-FinalProject/output"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
LATENT_DIM = 30          # Reduced from 50 for faster processing
MAX_TRAIN_SAMPLES = 20000  # Reduced from 40000 for speed
NEG_RATIO = 1.0
TFIDF_FEATURES = 100      # Reduced from 200

# DATA LOADING & PREPROCESSING
print("Loading DDI dataset...")
df = pd.read_csv(CSV_PATH)

# Detect drug column names
if "drug1_name" in df.columns and "drug2_name" in df.columns:
    drug1_col, drug2_col = "drug1_name", "drug2_name"
elif "drug1" in df.columns and "drug2" in df.columns:
    drug1_col, drug2_col = "drug1", "drug2"
else:
    drug1_col, drug2_col = df.columns[0], df.columns[1]

# Standardize: lowercase and strip whitespace
df[drug1_col] = df[drug1_col].astype(str).str.strip().str.lower()
df[drug2_col] = df[drug2_col].astype(str).str.strip().str.lower()
df["interaction_type"] = df["interaction_type"].astype(str).str.strip().str.lower()
df = df.drop_duplicates(subset=[drug1_col, drug2_col, "interaction_type"]).reset_index(drop=True)

print(f"Loaded {len(df)} interactions")

# BUILD DRUG INDEX & INTERACTION MATRIX
all_drugs = pd.Index(np.unique(np.concatenate([df[drug1_col].unique(), df[drug2_col].unique()])))
drug_to_idx = {d: i for i, d in enumerate(all_drugs)}
idx_to_drug = {i: d for d, i in drug_to_idx.items()}
print(f"Unique drugs: {len(all_drugs)}")

# Sparse binary interaction matrix (symmetric)
n = len(all_drugs)
rows, cols, data_vals = [], [], []
for _, r in df.iterrows():
    i, j = drug_to_idx[r[drug1_col]], drug_to_idx[r[drug2_col]]
    rows.extend([i, j])
    cols.extend([j, i])
    data_vals.extend([1, 1])

interaction_matrix = csr_matrix((data_vals, (rows, cols)), shape=(n, n))

# MATRIX FACTORIZATION (LATENT FEATURES)
print(f"Computing latent features with SVD (dim={LATENT_DIM})...")
svd = TruncatedSVD(n_components=LATENT_DIM, random_state=RANDOM_STATE)
latent = svd.fit_transform(interaction_matrix)
latent_df = pd.DataFrame(latent, index=all_drugs, columns=[f"svd_{i}" for i in range(LATENT_DIM)])

# PREPARE POSITIVE & NEGATIVE EXAMPLES
print("Creating training dataset...")

# Positive pairs (observed interactions)
pos_pairs = set()
pos_rows = []
for _, r in df.iterrows():
    a, b = r[drug1_col], r[drug2_col]
    if a == b:
        continue
    key = tuple(sorted([a, b]))
    if key not in pos_pairs:
        pos_pairs.add(key)
        pos_rows.append({"drug1": key[0], "drug2": key[1], "interaction_type": r["interaction_type"]})

pos_df = pd.DataFrame(pos_rows)

# Negative sampling (non-interacting pairs)
num_neg = int(len(pos_df) * NEG_RATIO)
rng = np.random.RandomState(RANDOM_STATE)
neg_rows = []
attempts = 0
while len(neg_rows) < num_neg and attempts < num_neg * 10:
    i, j = rng.randint(0, n), rng.randint(0, n)
    if i == j:
        attempts += 1
        continue
    key = tuple(sorted([idx_to_drug[i], idx_to_drug[j]]))
    if key not in pos_pairs:
        neg_rows.append({"drug1": key[0], "drug2": key[1], "interaction_type": "no_interaction"})
        pos_pairs.add(key)
    attempts += 1

neg_df = pd.DataFrame(neg_rows)
pairs_df = pd.concat([pos_df, neg_df], ignore_index=True)

# Subsample if too large
if len(pairs_df) > MAX_TRAIN_SAMPLES:
    pairs_df = pairs_df.sample(n=MAX_TRAIN_SAMPLES, random_state=RANDOM_STATE).reset_index(drop=True)

print(f"Training pairs: {len(pos_df)} positive + {len(neg_df)} negative")

# FEATURE ENGINEERING
# Encode labels for all data
label_encoder = LabelEncoder()
pairs_df["label"] = label_encoder.fit_transform(pairs_df["interaction_type"])

print(f"Total interaction types: {len(label_encoder.classes_)}")

# TF-IDF for drug names (reduced features for speed)
tfv = TfidfVectorizer(max_features=TFIDF_FEATURES)
tf_name_matrix = tfv.fit_transform(list(all_drugs))
name_to_idx = {name: i for i, name in enumerate(all_drugs)}

# Build feature vectors for each pair (with progress indicator)
print("Building feature vectors...")
X_list = []
for idx, (_, r) in enumerate(pairs_df.iterrows()):
    if idx % 5000 == 0 and idx > 0:
        print(f"  Processed {idx}/{len(pairs_df)} pairs...")
    
    a, b = r["drug1"], r["drug2"]
    
    # Latent vectors
    la, lb = latent_df.loc[a].values, latent_df.loc[b].values
    diff, prod = np.abs(la - lb), la * lb
    
    # Name TF-IDF vectors
    ia, ib = name_to_idx[a], name_to_idx[b]
    ta, tb = tf_name_matrix[ia].toarray().ravel(), tf_name_matrix[ib].toarray().ravel()
    
    # Similarity features
    name_cos = np.dot(ta, tb) / (np.linalg.norm(ta) * np.linalg.norm(tb) + 1e-9)
    len_diff = abs(len(a) - len(b))
    
    feat = np.concatenate([la, lb, diff, prod, ta, tb, [name_cos, len_diff]])
    X_list.append(feat)

X = np.vstack(X_list)
y = pairs_df["label"].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Remove rare classes (classes with fewer than 2 samples)
class_counts = pd.Series(y).value_counts()
valid_classes = class_counts[class_counts >= 2].index
mask = pd.Series(y).isin(valid_classes).values
X_scaled_filtered = X_scaled[mask]
y_filtered = y[mask]
pairs_df_filtered = pairs_df[mask].reset_index(drop=True)

print(f"Filtered out {len(y) - len(y_filtered)} samples from rare classes")
print(f"Training samples: {len(y_filtered)}, Classes: {len(valid_classes)}")

# Train/test split with stratification (safe now)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_filtered, y_filtered, test_size=0.25, random_state=RANDOM_STATE, stratify=y_filtered)

# Create a mapping from filtered labels to original interaction types
y_train_types = pairs_df_filtered.iloc[y_train.index if hasattr(y_train, 'index') else range(len(y_train))]["interaction_type"].values
label_to_type = {label: itype for label, itype in zip(y_train, y_train_types)}
# Get unique mapping
label_to_type = {k: v for k, v in sorted(set((k, label_to_type[k]) for k in y_train))}

print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# MODEL TRAINING (Optimized for Speed)
print("Training models...")

# Use smaller SVM subset for speed (SVM is O(n²) complexity)
svm_sample_size = min(10000, len(X_train))
svm_idxs = np.random.RandomState(RANDOM_STATE).choice(len(X_train), size=svm_sample_size, replace=False)
svm_train_X, svm_train_y = X_train[svm_idxs], y_train[svm_idxs]
print(f"  SVM training on {len(svm_train_X)} samples...")

svm = SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE, cache_size=1000)
svm.fit(svm_train_X, svm_train_y)

# Neural Network (faster than SVM, can use more data)
print(f"  Neural Network training on {len(X_train)} samples...")
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=100, random_state=RANDOM_STATE, 
                    early_stopping=True, validation_fraction=0.1, n_iter_no_change=10)
mlp.fit(X_train, y_train)

# EVALUATION
print("\n" + "="*70)
print("MODEL EVALUATION")
print("="*70)

for model_name, clf in [("SVM", svm), ("Neural Network", mlp)]:
    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)
    
    print(f"\n{model_name}:")
    print("-" * 50)
    
    # Metrics
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
    print(f"Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    
    # AUC (handle potential issues with class mismatch)
    try:
        n_classes = len(np.unique(y_train))  # Use actual classes in training
        if len(np.unique(y_test)) == n_classes:
            y_test_bin = label_binarize(y_test, classes=range(n_classes))
            auc_val = roc_auc_score(y_test_bin, y_score, average='micro')
            print(f"ROC AUC (micro): {auc_val:.4f}")
    except Exception as e:
        print(f"ROC AUC: Could not compute ({str(e)[:50]})")

# Confusion Matrix - simplified approach
y_pred_mlp = mlp.predict(X_test)

# Use only labels that appear in training
train_labels = set(y_train)

# Filter both test and predictions
valid_indices = [i for i, (yt, yp) in enumerate(zip(y_test, y_pred_mlp)) 
                 if yt in train_labels and yp in train_labels]

if len(valid_indices) > 0:
    y_test_filtered = y_test[valid_indices]
    y_pred_filtered = y_pred_mlp[valid_indices]
    
    cm = confusion_matrix(y_test_filtered, y_pred_filtered)
    
    # Get the unique labels present
    all_labels_present = sorted(set(y_test_filtered) | set(y_pred_filtered))
    
    # Convert to interaction type names using our mapping
    label_names = [label_to_type.get(lbl, f"Class_{lbl}") for lbl in all_labels_present]
    
    # Limit display if too many classes
    if len(all_labels_present) > 20:
        print(f"\nConfusion matrix has {len(all_labels_present)} classes - showing simplified version")
        # Just show summary stats instead
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_test_filtered, y_pred_filtered)
        print(f"Accuracy on valid samples: {acc:.4f}")
    else:
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=label_names, yticklabels=label_names)
        plt.title("Confusion Matrix - Neural Network", fontsize=14, fontweight='bold')
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
        plt.close()
        print(f"\nConfusion matrix saved ({len(y_test_filtered)} samples, {len(all_labels_present)} classes)")
else:
    print("\nWarning: Could not create confusion matrix")

# ASPIRIN INTERACTION PREDICTION
def build_feature_vector(drugA, drugB):
    """Construct feature vector for a drug pair"""
    a, b = drugA.lower().strip(), drugB.lower().strip()
    
    # Latent vectors
    la = latent_df.loc[a].values if a in latent_df.index else np.zeros(LATENT_DIM)
    lb = latent_df.loc[b].values if b in latent_df.index else np.zeros(LATENT_DIM)
    diff, prod = np.abs(la - lb), la * lb
    
    # Name TF-IDF
    tf_dim = tf_name_matrix.shape[1]
    ta = tf_name_matrix[name_to_idx[a]].toarray().ravel() if a in name_to_idx else np.zeros(tf_dim)
    tb = tf_name_matrix[name_to_idx[b]].toarray().ravel() if b in name_to_idx else np.zeros(tf_dim)
    
    denom = np.linalg.norm(ta) * np.linalg.norm(tb) + 1e-9
    name_cos = np.dot(ta, tb) / denom if denom > 0 else 0.0
    len_diff = abs(len(a) - len(b))
    
    feat = np.concatenate([la, lb, diff, prod, ta, tb, [name_cos, len_diff]])
    return csr_matrix(feat.reshape(1, -1))

# Find aspirin
aspirin_name = None
for cand in ["aspirin", "acetylsalicylic acid", "acetylsalicylate"]:
    if cand in drug_to_idx:
        aspirin_name = cand
        break
if not aspirin_name:
    matches = [d for d in all_drugs if "aspirin" in d]
    aspirin_name = matches[0] if matches else None

if aspirin_name:
    print(f"\nGenerating interaction predictions for: {aspirin_name.upper()}")
    
    # Predict for all other drugs
    candidates = [d for d in all_drugs if d != aspirin_name]
    rows = []
    
    # Get valid class indices from training
    valid_class_indices = set(y_train)
    
    # Categorize interaction types into severity levels
    severity_mapping = {
        'major': ['contraindicated', 'toxic', 'severe', 'serious', 'fatal', 'dangerous'],
        'moderate': ['caution', 'monitor', 'moderate', 'warning', 'adjustment', 'dose'],
        'minor': ['minor', 'minimal', 'negligible', 'small']
    }
    
    def get_severity(interaction_type):
        """Determine severity level based on interaction type"""
        interaction_lower = interaction_type.lower()
        for severity, keywords in severity_mapping.items():
            if any(keyword in interaction_lower for keyword in keywords):
                return severity
        # Default classification based on confidence
        return 'moderate'
    
    for cand in candidates:
        try:
            feat = build_feature_vector(aspirin_name, cand)
            feat_scaled = scaler.transform(feat.toarray())
            
            prob_mlp = mlp.predict_proba(feat_scaled)[0]
            pred_label = mlp.predict(feat_scaled)[0]
            
            # Check if prediction is a valid class
            if pred_label not in valid_class_indices:
                continue
            
            # Get the interaction type name
            interaction_type = label_to_type.get(pred_label, "unknown")
            confidence = prob_mlp[pred_label] if pred_label < len(prob_mlp) else 0.0
            
            # Skip if predicted as no interaction with low confidence
            if interaction_type == "no_interaction" and confidence < 0.6:
                continue
            
            # Skip no_interaction entirely for the output table
            if interaction_type == "no_interaction":
                continue
            
            severity = get_severity(interaction_type)
            
            rows.append({
                "Drug": cand.title(),
                "Interaction_Type": interaction_type.replace("_", " ").title(),
                "Severity": severity.title(),
                "Confidence": f"{confidence:.1%}",
                "confidence_raw": confidence  # for sorting
            })
        except Exception as e:
            # Skip drugs that cause errors
            continue
    
    # Create clean results table
    results_df = pd.DataFrame(rows)
    
    if len(results_df) > 0:
        results_df = results_df.sort_values("confidence_raw", ascending=False).reset_index(drop=True)
        results_df = results_df.drop(columns=["confidence_raw"])  # Remove sorting column
        results_df.index += 1  # Start index at 1
        
        # Calculate statistics
        total_interactions = len(results_df)
        severity_counts = results_df["Severity"].value_counts()
        major_count = severity_counts.get("Major", 0)
        moderate_count = severity_counts.get("Moderate", 0)
        minor_count = severity_counts.get("Minor", 0)
        
        # Count unique interaction types
        interaction_type_counts = results_df["Interaction_Type"].value_counts()
        
        # Save full results
        results_df.to_csv(os.path.join(OUT_DIR, "aspirin_interactions.csv"))
        
        # Display summary (like drugs.com)
        print("\n" + "="*70)
        print(f"ASPIRIN DRUG INTERACTION SUMMARY")
        print("="*70)
        print(f"\nThere are {total_interactions} drugs predicted to interact with aspirin.")
        print(f"Of the total drug interactions:")
        print(f"  • {major_count} are MAJOR")
        print(f"  • {moderate_count} are MODERATE")
        print(f"  • {minor_count} are MINOR")
        
        # Display top interaction types
        print(f"\nTop Interaction Types:")
        for idx, (itype, count) in enumerate(interaction_type_counts.head(5).items(), 1):
            print(f"  {idx}. {itype}: {count} drugs")
        
        # Display top 30 interactions by severity
        print("\n" + "="*70)
        print("TOP PREDICTED INTERACTIONS (Sorted by Confidence)")
        print("="*70)
        
        # Show major interactions first
        major_df = results_df[results_df["Severity"] == "Major"]
        if len(major_df) > 0:
            print(f"\n  MAJOR INTERACTIONS ({len(major_df)} total):")
            print("-" * 70)
            print(major_df.head(15).to_string(index=True))
        
        # Then moderate
        moderate_df = results_df[results_df["Severity"] == "Moderate"]
        if len(moderate_df) > 0:
            print(f"\n⚡ MODERATE INTERACTIONS ({len(moderate_df)} total, showing top 10):")
            print("-" * 70)
            print(moderate_df.head(10).to_string(index=True))
        
        # Then minor
        minor_df = results_df[results_df["Severity"] == "Minor"]
        if len(minor_df) > 0:
            print(f"\nℹ  MINOR INTERACTIONS ({len(minor_df)} total, showing top 5):")
            print("-" * 70)
            print(minor_df.head(5).to_string(index=True))
        
        # Export by severity
        for severity in ["Major", "Moderate", "Minor"]:
            sev_df = results_df[results_df["Severity"] == severity]
            if len(sev_df) > 0:
                sev_df.to_csv(os.path.join(OUT_DIR, f"aspirin_interactions_{severity.lower()}.csv"))
        
        print(f"\n\nDetailed results saved:")
        print(f"  • aspirin_interactions.csv (all interactions)")
        print(f"  • aspirin_interactions_major.csv")
        print(f"  • aspirin_interactions_moderate.csv")
        print(f"  • aspirin_interactions_minor.csv")
        
    else:
        print("\nNo significant interactions predicted for aspirin")
        
else:
    print("\nWARNING: Aspirin not found in dataset")

# SAVE MODELS & ARTIFACTS
joblib.dump(svm, os.path.join(OUT_DIR, "svm_model.joblib"))
joblib.dump(mlp, os.path.join(OUT_DIR, "mlp_model.joblib"))
joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.joblib"))
joblib.dump(svd, os.path.join(OUT_DIR, "svd_latent.joblib"))
latent_df.to_csv(os.path.join(OUT_DIR, "drug_latent_features.csv"))

print(f"\n✓ All outputs saved to: {OUT_DIR}")
print("\nFiles generated:")
print("  • aspirin_interactions.csv - Full prediction results")
print("  • confusion_matrix.png - Model performance visualization")
print("  • Models: svm_model.joblib, mlp_model.joblib")
print("  • Features: drug_latent_features.csv")
