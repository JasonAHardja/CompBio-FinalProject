"""
DDI final pipeline:
- Reads /mnt/data/DDI_data.csv and /mnt/data/DDI_types.xlsx
- Builds a drugxdrug interaction matrix (binary)
- Performs matrix factorization (TruncatedSVD + optional NMF)
- Creates features for ML: latent vectors + simple similarity features
- Negative sampling to create "no_interaction" examples
- Trains SVM and MLP classifiers
- Evaluates with AUC (micro), Precision/Recall/F1
- Produces aspirin-centered ranking and saves outputs
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix, roc_curve, auc)
from scipy.sparse import csr_matrix, hstack
import joblib
import random

# Paths (uploaded files)
# REMEMBER to change the Paths and Directory based on your OWN device or Where you place the file.
CSV_PATH = "/Users/jasonhardjawidjaja/Desktop/CompBio-FinalProject/DDI_data.csv"
XLSX_PATH = "/Users/jasonhardjawidjaja/Desktop/CompBio-FinalProject/DDI_types_merged.xlsx"
OUT_DIR = "/Users/jasonhardjawidjaja/Desktop/CompBio-FinalProject/output"
os.makedirs(OUT_DIR, exist_ok=True)

# Parameters (adjustable)
RANDOM_STATE = 42
LATENT_DIM = 50           # latent dimensions for SVD/NMF
MAX_PAIRS = 50000         # limit pairs used to keep runtime reasonable (adjust as needed)
NEG_RATIO = 1.0           # negative samples ratio relative to positive samples
SAMPLE_FOR_TRAINING = True
MAX_TRAIN_SAMPLES = 40000  # if True and dataset > this, subsample to this many pairs

print("Loading data...")
df = pd.read_csv(CSV_PATH)
print("Rows loaded:", len(df))
# check expected column names (adjust if different)
# Many DDI datasets call columns drug1_name / drug2_name or drug1 / drug2
# We'll try to detect common names:
if "drug1_name" in df.columns and "drug2_name" in df.columns:
    drug1_col = "drug1_name"
    drug2_col = "drug2_name"
elif "drug1" in df.columns and "drug2" in df.columns:
    drug1_col = "drug1"
    drug2_col = "drug2"
else:
    # fallback to first two columns that look like names
    drug1_col = df.columns[0]
    drug2_col = df.columns[1]
print("Using columns:", drug1_col, drug2_col)

# load types if present
if os.path.exists(XLSX_PATH):
    types_df = pd.read_excel(XLSX_PATH)
    print("Loaded types xlsx:", types_df.shape)
else:
    types_df = pd.DataFrame()

# Standardize drug name columns
df[drug1_col] = df[drug1_col].astype(str).str.strip()
df[drug2_col] = df[drug2_col].astype(str).str.strip()
df["interaction_type"] = df["interaction_type"].astype(str).str.strip()

# lowercase names for consistency
df[drug1_col] = df[drug1_col].str.lower()
df[drug2_col] = df[drug2_col].str.lower()
df["interaction_type"] = df["interaction_type"].str.lower()

# remove exact duplicates (optional)
df = df.drop_duplicates(subset=[drug1_col, drug2_col, "interaction_type"]).reset_index(drop=True)
print("After dedup:", len(df))

# Build drug list and mapping

all_drugs = pd.Index(np.unique(np.concatenate([df[drug1_col].unique(), df[drug2_col].unique()])))
print("Unique drugs:", len(all_drugs))
drug_to_idx = {d: i for i, d in enumerate(all_drugs)}
idx_to_drug = {i: d for d, i in drug_to_idx.items()}

# ---------------------------
# Build binary interaction matrix (symmetric)
# We will mark 1 if any interaction exists between pair
# ---------------------------
print("Building sparse interaction matrix (binary)")
n = len(all_drugs)
rows = []
cols = []
data_vals = []
for _, r in df.iterrows():
    i = drug_to_idx[r[drug1_col]]
    j = drug_to_idx[r[drug2_col]]
    rows.append(i)
    cols.append(j)
    data_vals.append(1)
    # also add symmetric cell if not same
    if i != j:
        rows.append(j)
        cols.append(i)
        data_vals.append(1)

interaction_matrix = csr_matrix((data_vals, (rows, cols)), shape=(n, n))
print("Interaction matrix shape:", interaction_matrix.shape, " nnz:", interaction_matrix.nnz)

# ---------------------------
# Matrix Factorization (latent features)
# Use TruncatedSVD on the interaction matrix (works for sparse)
# Also compute NMF on dense version if needed (optional)
# ---------------------------
print("Computing TruncatedSVD latent factors (k=%d)..." % LATENT_DIM)
svd = TruncatedSVD(n_components=LATENT_DIM, random_state=RANDOM_STATE)
# use the sparse interaction matrix
latent = svd.fit_transform(interaction_matrix)  # shape (n_drugs, LATENT_DIM)
print("Latent shape:", latent.shape)

# Save latent features for later
latent_df = pd.DataFrame(latent, index=all_drugs, columns=[f"svd_{i}" for i in range(LATENT_DIM)])
latent_df.to_csv(os.path.join(OUT_DIR, "drug_latent_svd.csv"))

# (Optional) NMF: if you want non-negative parts, uncomment:
# print("Computing NMF (this may be slower)...")
# nmf = NMF(n_components=LATENT_DIM, init='nndsvda', random_state=RANDOM_STATE, max_iter=200)
# latent_nmf = nmf.fit_transform(interaction_matrix.toarray())
# pd.DataFrame(latent_nmf, index=all_drugs).to_csv(os.path.join(OUT_DIR, "drug_latent_nmf.csv"))

# ---------------------------
# Prepare dataset for supervised ML
# For ML we need positive and negative examples:
# - Positives: rows in df (unique pairs)
# - Negatives: random drug pairs not in observed set (sampled)
# We'll create features by concatenating drug1_latent and drug2_latent and some simple name/text features.
# ---------------------------

print("Preparing positive examples")
# get unique observed unordered pairs (force i < j for uniqueness)
pos_pairs = set()
pos_rows = []
for _, r in df.iterrows():
    a = r[drug1_col]
    b = r[drug2_col]
    if a == b:
        continue
    key = tuple(sorted([a, b]))
    if key not in pos_pairs:
        pos_pairs.add(key)
        pos_rows.append({"drug1": key[0], "drug2": key[1], "interaction_type": r["interaction_type"]})

pos_df = pd.DataFrame(pos_rows)
print("Unique positive pairs:", len(pos_df))

# map interaction types to labels (we'll add 'no_interaction' later)
label_encoder = LabelEncoder()
# ensure we will include 'no_interaction' after creating negatives
unique_types = pos_df["interaction_type"].unique().tolist()
print("Interaction types (observed):", unique_types[:10], " ... count:", len(unique_types))

# Negative sampling: sample random drug pairs not in pos_pairs
num_pos = len(pos_df)
num_neg = int(num_pos * NEG_RATIO)
print("Sampling negatives:", num_neg)
rng = np.random.RandomState(RANDOM_STATE)
neg_rows = []
attempts = 0
while len(neg_rows) < num_neg and attempts < num_neg * 10:
    i = rng.randint(0, n)
    j = rng.randint(0, n)
    if i == j:
        attempts += 1
        continue
    a = idx_to_drug[i]
    b = idx_to_drug[j]
    key = tuple(sorted([a, b]))
    if key in pos_pairs:
        attempts += 1
        continue
    neg_rows.append({"drug1": key[0], "drug2": key[1], "interaction_type": "no_interaction"})
    pos_pairs.add(key)  # prevent duplicates
    attempts += 1

neg_df = pd.DataFrame(neg_rows)
print("Negatives sampled:", len(neg_df))

# Combine
pairs_df = pd.concat([pos_df, neg_df], ignore_index=True)
print("Total pairs for ML:", len(pairs_df))

# Optionally limit size to MAX_PAIRS for speed
if SAMPLE_FOR_TRAINING and len(pairs_df) > MAX_TRAIN_SAMPLES:
    pairs_df = pairs_df.sample(n=MAX_TRAIN_SAMPLES, random_state=RANDOM_STATE).reset_index(drop=True)
    print("Subsampled to", len(pairs_df))

# Encode labels
pairs_df["label"] = label_encoder.fit_transform(pairs_df["interaction_type"])
print("Label classes:", list(label_encoder.classes_))

# ---------------------------
# Feature construction
# For each pair:
# - latent vector for drug1
# - latent vector for drug2
# - absolute difference of latent vectors
# - name-based TF-IDF similarity (simple)
# ---------------------------
print("Building features for ML...")

# latent features
def get_latent(drug):
    return latent_df.loc[drug].values

# prepare TF-IDF on drug names to build name vectors
all_names = list(all_drugs)
tfv = TfidfVectorizer(max_features=200)  # small for speed
tf_name_matrix = tfv.fit_transform(all_names)
name_to_idx = {name: i for i, name in enumerate(all_names)}

# build feature matrix
X_list = []
y = pairs_df["label"].values
for _, r in pairs_df.iterrows():
    a = r["drug1"]
    b = r["drug2"]
    # latent vectors
    la = latent_df.loc[a].values
    lb = latent_df.loc[b].values
    diff = np.abs(la - lb)
    prod = la * lb
    # name tf vectors
    ia = name_to_idx[a]
    ib = name_to_idx[b]
    ta = tf_name_matrix[ia].toarray().ravel()
    tb = tf_name_matrix[ib].toarray().ravel()
    # name similarity features
    name_cos = np.dot(ta, tb) / (np.linalg.norm(ta) * np.linalg.norm(tb) + 1e-9)
    len_diff = abs(len(a) - len(b))
    # combine into single vector (flatten)
    feat = np.concatenate([la, lb, diff, prod, ta, tb, [name_cos, len_diff]])
    X_list.append(feat)

X = np.vstack(X_list)
print("Feature matrix shape:", X.shape)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=RANDOM_STATE)

print("Train/test shapes:", X_train.shape, X_test.shape)

# ---------------------------
# Model 1: SVM (RBF) with probability=True
# Keep it small — use a subset if needed since SVC scales O(n^2)
# ---------------------------
print("Training SVM (may be slow)...")
# For speed, if large, use a subset for SVM training
svm_train_X = X_train
svm_train_y = y_train
if len(X_train) > 20000:
    idxs = np.random.RandomState(RANDOM_STATE).choice(len(X_train), size=20000, replace=False)
    svm_train_X = X_train[idxs]
    svm_train_y = y_train[idxs]
    print("SVM training on subset:", len(svm_train_X))

svm = SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)
svm.fit(svm_train_X, svm_train_y)
joblib.dump(svm, os.path.join(OUT_DIR, "svm_model.joblib"))

# ---------------------------
# Model 2: Neural Network (MLP)
# ---------------------------
print("Training MLPClassifier...")
mlp = MLPClassifier(hidden_layer_sizes=(256,128), max_iter=200, random_state=RANDOM_STATE)
mlp.fit(X_train, y_train)
joblib.dump(mlp, os.path.join(OUT_DIR, "mlp_model.joblib"))

# ---------------------------
# Graph-based baseline: network centrality
# Build graph from positives and rank neighbors by degree / common neighbors
# ---------------------------
print("Building graph baseline...")
G = nx.Graph()
G.add_nodes_from(all_drugs)
for _, r in pos_df.iterrows():
    a, b = r["drug1"], r["drug2"]
    G.add_edge(a, b)
# degree centrality
deg = nx.degree_centrality(G)
nx.write_gml(G, os.path.join(OUT_DIR, "ddi_graph.gml"))

# ---------------------------
# Evaluation helpers
# ---------------------------
def evaluate_model(clf, X_t, y_t, model_name):
    print(f"\n=== Evaluation: {model_name} ===")
    y_pred = clf.predict(X_t)
    if hasattr(clf, "predict_proba"):
        y_score = clf.predict_proba(X_t)
    else:
        # fallback: decision_function -> convert via softmax-like
        try:
            dec = clf.decision_function(X_t)
            # if binary, shape (n_samples,), convert to 2col
            if dec.ndim == 1:
                y_score = np.vstack([1-dec, dec]).T
            else:
                # apply softmax
                exp = np.exp(dec - dec.max(axis=1, keepdims=True))
                y_score = exp / exp.sum(axis=1, keepdims=True)
        except:
            y_score = None

    print("Classification report:")
    print(classification_report(y_t, y_pred, target_names=list(label_encoder.classes_)))
    # micro-averaged AUC (multi-class)
    if y_score is not None:
        n_classes = len(label_encoder.classes_)
        y_t_bin = label_binarize(y_t, classes=range(n_classes))
        auc_val = roc_auc_score(y_t_bin, y_score, average='micro')
        print("Micro-averaged ROC AUC:", auc_val)
    prec, rec, f1, _ = precision_recall_fscore_support(y_t, y_pred, average='macro')
    print(f"Macro Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    return y_pred, (y_score if y_score is not None else None)

# Evaluate SVM and MLP
svm_pred, svm_score = evaluate_model(svm, X_test, y_test, "SVM (RBF)")
mlp_pred, mlp_score = evaluate_model(mlp, X_test, y_test, "MLP")

# Confusion matrix for MLP
cm = confusion_matrix(y_test, mlp_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=list(label_encoder.classes_), yticklabels=list(label_encoder.classes_))
plt.title("Confusion matrix (MLP)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_mlp.png"))
plt.show()

# ---------------------------
# --- ADDED FUNCTION: build_feature_vector ---
# This constructs the same feature vector format used during training
# for a single drug pair (drugA, drugB) and returns a csr_matrix (1 x n_features)
# ---------------------------
def build_feature_vector(drugA, drugB):
    """Return a csr_matrix row matching the training feature layout."""
    # ensure lowercased and stripped (dataset used lowercase)
    a = drugA.lower().strip()
    b = drugB.lower().strip()

    # latent vectors (if missing, use zeros)
    if a in latent_df.index:
        la = latent_df.loc[a].values
    else:
        la = np.zeros(LATENT_DIM)
    if b in latent_df.index:
        lb = latent_df.loc[b].values
    else:
        lb = np.zeros(LATENT_DIM)

    diff = np.abs(la - lb)
    prod = la * lb

    # name tf vectors (if missing, use zero vector)
    tf_dim = tf_name_matrix.shape[1]
    if a in name_to_idx:
        ta = tf_name_matrix[name_to_idx[a]].toarray().ravel()
    else:
        ta = np.zeros(tf_dim)
    if b in name_to_idx:
        tb = tf_name_matrix[name_to_idx[b]].toarray().ravel()
    else:
        tb = np.zeros(tf_dim)

    # name similarity / length features
    denom = (np.linalg.norm(ta) * np.linalg.norm(tb) + 1e-9)
    name_cos = np.dot(ta, tb) / denom if denom > 0 else 0.0
    len_diff = abs(len(a) - len(b))

    feat = np.concatenate([la, lb, diff, prod, ta, tb, [name_cos, len_diff]])
    return csr_matrix(feat.reshape(1, -1))

# ---------------------------
# Focus on ASPIRIN (analysis + ranking)
# ---------------------------
aspirin_name = None
# try common variations
for cand in ["aspirin", "acetylsalicylic acid", "acetylsalicylate"]:
    if cand in drug_to_idx:
        aspirin_name = cand
        break
if aspirin_name is None:
    # fallback: find the closest match (case-insensitive)
    matches = [d for d in all_drugs if "aspirin" in d]
    aspirin_name = matches[0] if matches else None

if aspirin_name is None:
    print("WARNING: 'aspirin' not found in drug list. Cannot produce aspirin-centric ranking.")
else:
    print("Aspirin identifier used:", aspirin_name)
    # get all other drugs
    candidates = [d for d in all_drugs if d != aspirin_name]

    # build features and predict prob for each candidate
    rows = []
    for cand in candidates:
        feat = build_feature_vector(aspirin_name, cand)
        feat_scaled = scaler.transform(feat.toarray())
        # get mlp probability distribution
        prob_mlp = mlp.predict_proba(feat_scaled)[0]
        prob_svm = svm.predict_proba(feat_scaled)[0]
        # pick top class probability and its label
        top_idx_mlp = np.argmax(prob_mlp)
        top_class_mlp = label_encoder.inverse_transform([top_idx_mlp])[0]
        top_prob_mlp = prob_mlp[top_idx_mlp]
        rows.append({
            "candidate": cand,
            "mlp_top_class": top_class_mlp,
            "mlp_top_prob": float(top_prob_mlp),
            "svm_top_prob": float(np.max(prob_svm)),
            "degree_centrality": deg.get(cand, 0.0)
        })
    rank_df = pd.DataFrame(rows)
    # sort by mlp_top_prob desc
    rank_df = rank_df.sort_values("mlp_top_prob", ascending=False).reset_index(drop=True)
    rank_df.to_csv(os.path.join(OUT_DIR, "aspirin_ranking.csv"), index=False)
    print("Saved aspirin_ranking.csv (top 20):")
    print(rank_df.head(20))

# ---------------------------
# Save models + artifacts
# ---------------------------
joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.joblib"))
joblib.dump(svd, os.path.join(OUT_DIR, "svd_latent.joblib"))
latent_df.to_csv(os.path.join(OUT_DIR, "drug_latent_svd.csv"))

print("All done. Outputs saved to", OUT_DIR)
