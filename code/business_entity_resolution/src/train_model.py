import pandas as pd
import numpy as np
import os
import random
import difflib
from xgboost import XGBClassifier
import joblib

# Import your teammate's cleaning functions
from cleaning_patterns import canonicalize_suffix, canonicalize_street, extract_landmark

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
MODEL_DIR = os.path.join(BASE_DIR, "code", "business_entity_resolution")

def string_similarity(s1, s2):
    if pd.isna(s1) or pd.isna(s2): return 0.0
    return difflib.SequenceMatcher(None, str(s1).lower(), str(s2).lower()).ratio()

print("Sampling ground truth pairs line-by-line to save memory...")
positive_pairs = []

# Stream the ground truth to bypass RAM limits
with open(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
    header = f.readline()
    for line in f:
        parts = line.strip('\n').split('\t')
        if len(parts) == 2 and parts[1]:  # Ensure there are matches
            s1_id = parts[0]
            matches = parts[1].split(',')
            for m_id in matches:
                positive_pairs.append((s1_id, m_id.strip()))

# Take a 10,000 pair sample for training
random.seed(42)
if len(positive_pairs) > 10000:
    positive_pairs = random.sample(positive_pairs, 10000)

# Generate hard negative pairs by shuffling the matches
negative_pairs = []
s1_pool = [p[0] for p in positive_pairs]
s2_pool = [p[1] for p in positive_pairs]
random.shuffle(s2_pool)
for i in range(len(s1_pool)):
    if positive_pairs[i][1] != s2_pool[i]: 
        negative_pairs.append((s1_pool[i], s2_pool[i]))

# Build a set of exactly the IDs we need
needed_s1 = set(p[0] for p in positive_pairs + negative_pairs)
needed_s2_s3 = set(p[1] for p in positive_pairs + negative_pairs)

print(f"Need to extract {len(needed_s1)} S1 records and {len(needed_s2_s3)} S2/S3 records...")

s1_data = {}
s2_s3_data = {}

def extract_needed(filepath, needed_set, out_dict):
    """Scans a file line-by-line and saves ONLY the rows matching the needed_set."""
    print(f"Scanning {os.path.basename(filepath)}...")
    with open(filepath, "r", encoding="utf-8") as f:
        header = f.readline().strip('\n').split('\t')
        idx_id = header.index('entity_id')
        idx_name = header.index('business_name')
        idx_addr = header.index('business_address')
        idx_country = header.index('country')
        
        for line in f:
            parts = line.strip('\n').split('\t')
            if len(parts) > max(idx_id, idx_name, idx_addr, idx_country):
                eid = parts[idx_id]
                if eid in needed_set:
                    out_dict[eid] = {
                        'name': parts[idx_name],
                        'addr': parts[idx_addr],
                        'country': parts[idx_country]
                    }

# Extract exactly what we need, ignoring the millions of unused rows
extract_needed(os.path.join(TRAIN_DIR, "train_source1.tsv"), needed_s1, s1_data)
extract_needed(os.path.join(TRAIN_DIR, "train_source2.tsv"), needed_s2_s3, s2_s3_data)
extract_needed(os.path.join(TRAIN_DIR, "train_source3.tsv"), needed_s2_s3, s2_s3_data)

print("Applying cleaning functions and building features...")
features = []
labels = []

def build_feature(s1_id, m_id, label):
    if s1_id in s1_data and m_id in s2_s3_data:
        d1 = s1_data[s1_id]
        d2 = s2_s3_data[m_id]
        
        name1 = canonicalize_suffix(d1['name'])
        name2 = canonicalize_suffix(d2['name'])
        addr1, l1 = extract_landmark(canonicalize_street(d1['addr']))
        addr2, l2 = extract_landmark(canonicalize_street(d2['addr']))
        
        features.append({
            'name_sim': string_similarity(name1, name2),
            'addr_sim': string_similarity(addr1, addr2),
            'country_match': int(d1['country'] == d2['country']),
            'landmark_match': int(l1 == l2 and l1 is not None)
        })
        labels.append(label)

for s1_id, m_id in positive_pairs:
    build_feature(s1_id, m_id, 1)
    
for s1_id, m_id in negative_pairs:
    build_feature(s1_id, m_id, 0)

X = pd.DataFrame(features)
y = np.array(labels)

print("Training XGBoost Classifier...")
# Scale_pos_weight is set low to prioritize precision (F_0.5 focus)
model = XGBClassifier(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=5, 
    scale_pos_weight=0.25 
)
model.fit(X, y)

model_path = os.path.join(MODEL_DIR, "xgb_matcher.pkl")
joblib.dump(model, model_path)
print(f"Model saved successfully to {model_path}! Ready for final inference.")