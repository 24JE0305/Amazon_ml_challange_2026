import pandas as pd
import numpy as np
import os
import re
import gc
import concurrent.futures
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

candidate_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

def clean_text(text):
    if pd.isna(text) or text == 'nan': return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    replacements = {
        r'\bcorp\b': 'corporation', r'\binc\b': 'incorporated',
        r'\bltd\b': 'limited', r'\bpvt\b': 'private',
        r'\bco\b': 'company', r'\bst\b': 'street', r'\brd\b': 'road'
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    return " ".join(text.split())

def process_file(filename):
    print(f"Reading {filename}...")
    df = pd.read_csv(os.path.join(TEST_DIR, filename), sep="\t", dtype=str)
    ids = df['entity_id'].values
    text = (df['business_name'].fillna('') + " " + df['business_address'].fillna('')).apply(clean_text).values
    del df
    gc.collect()
    return ids, text

print("Loading and cleaning S2 and S3...")
s2_ids, s2_text = process_file("test_source2.tsv")
s3_ids, s3_text = process_file("test_source3.tsv")

s2_s3_ids = np.concatenate([s2_ids, s3_ids])
s2_s3_text = np.concatenate([s2_text, s3_text])
del s2_ids, s3_ids, s2_text, s3_text
gc.collect()

print("Fitting TF-IDF (CPU Optimized - High Accuracy)...")
# Tighter max_df and sublinear_tf=True protect precision (F0.5) while massively speeding up math
vectorizer = TfidfVectorizer(
    analyzer='word', 
    ngram_range=(1, 1), 
    min_df=3, 
    max_df=0.01, 
    max_features=50000, 
    sublinear_tf=True, 
    dtype=np.float32
)
s2_s3_matrix = vectorizer.fit_transform(s2_s3_text)

print("Transposing search space...")
s2_s3_matrix_T = s2_s3_matrix.T.tocsr()
del s2_s3_text
gc.collect()

print("Loading and cleaning S1...")
s1_ids, s1_text = process_file("test_source1.tsv")
s1_matrix = vectorizer.transform(s1_text)
del s1_text
gc.collect()

BLOCKING_THRESHOLD = 0.40 
batch_size = 5000 # Increased to 5000 because the matrix is now much lighter

print("Executing Multi-threaded Scipy sparse cosine similarity...")
batches = [(start_idx, min(start_idx + batch_size, s1_matrix.shape[0])) 
           for start_idx in range(0, s1_matrix.shape[0], batch_size)]

def process_batch(batch):
    start_idx, end_idx = batch
    batch_matrix = s1_matrix[start_idx:end_idx]
    
    # Scipy C-backend releases GIL, allowing true multithreading for dot product
    sim_sparse = batch_matrix.dot(s2_s3_matrix_T)
    
    results = []
    for i in range(end_idx - start_idx):
        s1_id = s1_ids[start_idx + i]
        row_start = sim_sparse.indptr[i]
        row_end = sim_sparse.indptr[i+1]
        
        cols = sim_sparse.indices[row_start:row_end]
        data = sim_sparse.data[row_start:row_end]
        
        candidate_cols = cols[data >= BLOCKING_THRESHOLD]
        if len(candidate_cols) > 0:
            candidate_ids_str = ",".join(s2_s3_ids[candidate_cols])
            results.append(f"{s1_id}\t{candidate_ids_str}\n")
        else:
            results.append(f"{s1_id}\t\n")
            
    return results, end_idx

# Writing to file sequentially in the main thread prevents file corruption
with open(candidate_path, "w") as f_cand:
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
    
    # max_workers=8 will fire up multiple CPU cores
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_batch = {executor.submit(process_batch, batch): batch for batch in batches}
        
        for future in concurrent.futures.as_completed(future_to_batch):
            results, end_idx = future.result()
            f_cand.writelines(results)
            print(f"Processed batch ending at {end_idx} / {s1_matrix.shape[0]} records...")

print("Blocking complete! Candidates saved to output/candidate_pairs.tsv")