import pandas as pd
import numpy as np
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define relative paths based on the project structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

def clean_text(text):
    """Normalize text by lowercasing, removing punctuation, and standardizing common suffixes."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Standardize common business abbreviations
    replacements = {
        r'\bcorp\b': 'corporation',
        r'\binc\b': 'incorporated',
        r'\bltd\b': 'limited',
        r'\bpvt\b': 'private',
        r'\bco\b': 'company',
        r'\bst\b': 'street',
        r'\brd\b': 'road'
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    
    return " ".join(text.split())

def load_data():
    s1 = pd.read_csv(os.path.join(TEST_DIR, "test_source1.tsv"), sep="\t", dtype=str)
    s2 = pd.read_csv(os.path.join(TEST_DIR, "test_source2.tsv"), sep="\t", dtype=str)
    s3 = pd.read_csv(os.path.join(TEST_DIR, "test_source3.tsv"), sep="\t", dtype=str)
    return s1, s2, s3

def generate_matches(s1, s2, s3):
    # Combine S2 and S3 for our search space
    s2_s3 = pd.concat([s2, s3], ignore_index=True)
    
    # Feature Engineering: Create a combined string
    s1['combined_text'] = (s1['business_name'].fillna('') + " " + s1['business_address'].fillna('')).apply(clean_text)
    s2_s3['combined_text'] = (s2_s3['business_name'].fillna('') + " " + s2_s3['business_address'].fillna('')).apply(clean_text)
    
    print("Fitting TF-IDF Vectorizer (Memory Optimized)...")
    # Switch to word analyzer and limit max_features to drastically reduce RAM usage
    vectorizer = TfidfVectorizer(
        analyzer='word', 
        ngram_range=(1, 2), 
        min_df=5, 
        max_features=250000, 
        dtype=np.float32  # Use 32-bit floats to save memory
    )
    s2_s3_matrix = vectorizer.fit_transform(s2_s3['combined_text'])
    s1_matrix = vectorizer.transform(s1['combined_text'])
    
    candidates = []
    matches = []
    
    MATCH_THRESHOLD = 0.85 
    BLOCKING_THRESHOLD = 0.40 
    
    print("Computing similarities and generating pairs...")
    # Reduced batch size to prevent memory spikes during cosine similarity math
    batch_size = 250 
    
    for start_idx in range(0, s1_matrix.shape[0], batch_size):
        end_idx = min(start_idx + batch_size, s1_matrix.shape[0])
        batch_matrix = s1_matrix[start_idx:end_idx]
        
        sim_scores = cosine_similarity(batch_matrix, s2_s3_matrix)
        
        for i, row_idx in enumerate(range(start_idx, end_idx)):
            s1_id = s1.iloc[row_idx]['entity_id']
            row_scores = sim_scores[i]
            
            candidate_indices = np.where(row_scores >= BLOCKING_THRESHOLD)[0]
            candidate_ids = s2_s3.iloc[candidate_indices]['entity_id'].tolist()
            
            match_indices = np.where(row_scores >= MATCH_THRESHOLD)[0]
            matched_ids = s2_s3.iloc[match_indices]['entity_id'].tolist()
            
            candidates.append({
                'source1_entity_id': s1_id, 
                'candidate_entity_ids': ",".join(candidate_ids)
            })
            matches.append({
                'source1_entity_id': s1_id, 
                'matched_entity_ids': ",".join(matched_ids)
            })
            
    return pd.DataFrame(candidates), pd.DataFrame(matches)

if __name__ == "__main__":
    print("Loading test data...")
    s1, s2, s3 = load_data()
    
    print("Executing pipeline...")
    candidate_df, matching_df = generate_matches(s1, s2, s3)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Saving outputs...")
    candidate_df.to_csv(os.path.join(OUTPUT_DIR, "candidate_pairs.tsv"), sep="\t", index=False)
    matching_df.to_csv(os.path.join(OUTPUT_DIR, "matching_results.tsv"), sep="\t", index=False)
    print("Done! Ensure you run the validation script next.")