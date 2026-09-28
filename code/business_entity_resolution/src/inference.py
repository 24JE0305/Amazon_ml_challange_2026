import pandas as pd
import os
import joblib
import time
import concurrent.futures
from rapidfuzz import fuzz
from cleaning_patterns import canonicalize_suffix, canonicalize_street, extract_landmark
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_PATH = os.path.join(BASE_DIR, "code", "business_entity_resolution", "xgb_matcher.pkl")

def string_similarity(s1, s2):
    if not s1 or not s2: return 0.0
    return fuzz.ratio(str(s1).lower(), str(s2).lower()) / 100.0

print("Loading trained XGBoost model...")
model = joblib.load(MODEL_PATH)

print("Loading Source datasets into ultra-light Memory...")
cols = ['entity_id', 'business_name', 'business_address', 'country']

# Loads directly into Tuples instead of heavy Pandas dicts to save massive amounts of RAM
def load_to_dict(files):
    d = {}
    for f in files:
        df = pd.read_csv(f, sep="\t", usecols=cols, dtype=str).fillna("")
        for row in df.itertuples(index=False):
            d[row.entity_id] = (row.business_name, row.business_address, row.country)
        del df
    return d

s1_dict = load_to_dict([os.path.join(TEST_DIR, "test_source1.tsv")])
s2_s3_dict = load_to_dict([os.path.join(TEST_DIR, "test_source2.tsv"), os.path.join(TEST_DIR, "test_source3.tsv")])

candidates_file = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
out_file = os.path.join(OUTPUT_DIR, "matching_results.tsv")

# Global caches completely eliminate redundant Regex calculations
s1_cache = {}
s2_s3_cache = {}

def process_chunk(chunk_df):
    chunk_df = chunk_df.fillna("")
    batch_features = []
    batch_meta = []
    
    s1_ids = chunk_df['source1_entity_id'].tolist()
    cands_strs = chunk_df['candidate_entity_ids'].tolist()
    results_map = {s1_id: [] for s1_id in s1_ids}
    
    for s1_id, cands_str in zip(s1_ids, cands_strs):
        if not cands_str or s1_id not in s1_dict:
            continue
            
        cands = [c for c in cands_str.split(',') if c]
        if not cands: continue
        
        if s1_id not in s1_cache:
            name, addr, country = s1_dict[s1_id]
            c_name = canonicalize_suffix(name)
            c_addr, l1 = extract_landmark(canonicalize_street(addr))
            s1_cache[s1_id] = (c_name, c_addr, l1, country)
        
        d1_name, d1_addr, l1, d1_country = s1_cache[s1_id]
        
        for c_id in cands:
            if c_id not in s2_s3_dict: continue
            
            if c_id not in s2_s3_cache:
                name2, addr2, country2 = s2_s3_dict[c_id]
                c_name2 = canonicalize_suffix(name2)
                c_addr2, l2 = extract_landmark(canonicalize_street(addr2))
                s2_s3_cache[c_id] = (c_name2, c_addr2, l2, country2)
                
            d2_name, d2_addr, l2, d2_country = s2_s3_cache[c_id]
            
            batch_features.append({
                'name_sim': string_similarity(d1_name, d2_name),
                'addr_sim': string_similarity(d1_addr, d2_addr),
                'country_match': int(d1_country == d2_country),
                'landmark_match': int(l1 == l2 and l1 is not None)
            })
            batch_meta.append((s1_id, c_id))
    
    if batch_features:
        X_infer = pd.DataFrame(batch_features)
        preds = model.predict(X_infer)
        for (s1_id, c_id), p in zip(batch_meta, preds):
            if p == 1:
                results_map[s1_id].append(c_id)
                
    formatted_rows = []
    for s1_id, cands_str in zip(s1_ids, cands_strs):
        if cands_str and s1_id in s1_dict:
            matched_str = ",".join(results_map.get(s1_id, []))
            formatted_rows.append(f"{s1_id}\t{matched_str}\n")
        else:
            formatted_rows.append(f"{s1_id}\t\n")
            
    return formatted_rows, len(chunk_df)

print("Streaming Candidate Pairs (Bypassing RAM limits)...")
start_time = time.time()
processed = 0

with open(out_file, "w", encoding="utf-8") as f_out:
    f_out.write("source1_entity_id\tmatched_entity_ids\n")
    
    # Safely stream chunks from the hard drive
    chunk_iter = pd.read_csv(candidates_file, sep="\t", dtype=str, chunksize=15000)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = set()
        
        # Only allow 8 chunks in memory at once to prevent thrashing
        for chunk in chunk_iter:
            if len(futures) >= 8:
                done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    formatted_rows, count = future.result()
                    f_out.writelines(formatted_rows)
                    processed += count
                    print(f"Processed {processed} rows | Elapsed: {time.time() - start_time:.1f}s")
            
            futures.add(executor.submit(process_chunk, chunk))
            
        for future in concurrent.futures.as_completed(futures):
            formatted_rows, count = future.result()
            f_out.writelines(formatted_rows)
            processed += count
            print(f"Processed {processed} rows | Elapsed: {time.time() - start_time:.1f}s")

print("Pipeline complete! Results saved.")