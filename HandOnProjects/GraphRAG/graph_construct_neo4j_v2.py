import json
import os
import tiktoken
import numpy as np
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI, LengthFinishReasonError
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
import instructor
from pydantic import BaseModel, Field
from typing import List

# ================= CONFIGURATION =================
# 1. API KEYS
DEEPSEEK_API_KEY = "#GetYourOwnKey"  
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
OPENAI_API_KEY = '#GetYourOwnKey'
# 2. TUNING PARAMETERS
# Drastically reduced to prevent JSON explosion. 
# 600 tokens input -> ~3000 tokens output (Safe for 8k limit)
MAX_TOKENS_PER_CHUNK = 600  
CHUNK_OVERLAP = 100         

EMBEDDING_MODEL = "text-embedding-3-large"
VECTOR_DIMENSION = 1536

# File Paths
INPUT_FILE = r"graph_source_v3.jsonl" # machine language NOT human interpretable
OUTPUT_FILE = "graph_triplets_v6.json"

# ================= CLIENT SETUP =================
extract_client = instructor.patch(
    OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL),
    mode=instructor.Mode.JSON
)

embed_client = OpenAI(api_key=OPENAI_API_KEY)

# ================= ONTOLOGY =================
REF_ENTITY_TYPES = [
    "Component", "Physics_Principle", "Artifact", "Application", 
    "Metric", "Technology", "Anatomy", "Software_Concept"
]

REF_RELATIONS = [
    "IS_A", "PART_OF", "USED_FOR", "CAUSES", 
    "MEASURES", "AFFECTS", "LOCATED_AT", "HAS_PROPERTY"
]

# ================= SCHEMA =================
class Triplet(BaseModel):
    head: str
    head_type: str
    tail: str
    tail_type: str
    relation: str

class KnowledgeGraph(BaseModel):
    triplets: List[Triplet] = Field(default_factory=list)

# ================= TEXT UTILS =================
def get_tokens(text):
    enc = tiktoken.get_encoding("cl100k_base")
    return enc.encode(text)

def decode_tokens(tokens):
    enc = tiktoken.get_encoding("cl100k_base")
    return enc.decode(tokens)

def normalize_string(value, reference_list, threshold=80):
    if not value: return "Unknown"
    # Return best match from ontology or Title Case original
    best_match, score = process.extractOne(value, reference_list)
    return best_match if score >= threshold else value.title()

# ================= EXTRACTION LOGIC (WITH RECURSION) =================

def extract_with_recursion(text_segment, source_id, depth=0):
    """
    Tries to extract. If context length error occurs, 
    splits text in half and recurses.
    """
    # Safety break
    if depth > 3: 
        print(f"  [Warn] Max recursion depth reached for {source_id}")
        return []

    try:
        return _call_api(text_segment, source_id)
        
    except Exception as e:
        # Check if error is due to length/token limit
        is_length_error = "length" in str(e) or "max_tokens" in str(e)
        
        if is_length_error:
            print(f"  [Info] Chunk {source_id} too large (Output Limit). Splitting...")
            
            # Split text in half
            tokens = get_tokens(text_segment)
            mid = len(tokens) // 2
            part1 = decode_tokens(tokens[:mid])
            part2 = decode_tokens(tokens[mid:])
            
            # Recursive call
            res1 = extract_with_recursion(part1, f"{source_id}_L", depth+1)
            res2 = extract_with_recursion(part2, f"{source_id}_R", depth+1)
            return res1 + res2
        else:
            print(f"  [Error] Failed extraction {source_id}: {e}")
            return []

def _call_api(text, source_id):
    prompt = f"""
    Extract knowledge triplets from the text.
    Target Ontology: {json.dumps(REF_ENTITY_TYPES)}
    Relations: {json.dumps(REF_RELATIONS)}
    Text: "{text}"
    """
    
    resp = extract_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        response_model=KnowledgeGraph,
        max_retries=1, # We handle retries via recursion
        temperature=0.0
    )
    
    # Normalize results immediately
    clean_triplets = []
    for t in resp.triplets:
        clean_triplets.append({
            "head": t.head.strip(),
            "head_type": normalize_string(t.head_type, REF_ENTITY_TYPES),
            "tail": t.tail.strip(),
            "tail_type": normalize_string(t.tail_type, REF_ENTITY_TYPES),
            "relation": normalize_string(t.relation, REF_RELATIONS),
            "source_id": source_id
        })
    return clean_triplets

# ================= PRE-PROCESSING =================
def prepare_chunks_sliding_window(input_file):
    """Generates (text, id) tuples with strict sizing."""
    tasks = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                raw_text = data.get('content', '')
                base_id = data.get('chunk_id', 'unknown')
                
                # Sliding window logic
                all_tokens = get_tokens(raw_text)
                total_tokens = len(all_tokens)
                
                # If small enough, keep as is
                if total_tokens <= MAX_TOKENS_PER_CHUNK:
                    tasks.append((raw_text, base_id))
                    continue
                
                # Else slide
                stride = MAX_TOKENS_PER_CHUNK - CHUNK_OVERLAP
                for i in range(0, total_tokens, stride):
                    window = all_tokens[i : i + MAX_TOKENS_PER_CHUNK]
                    window_text = decode_tokens(window)
                    sub_id = f"{base_id}_w{i}"
                    tasks.append((window_text, sub_id))
                    
            except json.JSONDecodeError:
                continue
    return tasks

# ================= ENTITY RESOLUTION =================
class EntityResolver:
    def __init__(self, triplets):
        self.triplets = triplets
        self.entities = list(set(
            [t['head'] for t in triplets] + [t['tail'] for t in triplets]
        ))

    def resolve(self):
        print(f"Resolving {len(self.entities)} entities via OpenAI...")
        
        # 1. Get Embeddings
        embeddings = []
        batch_size = 200 # OpenAI allows large batches
        
        for i in tqdm(range(0, len(self.entities), batch_size)):
            batch = self.entities[i:i+batch_size]
            try:
                # Sanitize newlines
                clean_batch = [t.replace("\n", " ") for t in batch]
                res = embed_client.embeddings.create(
                    input=clean_batch,
                    model=EMBEDDING_MODEL,
                    dimensions=VECTOR_DIMENSION
                )
                # Ensure order
                res_data = sorted(res.data, key=lambda x: x.index)
                embeddings.extend([d.embedding for d in res_data])
            except Exception as e:
                print(f"Embedding failed: {e}")
                # Fallback: Zero vectors to maintain index alignment
                embeddings.extend([np.zeros(VECTOR_DIMENSION) for _ in batch])

        if not embeddings: return self.triplets

        # 2. Similarity Clustering
        matrix = np.array(embeddings)
        sim_matrix = cosine_similarity(matrix)
        
        mapping = {e: e for e in self.entities}
        processed = set()
        
        # Greedy Merge High Similarity
        for i in range(len(self.entities)):
            if i in processed: continue
            
            for j in range(i + 1, len(self.entities)):
                if j in processed: continue
                
                # Strict threshold for High Quality merging
                if sim_matrix[i][j] > 0.94: 
                    root = self.entities[i]
                    duplicate = self.entities[j]
                    
                    # Logic: Prefer UpperCase or Longer name (usually more specific)
                    if len(duplicate) > len(root):
                        root, duplicate = duplicate, root
                        
                    mapping[duplicate] = root
                    processed.add(j)

        # 3. Rewrite Triplets
        new_triplets = []
        for t in self.triplets:
            new_t = t.copy()
            new_t['head'] = mapping[t['head']]
            new_t['tail'] = mapping[t['tail']]
            if new_t['head'] != new_t['tail']:
                new_triplets.append(new_t)
                
        return new_triplets

# ================= MAIN =================
def main():
    # 1. Prepare Data
    print("--- 1. Chunking Source Data ---")
    tasks = prepare_chunks_sliding_window(INPUT_FILE)
    print(f"Prepared {len(tasks)} processing tasks.")

    # 2. Extract
    print("--- 2. Extraction (DeepSeek) ---")
    raw_triplets = []
    
    # Max workers depends on your rate limit
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(extract_with_recursion, text, cid): cid 
            for text, cid in tasks
        }
        
        for future in tqdm(as_completed(futures), total=len(tasks)):
            result = future.result()
            if result:
                raw_triplets.extend(result)
                
    print(f"Extracted {len(raw_triplets)} triplets.")

    # 3. Resolve
    print("--- 3. Resolution (OpenAI) ---")
    resolver = EntityResolver(raw_triplets)
    final_triplets = resolver.resolve()
    print(f"Resolved to {len(final_triplets)} unique triplets.")

    # 4. Save
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_triplets, f, indent=2)
    print(f"Done! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()