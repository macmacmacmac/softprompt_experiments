import os
import argparse
import random
import sqlite3
import json
import nltk
from nltk.corpus import brown
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
import re
from tqdm import tqdm


SENTENCE_GENERATION_SYSTEM_PROMPT = """
You are an expert linguistic data generator. 
Generate diverse, natural-sounding sentences that describe or imply the Target Keyword. 

CRITICAL CONSTRAINTS: 
1. You MUST NOT use the target keyword, its root, or any direct derivations anywhere in the text.
2. Ensure the sentences cover different contexts (e.g., everyday life, technical, emotional, professional).
"""

# Define the exact JSON schema we want vLLM to force the model to follow
# This guarantees an object with a "sentences" array of strings.
JSON_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "sentences": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 10,
            "maxItems": 10
        }
    },
    "required": ["sentences"]
})

def get_safe_keywords(target_pool_size=15000):
    nltk.download('brown')
    
    # Hardcode the target classes from the InSPEcT paper
    forbidden_vocab = {
        # SST2 and SST5 Classes
        "positive", "negative", "terrible", "bad", "neutral", "good", "great",

        # AGNews Classes
        "world", "sports", "business", "technology"

        # Subj Classes
        "objective", "subjective", 

        # TREC Classes
        "abbreviation", "entity", "description", "human", "location", "number"
    }
    
    # Get standard nouns and adjectives
    tagged_words = [(word.lower(), tag) for word, tag in brown.tagged_words()]
    valid_words = [
        word for word, tag in tagged_words 
        if (tag.startswith('NN') or tag.startswith('JJ')) and word.isalpha()
    ]
    
    # Filter by frequency to ensure the words are common enough for an LLM to understand
    freq_dist = nltk.FreqDist(valid_words)
    
    safe_pool = []
    for word, _ in freq_dist.most_common():
        if word not in forbidden_vocab and len(word) > 3: # Skip tiny words
            safe_pool.append(word)
            
        if len(safe_pool) >= target_pool_size:
            break
            
    return safe_pool


def setup_database(db_path):
    """Initializes the SQLite schema designed for PyTorch dataloading speed."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS datasets (
            dataset_id INTEGER PRIMARY KEY,
            hard_prompt TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS keywords (
            keyword_id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER NOT NULL,
            keyword TEXT NOT NULL,
            label_index INTEGER NOT NULL,
            FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
        );
        CREATE TABLE IF NOT EXISTS sentences (
            sentence_id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER NOT NULL,
            keyword_id INTEGER NOT NULL,
            sentence TEXT NOT NULL,
            split TEXT NOT NULL,
            FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id),
            FOREIGN KEY (keyword_id) REFERENCES keywords(keyword_id)
        );
        CREATE INDEX IF NOT EXISTS idx_sentences_dataset_split 
        ON sentences(dataset_id, split);
    """)
    conn.commit()
    return conn, cursor


def run(args_list):
    exp_name = os.path.basename(__file__)
    print(
        "="*100, "\n", 
        f"\t\t\tRunning script: {exp_name}", "\n",
        "="*100,"\n"
    )

    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini_dataset_size", type=int, default=500)
    parser.add_argument("--num_of_datasets", type=int, default=5)   # TODO: Change this value, once testing of this script is done
    parser.add_argument("--save_directory", type=str, default="./datasets/mapper_classification_datasets")
    parser.add_argument("--db_name", type=str, default="synthetic_datasets.sqlite")
    args, _ = parser.parse_known_args(args_list)

    # Parse all the arguments into Variables
    # TEACHER_MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"
    TEACHER_MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct-AWQ"
    MINI_DATASET_SIZE = args.mini_dataset_size
    NUM_OF_DATASETS = args.num_of_datasets
    SAVE_DIRECTORY = args.save_directory
    DB_NAME = args.db_name

    # Other variables
    SENTENCES_PER_KEYWORD = MINI_DATASET_SIZE // 5
    CHUNK_SIZE = 10 # Ask LLM to generate CHUNK_SIZE sentences at a time to maintain high quality
    NUM_CHUNKS_PER_KEYWORD = max(1, SENTENCES_PER_KEYWORD // CHUNK_SIZE)

    # Delete the Dataset if it already exists
    db_path = os.path.join(SAVE_DIRECTORY, DB_NAME)
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # Setup the SQLite DB
    os.makedirs(SAVE_DIRECTORY, exist_ok=True)
    conn, cursor = setup_database(db_path)

    # Get all Safe Keywords (Nouns / Adjectives) from the Brown Corpus
    safe_keywords = get_safe_keywords()

    # Maintain a Maps of mini-dataset -> keywords
    dod_keyword_maps = []

    print("Initializing mini datasets in SQLite ...")

    # For each mini dataset
    for i in range(NUM_OF_DATASETS):
        # Randomly sample 5 keywords from the keywords pool
        keywords = tuple(random.sample(safe_keywords, 5))

        # Add the entry for dataset id and its associated keywords
        dod_keyword_maps.append({
            "dataset_id": i,
            "keywords": keywords
        })

        # Init a Hard Prompt for this mini dataset and insert it into the DB's datasets table
        hard_prompt = f"Classify the following sentence as: {', '.join(keywords)}"
        cursor.execute("INSERT INTO datasets (dataset_id, hard_prompt) VALUES (?, ?)", (i, hard_prompt))

        # Insert keywords and related data into the keywords table
        for label_idx, kw in enumerate(keywords):
            cursor.execute("INSERT INTO keywords (dataset_id, keyword, label_index) VALUES (?, ?, ?)", (i, kw, label_idx))

    # Commit all the inserts into the DB
    conn.commit()

    # Load Teacher Model using vLLM
    print(f"Loading {TEACHER_MODEL_NAME} into vLLM...")
    llm = LLM(
        model = TEACHER_MODEL_NAME,
        quantization="awq",
        tensor_parallel_size = 1,
        gpu_memory_utilization = 0.9 # Let vLLM use 90% of GPU VRAM for KV Cache
    )

    # Setup SamplingParams for the vLLM along with a guided json schema for guided decoding
    sampling_params = SamplingParams(
        temperature = 1.0, 
        max_tokens = 2000,
        structured_outputs = StructuredOutputsParams(json = JSON_SCHEMA)
    ) # TODO: check this

    # Generate Sentences for each dataset
    generation_tasks = []
    print("Creating prompts for sentence generation")
    for dataset in tqdm(dod_keyword_maps, desc="Creating Prompts"):
        dataset_id = dataset["dataset_id"]
        for kw in dataset["keywords"]:
            cursor.execute("SELECT keyword_id FROM keywords WHERE dataset_id = ? AND keyword = ?", (dataset_id, kw))
            keyword_id = cursor.fetchone()[0]

            # Chunking Strategy: Send multiple small requests instead of a massive one
            for _ in range(NUM_CHUNKS_PER_KEYWORD):

                # Construct messages to be sent to the LLM
                messages = [
                    {"role": "system", "content": SENTENCE_GENERATION_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Target Keyword: {kw}. Generate a JSON with {CHUNK_SIZE} unique sentences without using the keyword."}
                ]

                generation_tasks.append({
                    "dataset_id": dataset_id,
                    "keyword_id": keyword_id,
                    "keyword": kw,
                    "messages": messages
                })

    # Load all messages to be sent to the vLLM
    conversations = [task["messages"] for task in generation_tasks]

    # Execute the Batch Generation request
    print(f"Submitting {len(conversations)} generation tasks to vLLM...")
    outputs = llm.chat(messages = conversations, sampling_params = sampling_params)

    # Parse and Insert Results into the DB
    print("Parsing vLLM outputs and inserting into SQLite...")
    success_count = 0
    for task, output in tqdm(zip(generation_tasks, outputs), total=len(generation_tasks), desc = "Parsing LLM Responses and Inserting into DB"):
        generated_text = output.outputs[0].text

        try:

            # Clean the raw text of markdown ticks BEFORE regex parsing
            # cleaned_raw_text = generated_text.replace("```json", "").replace("```", "").strip()

            # # Find ALL blocks that look like JSON (matching everything between { and })
            # matches = re.findall(r'\{.*?\}', cleaned_raw_text, re.DOTALL)
            
            # if not matches:
            #     raise ValueError("No JSON object found in the model's output.")
                
            # clean_text = matches[-1]

            # The model might output markdown ticks (```json ... ```)
            # So we need to clean it, incase it exists
            # clean_text = generated_text.replace("```json", "").replace("```", "").strip()

            # Try to load the clean text as JSON
            # data = json.loads(clean_text)


            data = json.loads(generated_text)

            # Validate the 'sentences' key exists to prevent silent KeyErrors
            # if "sentences" not in data or not isinstance(data["sentences"], list):
            #     raise ValueError("JSON parsed successfully, but missing the 'sentences' array.")
            
            # Simple 80/20 Train/Test split assignment
            for idx, sentence in enumerate(data.get("sentences", [])):
                split = "test" if idx % 10 >= 8 else "train"
                    
                cursor.execute(
                    "INSERT INTO sentences (dataset_id, keyword_id, sentence, split) VALUES (?, ?, ?, ?)",
                    (task["dataset_id"], task["keyword_id"], sentence, split)
                )

            success_count += 1
            
        # Catch any JSON decoding errors
        except (json.JSONDecodeError) as e: 
            print(f"CRITICAL: Guided decoding failed for dataset {task['dataset_id']}, keyword '{task['keyword']}'. Error: {e}")
            continue

    print(f"Done! Successfully generated data for {success_count * CHUNK_SIZE} sentences across {NUM_OF_DATASETS} datasets.")

    # Commit all inserts and close the connection
    conn.commit()
    conn.close()