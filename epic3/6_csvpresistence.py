import os
import csv
import time
from datetime import datetime

print("💾 Initializing CSV Persistence & Cached Model Loading Module...")

# =====================================================================
# 1. CACHED MODEL LOADING EMULATION
# =====================================================================
# Simulating Streamlit's @st.cache_resource to load parameters once
_GLOBAL_MODEL_CACHE = {}

def load_models_cached():
    """
    Simulates cached resource registration. If models exist in the 
    global cache runtime dictionary, they load instantly.
    """
    global _GLOBAL_MODEL_CACHE
    if "loaded_engines" in _GLOBAL_MODEL_CACHE:
        print("⚡ [CACHE HIT] Fetching model backends straight from shared RAM...")
        return _GLOBAL_MODEL_CACHE["loaded_engines"]
    
    print("⏳ [CACHE MISS] First-time setup: Reading weights from local disk...")
    # Simulating standard file disk read overhead
    time.sleep(0.8)
    
    # Store dummy status flag in local memory cache
    _GLOBAL_MODEL_CACHE["loaded_engines"] = {
        "bilstm_status": "ready",
        "bert_status": "ready",
        "initialized_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    print("✅ Model weights securely registered into memory cache.")
    return _GLOBAL_MODEL_CACHE["loaded_engines"]

# =====================================================================
# 2. CSV PERSISTENCE LOGGER
# =====================================================================
CSV_FILE_PATH = "predictions_history.csv"

def log_prediction_to_csv(text, detected_emotion, confidence):
    """
    Persists prediction events into a structured spreadsheet log file.
    """
    file_exists = os.path.exists(CSV_FILE_PATH)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Open file in append mode safely using standard csv writer
    with open(CSV_FILE_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Write headers if the file is being newly initialized
        if not file_exists:
            writer.writerow(["Timestamp", "Input Text", "Detected Emotion", "Confidence Score"])
            
        writer.writerow([timestamp, text, detected_emotion, f"{confidence:.4f}"])
    
    print(f"📝 Appended session transaction to historical database: [{detected_emotion}] -> {CSV_FILE_PATH}")

# =====================================================================
# 3. RUNTIME PIPELINE VERIFICATION
# =====================================================================
print("\n--- Testing Cached Loader Component ---")
# First call should result in a cache miss (disk loading simulation)
engines_run1 = load_models_cached()
# Second call should result in a fast cache hit
engines_run2 = load_models_cached()

print("\n--- Testing CSV Persistence Logging ---")
# Simulated inputs to populate database rows
mock_predictions = [
    ("The code threw an unexpected exception during execution", "Frustrated", 0.8842),
    ("I clearly understand how this function optimization operates", "Confident", 0.9415),
    ("Let's look into the dataset formatting documentation", "Curious", 0.7630)
]

for text, emotion, conf in mock_predictions:
    log_prediction_to_csv(text, emotion, conf)

# Read it back to verify the persistence file was saved properly
print("\n🔍 Reading persisted history tracking database file contents:")
if os.path.exists(CSV_FILE_PATH):
    with open(CSV_FILE_PATH, mode="r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            print(f"  {line.strip()}")

print("\n✅ TASK COMPLETE: Local persistence engines and caching frameworks validated successfully!")