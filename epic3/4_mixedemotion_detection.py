import time

print("🎯 Initializing Mixed-Emotion Detection Engine...")

# =====================================================================
# 1. CORE MIXED-EMOTION EXTRACTION ALGORITHM
# =====================================================================
# Replicating the exact workspace logical blueprint for threshold extraction
def get_mixed_emotions(scores, threshold=0.15):
    """
    Analyzes emotion score distributions and flags secondary attributes 
    climbing above the 15% confidence boundary criteria.
    """
    # Sorting items descending based on probability metric values
    sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    primary = sorted_emotions[0]
    mixed = [primary]  # Primary is always included
    
    # Check all remaining candidate emotions against the 15% cutoff threshold
    for emotion, score in sorted_emotions[1:]:
        if score >= threshold:
            mixed.append((emotion, score))
            
    # Returns the list of tuples matching detection constraints
    return mixed

# =====================================================================
# 2. VERIFICATION RUN BENCHMARKS
# =====================================================================
# Mock probability distributions mimicking realistic outputs from your models
simulated_model_outputs = [
    {
        "case": "Clear Single Emotion",
        "scores": {"Bored": 0.05, "Confident": 0.82, "Confused": 0.03, "Curious": 0.05, "Frustrated": 0.05}
    },
    {
        "case": "Dual Mixed State (Confused + Curious)",
        "scores": {"Bored": 0.08, "Confident": 0.02, "Confused": 0.55, "Curious": 0.25, "Frustrated": 0.10}
    },
    {
        "case": "Complex Layered State (Frustrated + Confused + Bored)",
        "scores": {"Bored": 0.16, "Confident": 0.04, "Confused": 0.20, "Curious": 0.10, "Frustrated": 0.50}
    }
]

print("\n🚀 Evaluating score matrices through mixed-emotion threshold loops...")
time.sleep(0.3)

for item in simulated_model_outputs:
    print(f"\n📊 Run Context: {item['case']}")
    detected_mix = get_mixed_emotions(item["scores"], threshold=0.15)
    
    # UI Text string formatter emulation
    if len(detected_mix) > 1:
        combination_string = " + ".join([em for em, _ in detected_mix])
        print(f" ➜ Result Status: Mixed State Caught! ({combination_string})")
    else:
        print(f" ➜ Result Status: Stable Single-Label Category ({detected_mix[0][0]})")
        
    print(" ➜ Captured Profile Components:")
    for rank, (emotion, score) in enumerate(detected_mix):
        tag = "Primary" if rank == 0 else "Secondary"
        print(f"    • [{tag}] {emotion}: {score:.2%}")

print("\n✅ TASK COMPLETE: Multi-label classification layer fully functional at 15% constraints!")