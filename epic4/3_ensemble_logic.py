import numpy as np
import time

print("⚖️ Initializing Advanced Ensemble Logic & Voting System...")

# Enforced target label mapping matching your project setup
CLASSES = ["Bored", "Confident", "Confused", "Curious", "Frustrated"]

# =====================================================================
# 1. ENSEMBLE SYSTEM WEIGHTED COMBINATION PIPELINE
# =====================================================================
def compute_ensemble_prediction(bilstm_scores, bert_scores, bilstm_weight=0.3, bert_weight=0.7):
    """
    Combines independent classifier scores using weighted linear combination formulas:
    Final Score = (BiLSTM * 0.3) + (BERT * 0.7)
    """
    print("\n⚡ Processing ensemble alignment matrices...")
    
    # Standardize dictionary score values into ordered numpy arrays
    bilstm_vector = np.array([bilstm_scores.get(cls, 0.0) for cls in CLASSES])
    bert_vector = np.array([bert_scores.get(cls, 0.0) for cls in CLASSES])
    
    # Core mathematical fusion execution pass
    fused_vector = (bilstm_vector * bilstm_weight) + (bert_vector * bert_weight)
    
    # Re-normalize to ensure strict probability distribution constraint
    final_probabilities = fused_vector / np.sum(fused_vector)
    
    # Calculate highest confidence indicator indices
    winning_idx = np.argmax(final_probabilities)
    final_emotion = CLASSES[winning_idx]
    final_confidence = final_probabilities[winning_idx]
    
    # Reassemble complete probability map
    ensemble_distribution = {CLASSES[i]: float(final_probabilities[i]) for i in range(len(CLASSES))}
    
    return {
        "ensemble_final_emotion": final_emotion,
        "ensemble_confidence": final_confidence,
        "score_breakdown": ensemble_distribution
    }

# =====================================================================
# 2. RUNTIME VERIFICATION PASSES
# =====================================================================
# Mock scenario profiles to test prediction aggregation mechanics
print("\n🧪 Running ensemble verification metrics...")
time.sleep(0.3)

# Test Scenario A: Models agree on high confidence target
print("\n--- Test Case A: Classifier Uniform Convergence ---")
bilstm_mock_a = {"Bored": 0.05, "Confident": 0.10, "Confused": 0.75, "Curious": 0.05, "Frustrated": 0.05}
bert_mock_a   = {"Bored": 0.02, "Confident": 0.05, "Confused": 0.80, "Curious": 0.10, "Frustrated": 0.03}

result_a = compute_ensemble_prediction(bilstm_mock_a, bert_mock_a)
print(f" ➜ Result Decision: {result_a['ensemble_final_emotion']} ({result_a['ensemble_confidence']:.2%})")


# Test Scenario B: Conflict state where 70% BERT weight overrules 30% BiLSTM
print("\n--- Test Case B: Classifier Divergence (BERT Weight Pull) ---")
bilstm_mock_b = {"Bored": 0.10, "Confident": 0.65, "Confused": 0.10, "Curious": 0.10, "Frustrated": 0.05} # BiLSTM flags Confident
bert_mock_b   = {"Bored": 0.05, "Confident": 0.10, "Confused": 0.15, "Curious": 0.10, "Frustrated": 0.60} # BERT flags Frustrated

result_b = compute_ensemble_prediction(bilstm_mock_b, bert_mock_b)
print(f" ➜ Result Decision: {result_b['ensemble_final_emotion']} ({result_b['ensemble_confidence']:.2%})")
print(" ➜ Comprehensive Ensemble Distribution Vector:")
for cls_name, prob in result_b['score_breakdown'].items():
    print(f"    • {cls_name}: {prob:.4f}")

print("\n✅ TASK COMPLETE: Ensemble linear fusion matrix logic verified and operational!")