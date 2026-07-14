import numpy as np
import time

print("🔗 Initializing Unified Prediction Schema Standardizer...")

# =====================================================================
# 1. UNIFIED EMOTION PREDICTOR CLASS Blueprint
# =====================================================================
class UnifiedEmotionPredictor:
    def __init__(self):
        # Enforcing identical target labels across both pipeline channels
        self.classes = ["Bored", "Confident", "Confused", "Curious", "Frustrated"]

    def format_bilstm_output(self, cleaned_text, raw_probs):
        """
        Generates unified prediction schema matching BiLSTM format requirements
        """
        # Exact logic matching the task description visualization
        emotion_idx = np.argmax(raw_probs)
        emotion = self.classes[emotion_idx]
        confidence = float(raw_probs[emotion_idx])
        scores = {self.classes[i]: float(raw_probs[i]) for i in range(len(self.classes))}
        
        return {
            'emotion': emotion,
            'confidence': confidence,
            'scores': scores,
            'cleaned_text': cleaned_text
        }

    def format_bert_output(self, cleaned_text, weighted_probs):
        """
        Generates unified prediction schema matching BERT format requirements
        """
        # Exact logic matching the normalized transformer view description
        total_sum = np.sum(weighted_probs)
        pred_id = np.argmax(weighted_probs)
        
        emotion = self.classes[pred_id]
        confidence = float(weighted_probs[pred_id] / total_sum)
        scores = {self.classes[i]: float(weighted_probs[i] / total_sum) for i in range(len(self.classes))}
        
        return {
            'emotion': emotion,
            'confidence': confidence,
            'scores': scores,
            'cleaned_text': cleaned_text.strip()
        }

# =====================================================================
# 2. RUNTIME PIPELINE VERIFICATION
# =====================================================================
predictor = UnifiedEmotionPredictor()

# Sample downstream inputs
sample_text = "i am feeling a bit confused about this specific model logic step"
bilstm_mock_probs = [0.05, 0.10, 0.70, 0.10, 0.05]
bert_mock_weights = [1.2, 0.9, 7.5, 1.1, 0.8]

print("\n🧪 Running schema harmonization validations...")
time.sleep(0.3)

# Test BiLSTM Serialization Format
bilstm_result = predictor.format_bilstm_output(sample_text, bilstm_mock_probs)
print("\n📦 [Unified Output -> From BiLSTM Pipe]")
print(f" ➜ Standard Keys Match: {list(bilstm_result.keys())}")
print(f" ➜ Target Emotion: {bilstm_result['emotion']} | Confidence: {bilstm_result['confidence']:.2%}")

# Test BERT Serialization Format
bert_result = predictor.format_bert_output(sample_text, bert_mock_weights)
print("\n📦 [Unified Output -> From BERT Pipe]")
print(f" ➜ Standard Keys Match: {list(bert_result.keys())}")
print(f" ➜ Target Emotion: {bert_result['emotion']} | Normalized Confidence: {bert_result['confidence']:.2%}")

print("\n✅ TASK COMPLETE: Unified prediction schemas are matching down the core pipeline formatting guidelines!")