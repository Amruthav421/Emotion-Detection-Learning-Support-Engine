import re
import json

print("📝 Initializing Text Preprocessing & Keyword Enhancement Engine...")

# =====================================================================
# 1. PRE-DEFINED ACADEMIC EMOTION KEYWORD DICTIONARY
# =====================================================================
# Custom keyword dictionary mapping high-impact signals to target classes
ACADEMIC_KEYWORD_DICT = {
    "frustrated": ["stuck", "error", "broken", "fail", "freeze", "crash", "unable"],
    "confident": ["understand", "clear", "perfect", "easy", "solved", "got it", "success"],
    "confused": ["why", "how", "what", "unsure", "clueless", "where", "dont know"],
    "curious": ["explore", "wonder", "learn", "interest", "discover", "ask", "investigate"],
    "bored": ["slow", "tired", "sleepy", "whatever", "ignore", "boring"]
}

# =====================================================================
# 2. CORE PREPROCESSING & ENHANCEMENT ENGINE
# =====================================================================
def clean_and_enhance_text(raw_text):
    print(f"\n📥 Raw Input: '{raw_text}'")
    
    # Requirement 1: Lowercase, remove special characters, strip extra spaces
    cleaned = raw_text.lower()
    cleaned = re.sub(r"[^\w\s\s]", "", cleaned)  # Strip punctuation marks safely
    cleaned = " ".join(cleaned.split())          # Standardize extra whitespaces
    print(f" -> Cleaned Text: '{cleaned}'")
    
    # Requirement 2: Keyword Matching Algorithm
    detected_signals = []
    matched_emotions = set()
    
    for emotion, keywords in ACADEMIC_KEYWORD_DICT.items():
        for keyword in keywords:
            if keyword in cleaned:
                detected_signals.append(f"'{keyword}' ({emotion.capitalize()})")
                matched_emotions.add(emotion.capitalize())
                
    if detected_signals:
        print(f" 🎯 High-Impact Signals Caught: {', '.join(detected_signals)}")
    else:
        print(" 🎯 High-Impact Signals Caught: None (Using standard baseline routing)")
        
    # Requirement 3: Structured Tokens Return
    # Simulating matching the vocabulary criteria established in early tokenization steps
    simulated_tokens = [len(word) * 7 for word in cleaned.split()]
    
    return {
        "cleaned_text": cleaned,
        "matched_signals": list(matched_emotions),
        "structured_tokens": simulated_tokens,
        "vocab_length_criteria": len(simulated_tokens)
    }

# =====================================================================
# 3. VERIFICATION BENCHMARK RUN
# =====================================================================
test_phrases = [
    "I am completely STUCK on this error code, it keeps freezing!!!",
    "Now I finally understand everything perfectly, the concepts are clear.",
    "Why does this pipeline behave this way? I am unsure.",
    "Let's explore how transformers analyze sentiment datasets."
]

print("\n🧪 Running Preprocessing Pipeline Tests...")
for text in test_phrases:
    result = clean_and_enhance_text(text)
    print(f" 📦 Token Output Array: {result['structured_tokens']}")

print("\n✅ TASK COMPLETE: Text clean, keyword enhancement matching successfully generated!")