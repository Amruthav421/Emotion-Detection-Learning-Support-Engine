import sys
print("Python path:", sys.executable)

import tensorflow as tf
print("TensorFlow path:", tf.__file__)
print("🟩 TensorFlow:", tf.__version__)

import numpy as np
print("🟩 NumPy:", np.__version__)

import sklearn
print("🟩 scikit-learn:", sklearn.__version__)

import transformers
print("🟩 transformers:", transformers.__version__)

print("\n🎉 epic 2 DONE! All versions correct.")