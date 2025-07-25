import os
import sys

print("--- Python Executable ---")
print(sys.executable)
print("\n--- sys.path ---")
for p in sys.path:
    print(p)
print("\n--- LD_LIBRARY_PATH ---")
ld_path = os.environ.get('LD_LIBRARY_PATH', 'Not Set')
for p in ld_path.split(':'):
    print(p)

# 让我们尝试加载 torch，看看它会给出什么错误
print("\n--- Attempting to import torch ---")
try:
    import torch
    print("Successfully imported torch!")
    print(f"Torch version: {torch.__version__}")
    print(f"Torch file: {torch.__file__}")
except ImportError as e:
    print(f"Failed to import torch: {e}")