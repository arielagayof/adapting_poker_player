import sys
from pathlib import Path

# Ensure the project root is on sys.path so `import poker_adapt` works reliably.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
