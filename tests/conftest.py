import sys
from pathlib import Path

# make `import nbclassifier` work wherever pytest is launched from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
