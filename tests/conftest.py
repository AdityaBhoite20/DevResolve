# conftest.py - lets the tests import the files in the src folder
import sys  # to change the import path
from pathlib import Path  # to build file paths

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))  # add src/ at the front of the import path
