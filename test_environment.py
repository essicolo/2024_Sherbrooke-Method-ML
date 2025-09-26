#!/usr/bin/env python3
"""
Simple test to check if we can import required packages for the ML analysis
"""

try:
    import numpy as np
    print(f"✓ numpy {np.__version__}")
except ImportError:
    print("✗ numpy not available")

try:
    import polars as pl
    print(f"✓ polars {pl.__version__}")
except ImportError:
    print("✗ polars not available")

try:
    import sklearn
    print(f"✓ sklearn {sklearn.__version__}")
except ImportError:
    print("✗ sklearn not available")

try:
    import matplotlib
    print(f"✓ matplotlib {matplotlib.__version__}")
except ImportError:
    print("✗ matplotlib not available")

try:
    import scipy
    print(f"✓ scipy {scipy.__version__}")
except ImportError:
    print("✗ scipy not available")

print("\nEnvironment test complete!")
