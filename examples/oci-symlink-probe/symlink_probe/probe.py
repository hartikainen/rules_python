import json
import os
import sys
from pathlib import Path

import absl
import numpy as np

venv = Path(sys.prefix)
python_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
site = venv / "lib" / python_dir / "site-packages"
assert Path(sys.executable).is_symlink(), sys.executable
assert (site / "absl").is_symlink(), site
assert Path(absl.__file__).is_file()
assert np.dot([1, 2], [3, 4]) == 11

runfiles = Path(__file__).parents[2]
link_count = 0
for root, directories, files in os.walk(runfiles):
    for name in directories + files:
        path = Path(root) / name
        if path.is_symlink():
            assert path.exists(), path
            assert not os.path.isabs(os.readlink(path)), path
            link_count += 1

print(json.dumps({
    "executable": sys.executable,
    "interpreter_link": os.readlink(sys.executable),
    "absl_link": os.readlink(site / "absl"),
    "numpy": np.__version__,
    "verified_symlinks": link_count,
}))
