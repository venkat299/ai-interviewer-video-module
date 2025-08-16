import importlib
import sys
from pathlib import Path

import importlib.util

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))


def test_import_modules():
    if importlib.util.find_spec('numpy') is not None:
        importlib.import_module('src.pipeline.face_track')
    if importlib.util.find_spec('cv2') is not None and importlib.util.find_spec('numpy') is not None:
        importlib.import_module('src.pipeline.video_io')
        importlib.import_module('src.app.main')
