import hashlib
import json

import pytest

from tools.doctor import verify_model
from app.detector import REQUIRED_MODEL_FILES


def test_corrupted_weight_is_reported(tmp_path):
    hashes = {}
    for name in REQUIRED_MODEL_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'initial contents')
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / 'download-manifest.json').write_text(json.dumps({'sha256': hashes}), encoding='utf-8')
    assert verify_model(tmp_path)['verified_files'] == len(REQUIRED_MODEL_FILES)
    (tmp_path / 'model.safetensors').write_bytes(b'corrupted contents')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        verify_model(tmp_path)


def test_complete_model_without_manifest_is_explicitly_unverified(tmp_path):
    for name in REQUIRED_MODEL_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'contents')
    assert verify_model(tmp_path)['hash_verification'] == 'not_checked_no_manifest'
    with pytest.raises(FileNotFoundError, match='manifest is required'):
        verify_model(tmp_path, require_hashes=True)
