"""Validate deployment and optionally execute a real local prediction."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PureWindowsPath
import shutil
import sys
import tempfile

from app.bootstrap import PROJECT_ROOT
from app.config import Settings
from app.detector import LayaDetector, require_local_files


def verify_model(model: Path, require_hashes=False):
    require_local_files(model)
    manifest_path = model / 'download-manifest.json'
    if not manifest_path.is_file():
        if require_hashes:
            raise FileNotFoundError('Model hash manifest is required: download-manifest.json')
        return {'verified_files': 0, 'hash_verification': 'not_checked_no_manifest'}
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    hashes = manifest.get('sha256', {})
    if not hashes or 'model.safetensors' not in hashes:
        raise ValueError('Model manifest must contain weight SHA256')
    for name, expected in hashes.items():
        relative = Path(name)
        path = (model / relative).resolve()
        if relative.is_absolute() or PureWindowsPath(name).drive or not path.is_relative_to(model.resolve()):
            raise ValueError(f'Invalid manifest path: {name}')
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected:
            raise ValueError(f'Model SHA256 mismatch: {name}')
    return {'hash_verification': 'passed', 'repo_id': manifest.get('repo_id'), 'revision': manifest.get('revision'),
            'verified_files': len(hashes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--load-model', action='store_true', help='Also warm up and run real inference')
    parser.add_argument('--require-hashes', action='store_true', help='Require download-manifest.json and verify its hashes')
    args = parser.parse_args()
    detector = None
    try:
        if sys.version_info[:2] != (3, 11):
            raise RuntimeError('This release is validated with Python 3.11')
        settings = Settings()
        for key in ('RUNTIME_DIR', 'HF_HOME', 'HF_HUB_CACHE', 'HF_XET_CACHE',
                    'TORCH_HOME', 'APP_TEMP_DIR', 'LOG_DIR'):
            with tempfile.TemporaryFile(dir=os.environ[key]) as probe:
                probe.write(b'deployment write check')
                probe.flush()
        report = {'status': 'ok', 'python': sys.version.split()[0],
                  'project_root': str(PROJECT_ROOT), 'model_path': str(settings.laya_model_path),
                  'runtime_dir': os.environ['RUNTIME_DIR'],
                  'free_runtime_gib': round(shutil.disk_usage(os.environ['RUNTIME_DIR']).free / 1024**3, 2),
                  'model': verify_model(settings.laya_model_path, require_hashes=args.require_hashes)}
        versions = {}
        for package in ('torch', 'laya', 'transformers', 'huggingface-hub', 'fastapi', 'uvicorn',
                        'safetensors', 'numpy', 'pydantic', 'pydantic-settings', 'python-dotenv'):
            versions[package] = importlib.metadata.version(package)
        report['versions'] = versions
        import torch
        report['cuda_available'] = torch.cuda.is_available()
        report['torch_cuda'] = torch.version.cuda
        if settings.laya_device == 'cuda' and not report['cuda_available']:
            raise RuntimeError('LAYA_DEVICE=cuda but CUDA is unavailable')
        if report['cuda_available'] and settings.laya_device != 'cpu':
            report['gpu'] = torch.cuda.get_device_name(0)
            # Availability alone does not verify supported CUDA kernels.
            torch.ones(1, device='cuda').add_(1).cpu()
        if args.load_model:
            detector = LayaDetector(settings)
            detector.load()
            report['prediction'] = detector.detect('我今天在图书馆学习。').model_dump()
            report['device'] = detector.device
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({'status': 'error', 'error': f'{type(exc).__name__}: {exc}'}, ensure_ascii=False))
        return 1
    finally:
        if detector is not None:
            detector.close()


if __name__ == '__main__':
    sys.exit(main())
