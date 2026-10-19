"""Resolve machine configuration before any model library is imported."""
import os
from pathlib import Path, PureWindowsPath

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str | Path) -> Path:
    raw = str(value).strip()
    if not raw:
        raise ValueError('Deployment paths must not be empty')
    if os.name != 'nt' and PureWindowsPath(raw).drive:
        raise ValueError(f'Windows path cannot be used on this operating system: {raw}')
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def initialize_environment():
    # An externally supplied value takes precedence over .env, then defaults.
    load_dotenv(PROJECT_ROOT / '.env', override=False, encoding='utf-8-sig')
    for key, default in {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'USE_TF': '0'}.items():
        os.environ.setdefault(key, default)
    runtime = resolve_path(os.environ.get('RUNTIME_DIR', './runtime'))
    os.environ['RUNTIME_DIR'] = str(runtime)
    defaults = {
        'HF_HOME': runtime / 'cache/huggingface',
        'TORCH_HOME': runtime / 'cache/torch',
        'APP_TEMP_DIR': runtime / 'tmp',
        'LOG_DIR': runtime / 'logs',
    }
    for key, default in defaults.items():
        os.environ[key] = str(resolve_path(os.environ.get(key, str(default))))
    for key, subdir in (('HF_HUB_CACHE', 'hub'), ('HF_XET_CACHE', 'xet')):
        default = Path(os.environ['HF_HOME']) / subdir
        os.environ[key] = str(resolve_path(os.environ.get(key, str(default))))
    model_path = resolve_path(os.environ.get('LAYA_MODEL_PATH', './models/laya-multilingual'))
    os.environ['LAYA_MODEL_PATH'] = str(model_path)
    # TEMP/TMP are inherited OS variables; APP_TEMP_DIR is the explicit app setting.
    os.environ['TEMP'] = os.environ['TMP'] = os.environ['APP_TEMP_DIR']
    for key in ('RUNTIME_DIR', 'HF_HOME', 'HF_HUB_CACHE', 'HF_XET_CACHE',
                'TORCH_HOME', 'APP_TEMP_DIR', 'LOG_DIR'):
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
    import tempfile
    tempfile.tempdir = None
