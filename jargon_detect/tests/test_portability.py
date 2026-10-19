"""Exercise fresh process imports from a relocated project and unrelated cwd."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from app.config import PROJECT_ROOT

APP_ENV = ('PROJECT_NAME', 'LAYA_MODEL_PATH', 'LAYA_DEVICE', 'API_HOST', 'API_PORT',
           'MAX_TEXT_LENGTH', 'LAYA_MAX_TOKENS', 'RUNTIME_DIR', 'APP_TEMP_DIR', 'LOG_DIR',
           'HF_HOME', 'HF_HUB_CACHE', 'HF_XET_CACHE', 'TORCH_HOME',
           'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'USE_TF')


def probe(project, cwd, overrides=None):
    env = os.environ.copy()
    for name in APP_ENV:
        env.pop(name, None)
    env.update(overrides or {})
    code = '''import sys, json, os, tempfile
sys.path.insert(0, sys.argv[1])
import app
before = 'torch' not in sys.modules and 'transformers' not in sys.modules
from app.config import Settings
import huggingface_hub.constants as hub
s = Settings()
print(json.dumps({'model': str(s.laya_model_path), 'port': s.api_port,
    'hf_home': hub.HF_HOME, 'offline': hub.HF_HUB_OFFLINE,
    'before': before, 'temp': tempfile.gettempdir()}))
'''
    result = subprocess.run([sys.executable, '-c', code, str(project)], cwd=cwd,
                            env=env, capture_output=True, text=True, encoding='utf-8', timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.fixture
def relocated(tmp_path):
    project = tmp_path / '交付目录 with spaces'
    project.mkdir()
    shutil.copytree(PROJECT_ROOT / 'app', project / 'app', ignore=shutil.ignore_patterns('__pycache__'))
    outside = tmp_path / 'unrelated cwd'
    outside.mkdir()
    return project, outside


def test_relocated_defaults_without_env_file(relocated):
    project, outside = relocated
    result = probe(project, outside)
    assert Path(result['model']) == project / 'models/laya-multilingual'
    assert Path(result['hf_home']) == project / 'runtime/cache/huggingface'
    assert Path(result['temp']) == project / 'runtime/tmp'
    assert result['offline'] and result['before']


def test_dotenv_paths_are_relative_to_project_and_loaded_early(relocated):
    project, outside = relocated
    (project / '.env').write_text('LAYA_MODEL_PATH=assets/model\nAPI_PORT=8124\n'
        'HF_HOME=cache/hub-home\nAPP_TEMP_DIR=work/tmp\n', encoding='utf-8-sig')
    result = probe(project, outside)
    assert Path(result['model']) == project / 'assets/model'
    assert Path(result['hf_home']) == project / 'cache/hub-home'
    assert Path(result['temp']) == project / 'work/tmp'
    assert result['port'] == 8124 and result['before']


def test_external_values_override_dotenv_including_temp(relocated):
    project, outside = relocated
    model = outside / 'external model'
    (project / '.env').write_text('LAYA_MODEL_PATH=wrong\nAPI_PORT=8124\n'
                                'HF_HUB_OFFLINE=1\nAPP_TEMP_DIR=wrong-tmp\n', encoding='utf-8')
    result = probe(project, outside, {'LAYA_MODEL_PATH': str(model), 'API_PORT': '8125',
                                    'HF_HUB_OFFLINE': '0', 'APP_TEMP_DIR': 'configured-tmp'})
    assert Path(result['model']) == model
    assert result['port'] == 8125 and not result['offline']
    assert Path(result['temp']) == project / 'configured-tmp'
