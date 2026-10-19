import copy
import logging
import math
from numbers import Real
from threading import Lock
from typing import Any

from app.config import Settings
from app.questions import CHOICE_LABEL_MAP, JARGON_QUESTIONS
from app.schemas import DetectionData

logger = logging.getLogger(__name__)


class ModelNotReadyError(RuntimeError):
    pass


class InferenceError(RuntimeError):
    pass


class ChoiceFormatError(InferenceError):
    pass


class TextTooLongError(ValueError):
    pass


REQUIRED_MODEL_FILES = ('rl_agent_config.json', 'model.safetensors',
                        'tokenizer/tokenizer.json', 'tokenizer/tokenizer_config.json', 'encoder/config.json')


def require_local_files(path):
    missing = [name for name in REQUIRED_MODEL_FILES if not (path / name).is_file() or (path / name).stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f'本地模型不完整：{path}；缺少 {missing}。请配置 LAYA_MODEL_PATH 指向完整模型目录')


def parse_choice(result: Any) -> DetectionData:
    try:
        answer = result['answers']['jargon_detection']
        if answer.get('type') != 'choice':
            raise ValueError('type must be choice')
        choice = answer['choice']
        probabilities = answer['probabilities']
        if not isinstance(choice, str) or choice not in CHOICE_LABEL_MAP or not isinstance(probabilities, dict) or set(probabilities) != set(CHOICE_LABEL_MAP):
            raise ValueError('unexpected choice or probability keys')
        for value in probabilities.values():
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('probabilities must be finite numbers in [0,1]')
        # Laya rounds each probability to four decimals. Never renormalize its output.
        if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=0.001):
            raise ValueError('probabilities do not sum to one')
        if probabilities[choice] < max(probabilities.values()):
            raise ValueError('choice disagrees with probabilities')
        confidence = answer.get('answer_confidence')
        if confidence is not None and (isinstance(confidence, bool) or not isinstance(confidence, Real)
                                       or not math.isfinite(confidence) or not 0 <= confidence <= 1):
            raise ValueError('invalid answer_confidence')
        business_choice = CHOICE_LABEL_MAP[choice]
        business_probabilities = {label: probabilities[option] for option, label in CHOICE_LABEL_MAP.items()}
        return DetectionData(is_jargon=business_choice == 'jargon', label='黑话' if business_choice == 'jargon' else '非黑话',
                             choice=business_choice, probabilities=business_probabilities, answer_confidence=confidence)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ChoiceFormatError(f'Laya Choice 输出格式异常：{exc}') from exc


class LayaDetector:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.agent = None
        self.load_error: str | None = None
        self._lock = Lock()
        self.last_usage: dict = {}

    @property
    def ready(self):
        return self.agent is not None

    @property
    def device(self):
        return str(self.agent.device) if self.agent is not None else None

    def load(self):
        with self._lock:
            if self.ready:
                return
            try:
                require_local_files(self.settings.laya_model_path)
                import torch
                import laya
                device = self.settings.laya_device
                if device == 'auto':
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                elif device == 'cuda' and not torch.cuda.is_available():
                    raise RuntimeError('已指定 CUDA，但 PyTorch 检测不到可用 CUDA 设备')
                agent = laya.load(str(self.settings.laya_model_path), device=device, backend='eager')
                # Warm up the actual choice path; readiness means inference initialization succeeded.
                probe = agent.predict({'text': '模型启动检查。'}, copy.deepcopy(JARGON_QUESTIONS),
                                      max_len=self.settings.laya_max_tokens)
                parse_choice(probe)
                self.agent = agent
                self.load_error = None
                logger.info('Laya loaded from %s on %s', self.settings.laya_model_path, self.device)
            except Exception as exc:
                self.load_error = f'{type(exc).__name__}: {exc}'
                raise

    def detect(self, text: str) -> DetectionData:
        if not isinstance(text, str) or not text.strip():
            raise ValueError('文本必须是非空字符串')
        if len(text) > self.settings.max_text_length:
            raise TextTooLongError(f'文本超过 {self.settings.max_text_length} 字符限制，未执行截断')
        with self._lock:
            if not self.ready:
                raise ModelNotReadyError('模型尚未就绪；请检查服务日志和本地模型文件')
            try:
                result = self.agent.predict({'text': text}, copy.deepcopy(JARGON_QUESTIONS),
                                            max_len=self.settings.laya_max_tokens)
                self.last_usage = result.get('usage', {})
                if self.last_usage.get('truncated') or self.last_usage.get('state_tokens_dropped', 0):
                    raise TextTooLongError('文本超过模型实际 token 容量；请缩短文本。截断结果未返回')
                return parse_choice(result)
            except (ChoiceFormatError, TextTooLongError):
                raise
            except Exception as exc:
                raise InferenceError(f'Laya 推理失败：{type(exc).__name__}: {exc}') from exc

    def close(self):
        with self._lock:
            if self.agent is not None:
                self.agent.__exit__(None, None, None)
                self.agent = None

