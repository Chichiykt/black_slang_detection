import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.detector import (ChoiceFormatError, InferenceError, LayaDetector,
                          ModelNotReadyError, TextTooLongError)
from app.schemas import DetectRequest, DetectResponse

logger = logging.getLogger(__name__)


def error_response(code: int, message: str):
    return JSONResponse(status_code=code, content={'code': code, 'message': message, 'data': None})


def create_app(settings: Settings | None = None, detector: LayaDetector | None = None):
    settings = settings or Settings()
    detector = detector or LayaDetector(settings)

    @asynccontextmanager
    async def lifespan(app):
        # Startup failure remains observable through /health and /ready.
        # Inference is never served from a substitute model.
        try:
            await run_in_threadpool(detector.load)
        except Exception:
            logger.exception('Local Laya initialization failed')
        yield
        await run_in_threadpool(detector.close)

    app = FastAPI(title=settings.project_name, version='1.0.0', lifespan=lifespan,
                  description='本地 Laya Multilingual 原生 Choice 中文黑话二分类。概率未经本任务校准。')
    app.state.detector = detector
    app.state.settings = settings

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        return error_response(422, '请求格式错误：dialogue 必须为非空字符串，且 JSON 必须合法')

    @app.get('/health')
    def health():
        return {'status': 'ok'}

    @app.get('/ready')
    def ready():
        status = 200 if detector.ready else 503
        return JSONResponse(status_code=status, content={
            'ready': detector.ready, 'model_path': str(settings.laya_model_path), 'device': detector.device,
            'error': detector.load_error,
        })

    @app.post('/api/v1/detect', response_model=DetectResponse,
              responses={422: {'description': '输入无效或超过字符/token 限制'},
                         503: {'description': '本地模型尚未就绪'}, 500: {'description': '推理失败或模型输出格式异常'}})
    def detect(payload: DetectRequest):
        # Synchronous route runs in a worker thread; detector lock serializes GPU inference.
        try:
            if len(payload.dialogue) > settings.max_text_length:
                raise TextTooLongError(f'文本超过 {settings.max_text_length} 字符限制，未执行截断')
            return DetectResponse(data=detector.detect(payload.dialogue))
        except TextTooLongError as exc:
            return error_response(422, str(exc))
        except ModelNotReadyError as exc:
            return error_response(503, str(exc))
        except ChoiceFormatError:
            logger.exception('Invalid Laya Choice output')
            return error_response(500, '本地模型 Choice 输出格式异常，请检查服务日志')
        except InferenceError:
            logger.exception('Laya inference failed')
            return error_response(500, '本地模型推理失败，请检查服务日志')
        except Exception:
            logger.exception('Unexpected detection failure')
            return error_response(500, '检测服务异常，请检查服务日志')

    return app


app = create_app()

