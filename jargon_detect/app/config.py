from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.bootstrap import PROJECT_ROOT, resolve_path


class Settings(BaseSettings):
    # app.__init__ has already loaded the single project .env into os.environ.
    model_config = SettingsConfigDict(extra='ignore')
    project_name: str = 'laya-jargon-api'
    laya_model_path: Path = PROJECT_ROOT / 'models/laya-multilingual'
    laya_device: Literal['auto', 'cpu', 'cuda'] = 'auto'
    api_host: str = '127.0.0.1'
    api_port: int = Field(default=8000, ge=1, le=65535)
    max_text_length: int = Field(default=2000, ge=1, le=20000)
    laya_max_tokens: int = Field(default=4096, ge=512, le=8192)

    @model_validator(mode='after')
    def local_path(self):
        self.laya_model_path = resolve_path(self.laya_model_path)
        return self
