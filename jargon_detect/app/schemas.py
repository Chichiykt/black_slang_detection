from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator


class DetectRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    dialogue: StrictStr = Field(min_length=1, examples=['这个菠菜平台今天还能上分吗？'])

    @field_validator('dialogue')
    @classmethod
    def non_blank(cls, dialogue):
        if not dialogue.strip():
            raise ValueError('文本不能为空或仅包含空白字符')
        return dialogue


class DetectionData(BaseModel):
    is_jargon: bool
    label: Literal['黑话', '非黑话']
    choice: Literal['jargon', 'normal']
    probabilities: dict[str, float]
    answer_confidence: float | None = None


class DetectResponse(BaseModel):
    code: Literal[200] = 200
    message: str = 'success'
    data: DetectionData

