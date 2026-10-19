# 模型需单独下载

本交付包不含模型。将从魔搭社区下载的完整模型目录放在：

`models/laya-multilingual/`

该目录必须直接包含 model.safetensors、rl_agent_config.json、encoder/config.json、tokenizer/tokenizer.json、tokenizer/tokenizer_config.json。

也可以放在任意其他位置，仅修改项目根目录 .env：

```dotenv
LAYA_MODEL_PATH=/data/models/laya-multilingual
```

Windows 示例：`LAYA_MODEL_PATH=D:/models/laya-multilingual`。相对路径以项目根目录为基准。不要填网页地址或仓库 ID；服务只加载下载完成后的本地模型。

配置后执行 `python -m tools.doctor --load-model`，再 `python -m app`。详见 docs/MODEL.md。
