# 模型交付说明

仓库：convaiinnovations/laya-multilingual。

版本：1720e3e3357cfe1e281542e223f8273b0890ca34。

本源码包不包含模型。请从你在魔搭社区上传的仓库下载完整模型目录，包含 model.safetensors、rl_agent_config.json、encoder/config.json 和完整 tokenizer；根 config、官方 README 和下载哈希清单建议保留。无需另下 mmBERT 权重；完整参数已包含在 Laya 权重中。

核心推理与原项目保持一致：eager 加载、启动预热、单进程复用、串行推理；app/questions.py 的说明和 A/B 顺序保持不变；对外 dialogue 字符串仍原样放入模型内部 text 状态键。A/B 映射为 jargon/normal，SDK 概率不重新计算。

运行 `python -m tools.doctor` 会检查必需文件；模型附带 download-manifest.json 时比对其中 SHA256，否则明确报告未校验哈希。用 `--require-hashes` 可拒绝缺少哈希清单的模型。`--load-model` 还会实际加载和预测。常规 API 启动检查必需文件并预热，但不重复计算全部哈希。

模型目录可移至外部，在 .env 设置 LAYA_MODEL_PATH 即可。不要随意修改 encoder/tokenizer 配置或 temperature。官方模型 README 标注 Apache-2.0，交付保留官方说明，第三方库许可亦应按各自要求处理。

不同硬件、CPU/GPU、库版本可能有数值差异；evidence/validation.json 中的接口与迁移验证不等于目标 Linux/A100 已实测，也不等于分类效果评估。

## 模型放置位置（必须先完成）

方式一：下载到项目默认目录，无须修改路径配置：

```text
laya-jargon-api/
└── models/
    └── laya-multilingual/
        ├── model.safetensors
        ├── rl_agent_config.json
        ├── encoder/
        │   └── config.json
        └── tokenizer/
            ├── tokenizer.json
            └── tokenizer_config.json
```

方式二：放在项目外，只修改根目录 `.env`：

```dotenv
# Linux 示例
LAYA_MODEL_PATH=/data/models/laya-multilingual
# Windows 示例（使用时只保留其中一行）
# LAYA_MODEL_PATH=D:/models/laya-multilingual
```

该变量必须指向直接包含 model.safetensors 与 rl_agent_config.json 的本地目录，不是权重文件本身、网页地址或魔搭仓库 ID。若下载工具生成“组织/仓库”子目录，请指向实际模型根目录。魔搭仓库尚未提供，因此此处不填写或猜测仓库名称；本服务不会自动下载模型或调用魔搭远程推理。下载完成后保持本地离线推理配置即可。

下载后执行 `python -m tools.doctor --load-model`，通过后再 `python -m app`。缺少模型时自检失败、服务 ready 返回 503，不会返回伪造结果。

## 分类效果实验

分类效果和真实案例见 [实验日志](EXPERIMENT_LOG.md)，原始记录随交付包保存。本文与交付版的分类指令、选项顺序和标签映射已核对一致；离线分类成绩不能替代对目标设备及实际接口的验收。
