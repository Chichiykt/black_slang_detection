# 黑话对话检测 

本目录包含 FastAPI 源码、配置、部署工具及实验说明，模型需从modelscope单独下载。
模型位置：https://www.modelscope.cn/models/chichi1218/Laya_BlackSlangDetectionModelParams
SDK下载：# 安装 ModelScope  
pip install modelscope
from modelscope import snapshot_download
model_dir = snapshot_download('chichi1218/Laya_BlackSlangDetectionModelParams')

Git 下载：
git clone https://www.modelscope.cn/chichi1218/Laya_BlackSlangDetectionModelParams.git

## 1. 准备运行环境

在目标 Windows/Linux/WSL 创建或选择 Python 3.11 环境。Conda 已安装时例如：

```bash
conda create -n laya-jargon python=3.11 pip -y
conda activate laya-jargon
```

从项目根目录执行安装。NVIDIA GPU 且驱动兼容 CUDA 12.8 构建时：

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt --index-url https://pypi.org/simple
python -m pip check
```

CPU 版则将第一条的索引改为 `https://download.pytorch.org/whl/cpu`。已准备好依赖的环境无需重复安装。`.env` 不能把 CPU 版 PyTorch 变成 CUDA 版。不要在迁移时顺便升级 Laya/Transformers。首次安装需要联网；完全离线安装还需提前准备匹配目标系统/Python/CPU或CUDA的全部依赖 wheel。

## 2. 只修改 .env 配置

部署时使用根目录 `.env` 保存当前机器配置；`.env.example` 是通用模板。如果尚无 `.env`，先复制模板并修改。当前开发目录中的 `.env` 可能已被用于本地测试，不保证与模板一致，接收方需要检查模型路径、设备及端口。本包不含模型，请先下载完整模型。常改的参数：

```dotenv
LAYA_MODEL_PATH=./models/laya-multilingual
LAYA_DEVICE=auto
API_HOST=127.0.0.1
API_PORT=8000
```

- auto 优先 GPU；cpu 强制 CPU；cuda 在不可用时明确报错。
- 对外提供网络服务可将 API_HOST 改为 0.0.0.0；调用方使用实际服务器 IP，访问权限由部署环境决定。
- 外置模型可填绝对路径，例如 Linux `/data/models/laya-multilingual` 或 Windows `D:/models/laya-multilingual`。
- 所有相对路径都以此项目根目录为基准，独立于启动工作目录。外部同名环境变量优先于 .env，再使用代码默认值。
- `RUNTIME_DIR` 默认 ./runtime，自动创建缓存/tmp/logs。可用 APP_TEMP_DIR、HF_HOME 等细分覆盖；APP_TEMP_DIR 决定本进程 TEMP/TMP，避免继承系统临时目录。
- CUDA_VISIBLE_DEVICES 可指定 GPU 索引/UUID，外部已设置时优先使用外部值。不填写则遵循机器的设备可见性。
- 保留 HF_HUB_OFFLINE=1、TRANSFORMERS_OFFLINE=1，可从本地加载；无需在线推理服务。
- 修改 .env 后重启进程生效。无须在 PyCharm 再填写相同环境变量，也无须修改 Python 源码。


### 2.1 模型放置位置（必须先完成）

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

模型准备好后，按第 3 节启动服务；如需先排查环境，可使用第 5 节的可选自检。缺少模型时服务 ready 返回 503，不会返回伪造结果。

## 3. 运行项目（正式启动服务）

完成依赖安装、模型下载和 `.env` 配置后，直接执行 `python -m app` 即可启动 FastAPI 服务。**启动服务不需要先运行测试。**下面的终端方式和 PyCharm 方式任选一种，不要同时重复启动。

### 3.1 在终端运行

先激活已安装项目依赖的环境，再进入包含 `app/` 和 `.env` 的项目根目录。

Windows PowerShell 示例（项目路径替换为实际位置）：

```powershell
conda activate laya-jargon
cd "D:\services\laya-jargon-api"
python -m app
```

Linux / WSL 示例（项目路径替换为实际位置）：

```bash
conda activate laya-jargon
cd /opt/laya-jargon-api
python -m app
```

如果环境名称不是 laya-jargon，将其换成实际环境名。也可以直接使用已安装依赖的环境解释器执行 `-m app`，不要求环境必须叫这个名字。终端启动后保持运行，服务在前台持续接收请求；不要在它仍运行时再点击 PyCharm 启动相同端口。

程序会自动读取根目录 `.env`，加载本地模型并预热。主机和端口取自 `API_HOST`、`API_PORT`，无需额外设置进程环境变量。不要直接运行 `app/main.py`，它负责定义接口；统一启动入口为模块 `app`。

### 3.2 在 PyCharm 中运行

打开本项目，在“运行 → 编辑配置”中新增 Python 配置：

| 配置项 | 设置 |
|---|---|
| Python 解释器 | 已安装 requirements.txt 依赖的 Python 3.11 环境 |
| 运行目标类型 | 模块名称（Module name），不要选择脚本路径 |
| 模块名称 | `app` |
| 工作目录 | 当前项目根目录，即包含 app/、.env 的目录 |
| 环境变量 | 无需重复填写 .env 中的值；清除旧项目遗留的同名配置 |
| 参数 | 留空 |

保存后点击运行按钮。首次加载模型时需等待预热完成，日志会显示加载情况。修改 `.env` 后需要停止并重新启动服务。

### 3.3 判断是否启动成功

终端通常会出现类似日志：

```text
Application startup complete.
Uvicorn running on http://127.0.0.1:8000
```

实际端口以 `.env` 为准。进程存活或出现启动日志不代表模型必然可用，需在浏览器打开：

```text
http://127.0.0.1:8000/ready
```

模型加载、预热成功时 HTTP 200，内容类似：

```json
{"ready": true, "model_path": "实际模型目录", "device": "cuda", "error": null}
```

CPU 模式的 device 为 cpu。若 ready 为 false、返回 503，检查 error 和终端日志；模型缺失或 GPU 不可用等问题必须先解决。

模型就绪后，可以打开接口文档：

```text
http://127.0.0.1:8000/docs
```

选择 `POST /api/v1/detect` → Try it out → 输入 JSON → Execute：

```json
{"dialogue": "A：为什么登录前需要短信验证？ B：这是平台的常规风控措施。"}
```

返回 HTTP 200 和 code=200 表示请求执行成功。dialogue 是当前请求字段，旧 text 字段会返回 422；具体响应字段见第 4 节“检测接口说明”。

如果 `.env` 配置的是 `API_PORT=8002`，上面的所有地址都应改为 8002。`API_HOST=0.0.0.0` 是监听配置，本机访问仍用 127.0.0.1；其他电脑使用实际服务器 IP。Swagger 默认静态资源依赖 CDN，物理断网时可以用 HTTP 客户端或下面的 smoke_test 验证 API。

### 3.4 停止和重新启动

终端前台运行时按 Ctrl+C，等待进程退出；PyCharm 点击红色停止按钮。重新运行同一启动命令即可再次加载模型提供服务。

若已有后台服务占用端口，先确认并停止对应实例，或修改 `.env` 中的 API_PORT，再启动。PyCharm 的停止按钮只能停止它自己启动的运行实例。

## 4. 检测接口说明

### 4.1 请求方式

`POST /api/v1/detect`，请求头 `Content-Type: application/json`。下面以本机 8000 端口为例，实际地址和端口以部署配置为准：

```http
POST http://127.0.0.1:8000/api/v1/detect
Content-Type: application/json

{
  "dialogue": "A：为什么登录前需要短信验证？ B：这是平台的常规风控措施。"
}
```

`dialogue` 必须为非空字符串，不接受额外字段；旧字段 `text` 返回 422。默认最多 2000 字符，可通过 `.env` 的 MAX_TEXT_LENGTH 配置。实际 token 超出模型上下文预算时同样返回 422，不返回截断预测。

### 4.2 成功响应与字段

成功时 HTTP 200，响应格式如下：

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "is_jargon": false,
    "label": "非黑话",
    "choice": "normal",
    "probabilities": {
      "jargon": 0.2,
      "normal": 0.8
    },
    "answer_confidence": 0.8
  }
}
```

以上数值仅用于说明响应格式，不是该输入的实际预测；概率未经本任务校准。

| 字段 | 含义 |
|---|---|
| code、message | 请求处理状态，成功为 200、success |
| data.is_jargon | 是否判为黑话，布尔值 |
| data.label | 中文标签：黑话或非黑话 |
| data.choice | 所选类别：jargon 或 normal |
| data.probabilities | 模型给出的两类概率 |
| data.answer_confidence | SDK 返回的所选答案置信度，缺失时为 null；当前配置下与所选类别概率一致，不代表实际正确率 |

### 4.3 错误响应

| HTTP 状态码 | 含义 |
|---|---|
| 422 | 请求格式或字段无效、输入为空、超过字符或 token 限制 |
| 503 | 本地模型尚未就绪或加载失败 |
| 500 | 模型推理失败、输出格式异常或其他服务异常 |

失败时统一返回以下结构，message 按实际错误给出，data 为 null：

```json
{
  "code": 422,
  "message": "请求格式错误：dialogue 必须为非空字符串，且 JSON 必须合法",
  "data": null
}
```

### 4.4 健康检查与接口文档

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | /health | HTTP 进程存活检查 |
| GET | /ready | 模型加载和预热状态、实际推理设备；就绪为 200，否则 503 |
| GET | /docs | Swagger 在线调试页面 |

Swagger 默认静态资源依赖 CDN，物理断网时页面可能无法完整显示；检测 API 本身不依赖这些静态资源。

本交付版直接运行 FastAPI，不包含公网隧道或原评测网关。若浏览器前端跨域调用或正式公网接入需要额外策略，由部署方接入层处理；原评测网关的 CORS、限流和认证不随核心服务保留。独立接口说明也保留在 [docs/API.md](docs/API.md)，便于单独提供给调用方。

## 5. 可选自检和测试

以下命令用于排查环境或验收，不是启动服务的必经步骤。

### 5.1 启动前自检

在已激活环境、进入项目根目录后执行：

```bash
python -m tools.doctor --load-model
```

doctor 检查模型文件、依赖、目录写权限和设备，并在指定 --load-model 时真实加载/预测。成功返回 status=ok，完成后会退出，不会持续提供 HTTP 服务；之后仍需执行 `python -m app`。为避免额外显存占用，建议在没有服务运行时执行真实模型自检。

模型附带 download-manifest.json 时还会验证 SHA256，未附带时明确显示未校验哈希；可用 `--require-hashes` 强制要求清单。

### 5.2 服务启动后的接口验收

保留服务所在终端运行，在另一个激活同环境的终端中执行：

```bash
python -m tools.smoke_test
```

它连接已运行服务，验证 ready、真实 dialogue 请求及错误输入，不启动/停止进程。默认读取 `.env` 的 API_PORT 并连接 127.0.0.1。需要指定其他地址时：

```bash
python -m tools.smoke_test --base-url http://服务器IP:8000
```

### 5.3 开发回归测试

仅开发或需要运行完整测试用例时安装测试依赖：

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## 6. 长期运行

Linux 可按 deploy/laya-jargon.service.example 配置 systemd，仅修改部署文件中的用户、工作目录及环境解释器路径，不改源码；应用会自行读取项目 .env。Windows 可由对方现有服务管理器执行同样的环境解释器 `-m app`。

保持单进程，不使用 --reload 或多 worker，以免重复加载模型。普通启动日志输出到终端；runtime/logs 为可用日志目录，不会自动写入文件，长期日志交给服务管理器收集。

重要：模型加载失败时为兼容原接口，HTTP 进程仍存活、/ready 返回 503；进程存活及 systemd Restart=on-failure 都不能替代模型就绪检查。

## 7. 目录与交付

app 为核心服务，models 为模型放置位置说明（不含权重），tools 为自检/验收，tests 为开发测试，deploy 为服务模板，docs 为接口/模型说明，evidence 为本交付版实际验证结果。runtime 为本机运行数据，不属于发布包。

开发测试见第 5 节，日常运行无需安装测试依赖。release-manifest.json 记录本源码发布文件与分类规则指纹。模型单独下载，必须保持完整目录结构；Git 的忽略规则会排除权重。



