# 检测接口

`POST /api/v1/detect`，Content-Type 为 application/json：

```json
{"dialogue":"A：为什么登录前需要短信验证？ B：这是平台的常规风控措施。"}
```

成功时 HTTP 200，响应字段保持原项目格式：

```json
{"code":200,"message":"success","data":{"is_jargon":false,"label":"非黑话","choice":"normal","probabilities":{"jargon":0.2,"normal":0.8},"answer_confidence":0.8}}
```

以上响应数值仅说明格式，不是该输入的实际预测。概率未经本任务校准。

dialogue 必须是非空字符串，不接受额外字段；旧 text 字段返回 422。默认最多 2000 字符，实际 token 超限时也返回 422，不返回截断预测。模型不可用返回 503，推理/格式异常返回 500，错误响应 data 为 null。

GET /health 检查 HTTP 进程存活；GET /ready 检查模型加载、预热结果及实际设备，模型可用时返回 200，否则 503。GET /docs 为 Swagger 文档；Swagger 默认依赖 CDN，物理断网时页面可能不能完整显示，API 本身不依赖 CDN。

本交付版只运行直接 FastAPI 服务，不包含公网隧道或原评测网关。若跨域浏览器前端或正式公网接入需要额外策略，由对方接入层处理；不能假定原网关的 CORS/限流/认证随核心服务保留。
