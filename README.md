# GPU_homework2
本项目是一个针对 GPU 体系结构、CUDA 编程及算子开发领域的垂直领域问答推理服务。

核心基于 `Qwen3-0.6B-GPU-Pro` 模型，采用 **vLLM** 推理引擎进行加速，并配合 FP8 量化与 Prefix Caching 技术，旨在 RTX 5090 平台上实现极致的推理吞吐量与 ROUGE-L 评测高分。

## 🚀 核心优化亮点

1.  **推理引擎升级**: 使用 `vLLM` 替代原生的 Transformers，利用 **PagedAttention** 技术显著提升显存利用率和吞吐量。
2.  **FP8 量化**: 启用 `quantization="fp8"`，在 RTX 5090 上大幅降低显存占用并加速计算。
3.  **Prompt 工程**: 内置针对 ROUGE-L(F1) 优化的 System Prompt，强制模型输出**单行、结构化**且包含特定 API 关键词（如 `cudaMemcpyAsync`, `Tensor Core`）的答案。
4.  **冷启动消除**: 服务启动阶段自动加载本地数据集 (`qwen-zh-basic.json`) 进行**预热 (Warmup)**，消除首次推理的编译延迟。
5.  **完全离线**: 强制设置 `TRANSFORMERS_OFFLINE=1`，确保在无外网环境下稳定运行。

## 📂 项目结构说明

**⚠️ 重要**: `serve.py` 中硬编码了模型加载路径，请务必严格保持以下目录结构，否则服务无法启动。

```text
.
├── Dockerfile              # 容器构建文件 (保持默认 CMD 和 EXPOSE)
├── serve.py                # [核心代码] vLLM 推理服务逻辑
├── requirements.txt        # 依赖库: vllm, fastapi, transformers, uvicorn
├── download_model.py       # 权重下载脚本
├── README.md               # 本文档
└── local-model/            # 挂载或存放权重的根目录
    └── Iammarcus/
        └── Qwen3-0.6B-GPU-Pro/    # 模型权重文件夹
            ├── config.json
            ├── model.safetensors
            ├── tokenizer.json
            └── qwen-zh-basic.json # [必须] 用于预热的 Prompt 数据集

```

## 🛠️ 环境依赖

* **Base Image**: 推荐 `vllm/vllm-openai:latest` 或 `nvcr.io/nvidia/pytorch:25.04-py3` (需 pip 安装 vllm)。
* **Python Packages**: 见 `requirements.txt`。
* **GPU 配置**: 代码针对 **RTX 5090 (32GB)** 优化，参数设置为 `gpu_memory_utilization=0.95`, `tensor_parallel_size=1`。

## 🔌 API 接口契约

服务启动后监听 **8000** 端口。

### 1. 推理接口 (POST `/predict`)

评测系统调用的主要端点。

* **请求 Body**:
```json
{
  "prompt": "如何优化卷积算子的显存访问？"
}

```


*(注：代码底层支持 `List[str]` 批处理输入，但此处展示标准评测格式)*
* **响应 Body**:
```json
{
  "response": [
    "GPU专家回答: 1) 优化卷积显存访问需利用 shared memory 减少全局内存带宽... (此处为单行长文本)"
  ]
}

```



### 2. 健康检查 (GET `/`)

* **返回**: `{"status": "batch"}`
* **说明**: 当且仅当模型加载完成**并且**预热推理结束后，才会返回此状态。

## ⚙️ 修改与配置细节

### System Prompt 策略

为了在 `jieba + ROUGE-L` 评测中获胜，我们在 `serve.py` 中定义了严格的 `SYSTEM_PROMPT`：

* **格式**: 严禁换行 (`\n`)，严禁 Markdown 标题。
* **内容**: 必须包含 CUDA 具体的 API 名称（如 `cudaStreamWaitEvent`）。
* **结构**: 采用 `1)... 2)... 3)...` 的紧凑列点形式。

### vLLM 初始化参数

位于 `serve.py` 中的关键配置：

```python
llm = LLM(
    model=model_dir,
    dtype="bfloat16",
    quantization="fp8",          # 关键：FP8 加速
    gpu_memory_utilization=0.95, # 关键：吃满 5090 显存
    enable_prefix_caching=True,  # 开启前缀缓存，加速重复 System Prompt 的场景
    max_model_len=1024,
    enforce_eager=False          # 使用 CUDA Graph
)

```

## ⚠️ 注意事项

1. **预热数据**: 请确保 `./local-model/.../qwen-zh-basic.json` 文件存在，否则服务启动会报错。
2. **显存警告**: 如果迁移到显存较小（<24GB）的显卡上运行，请在 `serve.py` 中将 `gpu_memory_utilization` 下调至 `0.85` 或更低。
3. **网络连接**: 代码包含网络检测逻辑，但在评测环境中应始终假设处于**断网**状态。
