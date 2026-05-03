# 多模态图像生成系统

一个基于 AI 的图像生成系统，支持文本生成图像、提示词增强优化、CLIP 语义相似度评估和 BLIP 图像描述；在 **SD 2.1** 之外，可选用 **SD 1.5 多风格** 文生图（与 SD 2.1 同一套 `/generate` 流程）。

**文档与仓库当前行为对齐说明（本版 README 更新于 2026-04，含 `HF_HOME` 默认到项目内 `huggingface/`、`HF_ENDPOINT` 镜像可选、`PORT`/`OPEN_BROWSER`、参考模式精简为 5 种）：** 启动前必须通过环境变量提供 `HF_TOKEN` 与 `DEEPSEEK_API_KEY`（推荐使用项目根目录 `.env`）。**Hugging Face 模型缓存**默认在**项目根目录下的 `huggingface/`**（环境变量 **`HF_HOME`**；`run_download.bat`、`download_models_stepwise.py` 与 `api_server.py` 在未设置 `HF_HOME` 时一致）。若需使用库的全局默认 **`~/.cache/huggingface`**，可在 `.env` 或系统中设置 **`HF_HOME`**。可选：`LOCAL_FILES_ONLY=true` 仅使用本地缓存；`ALLOW_HF_HUB_DOWNLOAD` 控制缓存缺失时是否允许联网从 Hugging Face 拉取（未设置时逻辑与 `LOCAL_FILES_ONLY` 联动，见下文）。**`huggingface/` 已列入 `.gitignore`**，勿将缓存提交进仓库。

## 功能特性

- **三种生成路径**
  - **SD 2.1 基础生成** — `stable-diffusion-2-1-base`，默认 **768×768**，VAE Tiling
  - **SD 1.5 风格生成** — 在「文生图」面板选择画风：`sd21`（默认）或 `sd15`，并指定 **五种风格之一**（二次元 / 水彩 / 油画 / 写实 / 素描），默认 **512×512**（由 `style_models.json` 中 `sd15_size` 配置）
  - **SD 1.5 + 参考模式** — 独立 Tab：单选 **OpenPose / Lineart / SoftEdge(HED) / Canny / IP-Adapter**；后端 **按当前模式懒加载** SD1.5 底模与对应 ControlNet 或 IP-Adapter 管线，**切换模式会卸载上一栈**，避免多套权重同时占满显存。**Canny** 预处理与旧版一致（OpenCV），不依赖 Annotators。其余控制类预处理（OpenPose/HED/Lineart 等）依赖 `controlnet-aux` 与 `lllyasviel/Annotators`（见下文分步下载）。
- **提示词增强** — DeepSeek API 扩展中文绘画提示词
- **中译英** — 优先本地 `Helsinki-NLP/opus-mt-zh-en`；缓存不可用或加载失败时 **回退 DeepSeek API**
- **CLIP 语义相似度** — 评估生成图与提示词匹配度
- **BLIP 图像描述** — 英文描述生成图
- **语音播报** — 浏览器 Web Speech API
- **提示词质量评估** — 多维度量化提示词质量（信息密度、结构完整性、语言纯净度、具体性、风格完整性、长度合理性、质量增强），优先评估 DeepSeek 扩充后的提示词，适配 CLIP 长度限制
- **图片生成质量评估** — 多维度量化生成图像质量（CLIP 语义一致性、美学评分、清晰度、色彩丰富度、构图平衡、噪声水平、对比度），评分标准经过放宽优化
- **质量评估弹窗** — 生成完成后自动展示图片质量评估，支持手动评估提示词质量，以弹窗形式直观展示各维度评分与改进建议

## 项目结构

```
├── api_server.py             # FastAPI 后端：API、`/static` 静态挂载、根路径返回 index.html
├── reference_modes.json      # 参考模式：HF 仓库 id、预处理类型、中文说明
├── reference_service.py      # 参考模式：按模式加载/卸载管线、预处理入口
├── index.html                # Web 入口 HTML（引用 static/ 下的 CSS/JS）
├── static/
│   ├── css/app.css           # 主界面样式：CSS 变量、布局、响应式、Toast、加载动画等
│   └── js/app.js             # 主界面脚本：请求封装、ControlNet 上传/重置、快捷键与本地偏好等
├── style_models.json         # SD1.5 风格：HF 仓库、权重文件名、本地回退路径、LoRA 强度与提示词提示
├── frontend/                 # React + Vite（可选/备用，与根目录 index 无自动联动）
├── scripts/
│   └── download_models_stepwise.py   # 分步预拉 Hugging Face 资源
├── run_app.bat               # Windows：启动服务并打开浏览器
├── install_deps.bat          # Windows：CUDA 版 PyTorch + requirements.txt（不下载模型）
├── run_download.bat          # Windows：先 call install_deps.bat，再 HF 预下载 --group webapp（默认 HF_HOME=项目\huggingface）
├── run_download_hf_cache_to_repo.bat  # 仅在本 CMD 窗口设置 HF_HOME 后下载 webapp；不装依赖、不改 .env（适合缓存与推理使用不同盘符时）
├── improve.md                # 功能与改进记录（可选，团队内可纳入版本库）
├── docs/
│   └── scoring_standards.md  # 提示词质量与图片生成质量评分标准文档
├── .env.example              # 环境变量模板
├── requirements.txt          # Python 依赖（含 diffusers、peft 等）
└── README.md
```

**静态资源说明：** 服务启动后，浏览器通过 **`/static/css/app.css`**、**`/static/js/app.js`** 加载样式与脚本（由 `api_server` 将本地目录 **`static/`** 挂载到 URL 前缀 **`/static`**）。修改前端时编辑上述文件即可，无需改路由名；若新增图片等资产，同样放在 `static/` 下并以 `/static/...` 引用。

## 快速开始

### 1. 安装依赖（GPU 推荐顺序）

`requirements.txt` **不再列出 `torch`**，避免一条 `pip install -r` 从 PyPI 装到 **CPU 版 PyTorch** 并卸掉你已装的 **`+cu124` 等 CUDA 轮子**。

**Windows（推荐）：** 平时只需记 **两个脚本** — **`install_deps.bat`**：装 **CUDA 版 PyTorch** + `requirements.txt`（不下载模型）；**`run_download.bat`**：先做与前者相同的环境安装，再 **预下载 Hugging Face 模型**（`webapp` 分组）。只需 Python 环境、模型已齐时，只运行 **`install_deps.bat`** 即可。修改 CUDA 版本：编辑 **`install_deps.bat`** 里的 `TORCH_INDEX`（如 `cu121` / `cu118`）。

**命令行示例（与 bat 默认一致，CUDA 12.4）：**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

版本与驱动对应关系以 [PyTorch 官网](https://pytorch.org/get-started/locally/) 为准。建议使用 **Python 3.10+**。

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并填写密钥：

**Windows（cmd）：** `copy .env.example .env`  
**类 Unix：** `cp .env.example .env`

```env
HF_TOKEN=your-huggingface-token-here
DEEPSEEK_API_KEY=your-deepseek-api-key-here
# 可选：HF 缓存目录（不设置则默认为「项目根/huggingface」）
# HF_HOME=E:\path\to\your\cache
# 可选：国内网络可尝试镜像（自行评估可用性）
# HF_ENDPOINT=https://hf-mirror.com
# 可选：首选 HTTP 端口（默认 8000）；若被占用会自动尝试 8001、8002…
# PORT=8000
# 可选：启动后是否自动打开浏览器（默认开启）；设为 0 则只打印控制台中的 URL
# OPEN_BROWSER=1
# 可选：仅离线/仅缓存
# LOCAL_FILES_ONLY=true
# 可选：是否允许 Hub 下载（未设置时，LOCAL_FILES_ONLY=true 则等价于不允许）
# ALLOW_HF_HUB_DOWNLOAD=false
```

不要将真实 `.env` 提交到仓库。

### 3. 启动服务

```bash
python api_server.py
```

控制台会打印实际监听地址（默认 **`http://127.0.0.1:8000`**；若首选端口被占用会自动顺延）。**Windows** 可使用 `run_app.bat`（自动查找上级 `.venv-gpu`、本项目 `.venv310` 或 `PATH` 中的 `python`；**在同一 CMD 窗口前台运行服务**，便于持续查看 Uvicorn 日志，**Ctrl+C** 停止）。通过 **`OPEN_BROWSER`** 可控制是否在就绪后自动打开系统浏览器（见上表）。

## 模型预下载（可选）

脚本 **`scripts/download_models_stepwise.py`** 按仓库或单文件拉取资源，并为子进程设置较长的 Hub 超时（`HF_HUB_ETAG_TIMEOUT` / `HF_HUB_DOWNLOAD_TIMEOUT`）。

**分组说明：**

| `--group` | 内容 |
|-----------|------|
| **`webapp`** | **推荐一键（分两阶段）**：**阶段 1** — `medium` + SD2.1 + **`sd15_styles` 同款**（DreamShaper、油画 LoRA 库、二次元/水彩/素描 **单文件**）；**阶段 2** — 参考模式专用（各 ControlNet、Annotators、IP-Adapter 单文件）。**不下载** `runwayml/stable-diffusion-v1-5` |
| `medium` | 翻译、CLIP、BLIP |
| `large` | SD2.1、ControlNet-Canny（**不含** runwayml SD1.5；底模统一为 DreamShaper，见 `reference_modes.json`） |
| `sd15_styles` | DreamShaper、油画 LoRA 仓库 **snapshot**；另对 **AOM3A3 二次元**、水彩、素描等使用 **`hf_hub_download` 单文件**，避免整库过大或 Windows 下过长路径问题 |
| `reference` | **参考模式全套缓存**：**`Lykon/dreamshaper-8`**（底模，与 `reference_modes.json` 中 `sd15_base` 一致）、各 ControlNet（openpose/lineart/hed/canny）、`lllyasviel/Annotators`（预处理）、`h94/IP-Adapter` 下单文件 `ip-adapter_sd15_light_v11.bin`。若只使用某一种模式，可改为只 **snapshot 对应仓库**（见 `reference_modes.json`），不必一次拉满 |
| `all` | 上述全部；**`sd15_styles` 中的 DreamShaper** 与 **`reference` 中重复**，脚本按仓库 id **去重只拉一次** |

**Windows：** 双击 **`run_download.bat`** 会先按 **`install_deps.bat` 相同顺序**安装 **GPU 版 PyTorch** 与 **`requirements.txt`**，再执行 **`--group webapp`**（**`--timeout 21600`**；**先**拉翻译/CLIP/BLIP、SD2.1、SD1.5 画风相关，**再**拉参考模式资源；失败可重复运行以续传）。脚本与推理进程默认将 **`HF_HOME`** 设为**项目下的 `huggingface`**（若已在 `.env` 中设置 `HF_HOME` 则尊重该值）。若下载机与运行机不同、或希望**仅当前窗口**把缓存下到项目目录而不改 `.env`，可使用 **`run_download_hf_cache_to_repo.bat`**（不执行 `install_deps`，仅设置 `HF_HOME` 后跑下载；结束后按提示把 `HF_HOME` 写入 `.env` 即可让 `api_server` 使用同一缓存）。若仅需 Python 依赖、不拉模型，可运行 **`install_deps.bat`**。若只要画风、不要参考模式快照，可单独运行 **`--group sd15_styles`**；全量则用 **`all`**。

示例：

```bash
python scripts/download_models_stepwise.py --python "C:\path\to\python.exe" --group sd15_styles --timeout 7200
```

仅预拉「参考模式」相关权重（体积大，可按需删减重）：

```bash
python scripts/download_models_stepwise.py --python "C:\path\to\python.exe" --group reference --timeout 7200
```

权重与本地路径见 **`style_models.json`**（`local_fallback`）。若 HF 失败，可将同名文件放到对应路径（需自行获取权重）。

## 环境变量与联网策略（摘要）

| 变量 | 含义 |
|------|------|
| `HF_HOME` | Hugging Face 缓存根目录；未设置时默认为**项目根目录下的 `huggingface/`** |
| `HF_ENDPOINT` | 可选；若访问 `huggingface.co` 不稳定，可在 `.env` 中配置镜像（如 `https://hf-mirror.com`，见 `.env.example` 注释） |
| `PORT` | 首选 HTTP 端口（默认 `8000`）；被占用时进程会自动尝试后续端口 |
| `OPEN_BROWSER` | 用 `python api_server.py` 启动时是否自动打开浏览器（默认开启；`0` / `false` 关闭） |
| `LOCAL_FILES_ONLY=true` | 加载模型时 `local_files_only=True`，不主动从 Hub 拉取 |
| `ALLOW_HF_HUB_DOWNLOAD` | 未设置时：**若未** `LOCAL_FILES_ONLY`，则允许在缓存缺失时联网；若 `LOCAL_FILES_ONLY=true`，则不允许。也可显式设为 `true` / `false` 覆盖 |

## 技术栈

- FastAPI、Uvicorn  
- PyTorch（**请按「快速开始」先装 CUDA 版，勿单靠 `requirements.txt` 装 torch**）、**Diffusers（≥0.27）**、Transformers、**PEFT**（SD1.5 LoRA）  
- **controlnet-aux**（参考模式中 OpenPose/HED/Lineart 等预处理；Canny 仍用 OpenCV）、OpenCV、Pillow、python-dotenv  

## 模型与风格说明（摘要）

- **SD 2.1：** `sd2-community/stable-diffusion-2-1-base`  
- **ControlNet（旧接口 `/generate-controlnet`，仍可用）：** 底模同 **`reference_modes.json` 的 `sd15_base`（默认 DreamShaper）** + `lllyasviel/sd-controlnet-canny`  
- **参考模式（`/process-reference`、`/generate-reference`）：** 底模默认为 **`Lykon/dreamshaper-8`**（`reference_modes.json` 的 `sd15_base`，可改为其它 SD1.5 兼容仓库）；各模式见同文件（如 openpose → `lllyasviel/sd-controlnet-openpose`；IP-Adapter → `h94/IP-Adapter` 下单文件）。输出边长默认 **768**（可在 `reference_modes.json` 顶层改 `output_size`，改为 **512** 可明显省显存）。  
- **SD 1.5 风格（`generation_mode=sd15`）：** 由 `style_models.json` 配置，例如二次元用 **`WarriorMama777/OrangeMixs`** 的 **`AOM3A3_orangemixs.safetensors`**（单文件）、写实基于 `Lykon/dreamshaper-8`，水彩/油画/素描为 DreamShaper + 对应 LoRA；请求体可传 **`sd15_resolution`**：`512` / `640` / `768`（服务端会按 GPU 空闲显存拒绝过高分辨率）。二次元默认 **CLIP Skip = 2**（见 `style_models.json` 中 `anime.clip_skip`）。  
- **CLIP：** `openai/clip-vit-base-patch32`  
- **BLIP：** `Salesforce/blip-image-captioning-large`  
- **翻译：** `Helsinki-NLP/opus-mt-zh-en`（可选 DeepSeek 回退）

## 安装部署（给他人使用）

1. 克隆后进入项目根目录，创建虚拟环境；先按上文安装 **CUDA 版 PyTorch**，再 `pip install -r requirements.txt`（或直接运行 **`install_deps.bat`**）。  
2. 复制 `.env.example` 为 `.env` 并填写密钥。  
3. 预留足够磁盘：**基础套件约 15GB+**；若使用 **SD1.5 风格**，额外需要 AOM3A3 单文件（约 2GB）/ DreamShaper / LoRA 等空间。  
4. 可将手动下载的权重放到 `style_models.json` 的 `local_fallback` 路径；**`models/` 目录已在 `.gitignore` 中忽略**，避免误提交大文件。  
5. 有 NVIDIA GPU + CUDA 时推理更快。

## 工作流程

```
用户输入 →（可选）扩充提示词 → 中译英 → SD 2.1 或 SD1.5 风格 或（另一入口）SD1.5 参考模式
         → BLIP + CLIP → 图片质量评估（自动弹窗）
         → 提示词质量评估（手动触发）→（可选）语音播报
```

### 质量评估流程

1. **提示词质量评估**
   - 输入提示词 → 点击「评估提示词质量」按钮
   - 系统优先对 DeepSeek 扩充后的提示词进行多维度评估
   - 弹窗展示综合评分、各维度得分与改进建议
   - 可根据建议修改提示词后重新评估

2. **图片生成质量评估**
   - 生成图片后自动触发图片质量评估
   - 弹窗展示 CLIP 语义一致性、美学评分、清晰度等 7 个维度
   - 支持随时点击「查看图片质量评估」重新查看

## API 接口摘要

### `GET /status`

返回 `device`、`cuda_available`、`models_loaded`（含 `sd21`、`sd15_anime`、`sd15_dreamshaper`、`blip`、`clip`、`controlnet`、`reference_mode`（当前已加载的参考模式 id，未加载则为 `null`））。

### `POST /generate`

| 字段 | 说明 |
|------|------|
| `prompt` | 用户提示（可中文） |
| `enhanced_prompt` | 可选；若提供则优先生效并参与翻译 |
| `num_inference_steps`、`guidance_scale` | 与惯例相同 |
| `generation_mode` | `"sd21"`（默认）或 `"sd15"` |
| `sd15_style` | 当 `generation_mode` 为 `sd15` 时 **必填**：`anime` \| `watercolor` \| `oil` \| `realistic` \| `sketch` |
| `sd15_resolution` | 可选；仅 `sd15` 时有效，正方形边长 **`512`** / **`640`** / **`768`**（缺省取 `style_models.json` 的 `sd15_size`）；服务端会按 GPU 空闲显存拒绝过高分辨率 |

成功时响应额外包含 `generation_mode`、`sd15_style`（仅 sd15 时有值）、**`used_sd15_resolution`**（仅 sd15 时有值，为实际使用的边长）。

### 参考模式（SD1.5）

- **`GET /reference-modes`**：返回 `reference_modes.json`（含各模式 `label_zh` / `hint_zh` 与仓库配置）。
- **`POST /process-reference`**：`multipart/form-data`，字段 **`ref_mode`**（如 `openpose`）、**`file`**（图片）。返回 **`control_image`**（data URL）与 **`output_size`**。不加载扩散模型，仅做预处理（或 IP-Adapter 下的缩放图）。
- **`POST /generate-reference`**：JSON 字段 **`ref_mode`**（与 `reference_modes.json` 中键一致，如 `openpose`、`lineart`、`softedge`、`canny`、`ip_adapter`）、**`prompt`**、**`control_image`**（纯 base64，与前端 `split(',')[1]` 一致）、**`num_inference_steps`**、**`guidance_scale`**；ControlNet 类模式可选 **`controlnet_conditioning_scale`**（默认 `1.0`）；**`ip_adapter`** 可选 **`ip_adapter_scale`**（默认 `0.6`）；可选 **`enhanced_prompt`**。

### 其他

- `POST /enhance`、`POST /process-canny`、`POST /generate-controlnet`、`POST /clip-evaluate`、`POST /caption` 仍可用；旧 Canny 流程与新版参考模式独立。

## 前端使用

1. 打开服务根 URL（`GET /` 返回 **`index.html`**，样式与逻辑见 **`static/css/app.css`**、**`static/js/app.js`**）。  
2. **文生图（SD 2.1 / SD 1.5 风格）**：在「画风 / 模型」中选择 **SD 2.1** 或 **SD 1.5** 及五种风格（二次元 / 水彩 / 油画 / 写实 / 素描）。  
3. **SD 1.5 + 参考模式**：选择 **参考模式**（单选）→ 上传参考图 → 服务端按模式生成控制图或缩略图 → 生成（与文生图五种风格为不同 Tab；服务端同一时间只保留 **一种** 参考模式对应的权重）。  
4. **扩充提示词**（DeepSeek）、**CLIP / BLIP** 结果展示；**朗读描述**默认关闭，可在顶栏勾选「生成成功后自动朗读」；支持在提示词框内 **Ctrl+Enter** 触发生成；错误与提示以 **Toast** 展示（非阻塞弹窗）。
5. **质量评估弹窗**：生成完成后自动弹出图片质量评估；点击「评估提示词质量」可评估提示词；点击「查看图片质量评估」/「查看提示词质量评估」可随时查看评估结果。
6. 生成结果区可 **下载图片**、**复制描述**。

## CLIP 分数参考

| 相似度（约） | 含义 |
| ------------ | ---- |
| > 0.80       | 很高匹配 |
| 0.60–0.80    | 较好 |
| 0.40–0.60    | 一般 |
| < 0.40       | 偏低 |

*CLIP 输出为约 0～1 的余弦相似度，表内为经验区间。*

## 常见问题

### 启动报错缺少 Token？

检查 `.env` 中 `HF_TOKEN`、`DEEPSEEK_API_KEY`。

### SD1.5 风格加载失败？

先运行 `download_models_stepwise.py --group sd15_styles`（或 `all`），或按 `style_models.json` 手动放置 `local_fallback` 文件；确认 `ALLOW_HF_HUB_DOWNLOAD` / `LOCAL_FILES_ONLY` 与网络环境一致。

### `run_download.bat` 没有拉 SD1.5 风格？

默认 **`webapp`**（分两阶段：先 **核心+SD2.1+画风**，再 **参考模式**）。若只要子集，可用 **`sd15_styles`** / **`reference`** / **`all`**。

### 翻译不走本地模型？

缓存中无 Helsinki 或加载失败时会使用 DeepSeek 做中译英（需有效 `DEEPSEEK_API_KEY`）。

## 环境要求

- Python 3.10+  
- PyTorch 2.x  
- 内存建议 8GB+；磁盘建议 **15GB+**（含 SD1.5 风格时更多）；若使用 **参考模式全套**，建议预留 **约 25GB+** 用于 HF 缓存（含多份 ControlNet 与 Annotators）。  
- 推荐 NVIDIA GPU；参考模式在 **768×768、fp16** 下经验值如下（仅供规划，以实际机型为准）：

| 场景 | 推荐显存 | 说明 |
|------|-----------|------|
| 仅一种 ControlNet（OpenPose/Lineart/HED/Canny） | **10GB+** 更从容 | SD1.5 + 单 ControlNet + 768²；8GB 可尝试关其它占显存程序或把 `output_size` 改为 512 |
| IP-Adapter（light 权重） | **8–10GB+** | 无底模重复加载时略省于部分 ControlNet 组合 |
| CPU 推理 | 内存 **16GB+** 建议 | 极慢，仅作兜底 |

## License

MIT License
