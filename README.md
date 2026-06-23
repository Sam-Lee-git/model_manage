# mm — AI Model Deployment Agent

> 跨平台 AI 模型部署助手，一条命令完成模型推荐、下载、安装与启动。

```
mm
```

---

## 简介

`mm` 是一个智能命令行工具，帮助你在 Windows / Linux / macOS 上快速部署本地 AI 模型。
它通过对话理解你的需求，根据你的硬件配置推荐合适的模型，然后自动完成全流程：

- **硬件检测** — 自动识别 CPU、内存、GPU（CUDA/ROCm/Metal）、磁盘
- **模型推荐** — 结合硬件给出量化版本建议（Q4/Q8/fp16）
- **对话选型** — 用自然语言描述需求，AI 帮你挑选最合适的模型和参数
- **全自动安装** — 创建目录、安装依赖、从 HuggingFace 下载模型文件
- **错误自愈** — 安装失败时自动诊断原因，生成修复方案并执行（Branch-and-Resume 机制）
- **多 LLM 后端** — 自动检测环境变量中的 API Key，支持 Claude、OpenAI、Gemini、Qwen、DeepSeek、MiniMax

---

## 工作原理

### 整体架构

```
用户输入 (prompt_toolkit)
     │
     ▼
┌─────────────────────────────────────────────────────┐
│                    App Orchestrator                 │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────┐│
│  │ Hardware │  │ Catalog  │  │  ConversationMgr   ││
│  │ Detector │  │Recommender│  │  (LLM Provider)   ││
│  └──────────┘  └──────────┘  └────────────────────┘│
│  ┌──────────┐  ┌──────────┐  ┌────────────────────┐│
│  │  State   │  │ Storage  │  │  Install Backends  ││
│  │ Machine  │  │ Planner  │  │  pip/conda/docker  ││
│  └──────────┘  └──────────┘  └────────────────────┘│
│  ┌──────────────────────────────────────────────── ┐│
│  │         Error Recovery (Branch & Resume)        ││
│  └─────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────┘
     │
     ▼
Rich Terminal UI (sequential console output)
```

### 安装流程状态机

```
IDLE
 → DETECTING_HARDWARE      # 检测 CPU/RAM/GPU/磁盘
 → BROWSING_CATALOG        # 加载模型目录，AI 对话推荐
 → MODEL_SELECTED          # 用户确认模型
 → ANALYZING_STORAGE       # 规划安装路径
 → CONFIRMING_PLAN         # 展示安装计划
 → SELECTING_BACKEND       # 选择 pip/conda/docker
 → INSTALLING_DEPENDENCIES # pip install huggingface_hub 等
 → DOWNLOADING_MODEL       # 从 HuggingFace 下载模型文件
 → VERIFYING_INSTALL       # 校验文件完整性
 → COMPLETED               # 输出启动命令
```

每个状态完成后自动写入 `~/.model_manager/sessions/<id>.json`，支持 `mm --resume <id>` 断点续传。

### Branch-and-Resume 错误自愈机制

灵感来源于 [OpenClaw](https://github.com/openClaw) 的分支恢复思路：

```
安装步骤报错
     │
     ▼
[ErrorContext 快照]
  - 报错信息 & traceback
  - 当前环境（Python版本、已装包、环境变量）
  - 硬件信息、剩余磁盘
     │
     ▼
[ErrorDiagnosisAgent] → 发给 LLM 诊断
  - 错误分类（网络/依赖冲突/CUDA不匹配/权限/磁盘）
  - 根因分析 + 置信度
  - 修复步骤列表（fix_plan）
  - 备选方案（alternative_plans）
     │
     ▼
[BranchExecutor] 在隔离环境中执行修复步骤
  - 最多递归 2 层（超出则提示用户手动处理）
  - 修复成功 → ResumeCoordinator 合并状态，从断点继续
  - 修复失败 → 尝试备选方案 → 仍失败则上报用户
```

### 多 LLM 提供商自动检测

启动时扫描环境变量，按优先级自动选择可用的 LLM：

| 优先级 | 提供商 | 环境变量 |
|--------|--------|----------|
| 1 | Claude (Anthropic) | `ANTHROPIC_API_KEY` |
| 2 | DeepSeek | `DEEPSEEK_API_KEY` |
| 3 | Qwen (通义千问) | `QWEN_API_KEY` |
| 4 | OpenAI | `OPENAI_API_KEY` |
| 5 | Gemini | `GEMINI_API_KEY` |
| 6 | MiniMax | `MINIMAX_API_KEY` |

无需配置文件，设置好环境变量即可运行。

---

## 快速开始

### 1. 安装

```bash
git clone https://github.com/Sam-Lee-git/model_manage.git
cd model_manage
pip install -e .
```

### 2. 配置 API Key（选一个即可）

```bash
# Windows
set ANTHROPIC_API_KEY=sk-ant-xxxxxxxx

# Linux / macOS
export ANTHROPIC_API_KEY=sk-ant-xxxxxxxx
```

或者复制 `.env.example` 为 `.env` 填入 Key：

```bash
cp .env.example .env
# 编辑 .env，填入你的 API Key
```

### 3. 启动

```bash
mm
```

启动后进入交互对话，直接用自然语言告诉 agent 你想装什么模型：

```
You: 我想在 D:\models\gemma4 安装 Gemma 4 1B
Agent: 好的！检测到你的硬件支持 CPU 推理，将下载 Q8_0 量化版本（1.2 GB）...
       [INSTALL: google/gemma-4-1b-it]
▶ [1/5] Create install directory
✓ Created: D:\models\gemma4\google__gemma-4-1b-it
▶ [2/5] Install huggingface_hub
▶ [3/5] Download google/gemma-4-1b-it from HuggingFace
...
✓ Startup script written: D:\models\gemma4\google__gemma-4-1b-it\start_gemma-4-1b-it.sh
✓ Installation complete!
Starting the model now. Press Ctrl+C to stop it.
```

安装完成后会自动启动模型。之后如需再次启动，可运行安装目录中的脚本：

```bash
bash /path/to/model/start_<model-name>.sh
```

### 4. 其他启动参数

```bash
mm --resume <session-id>          # 恢复中断的安装
mm --provider deepseek            # 指定 LLM 提供商
mm --deploy-model google/gemma-4-4b-it  # 跳过推荐，直接指定模型
mm --path D:\models               # 指定安装目录
mm --list-sessions                # 查看历史会话
mm --version                      # 显示版本
```

### 5. 内置指令

在对话中可使用 `/` 指令：

| 指令 | 说明 |
|------|------|
| `/help` | 显示帮助 |
| `/status` | 当前安装状态 |
| `/sessions` | 历史会话列表 |
| `/cancel` | 取消当前操作 |
| `/exit` | 退出 |

---

## 支持的模型

内置模型目录（可自动从远端更新）：

| 模型 | 参数量 | 模态 | 最低 VRAM | 推荐量化 |
|------|--------|------|-----------|----------|
| Gemma 4 1B | 1B | 文本 | CPU only | Q8_0 (1.2 GB) |
| Gemma 4 4B | 4B | 文本+图像 | 3 GB | Q4_K_M (2.5 GB) |
| Gemma 4 12B | 12B | 文本+图像 | 7 GB | Q4_K_M (7.2 GB) |
| Gemma 4 27B | 27B | 文本+图像 | 16 GB | Q4_K_M (16 GB) |
| Llama 3.2 3B | 3B | 文本 | CPU only | Q4_K_M (1.9 GB) |
| Llama 3.1 8B | 8B | 文本 | 5 GB | Q4_K_M (4.9 GB) |
| Qwen 2.5 0.5B | 0.5B | 文本 | CPU only | Q4_K_M (0.4 GB) |
| Qwen 2.5 7B | 7B | 文本 | 5 GB | Q4_K_M (4.7 GB) |

---

## 项目结构

```
model_manager/
├── __main__.py          # 入口：mm 命令
├── cli.py               # 命令行参数解析
├── app.py               # 主编排器
├── agent/               # LLM 客户端（多提供商）
│   ├── factory.py       # 自动检测并创建 LLM 客户端
│   ├── conversation.py  # 有状态对话管理
│   ├── error_agent.py   # 错误诊断 Agent
│   └── providers/       # Anthropic / OpenAI-compatible 实现
├── catalog/             # 模型目录
│   └── data/catalog.json
├── hardware/            # 硬件检测（CPU/GPU/内存/磁盘）
├── recovery/            # Branch-and-Resume 错误自愈
├── state/               # 状态机 + 断点持久化
├── backends/            # 安装后端（pip/conda/docker）
├── permissions/         # 权限管理（Windows UAC / Linux sudo / macOS SIP）
├── storage/             # 存储规划
└── ui/                  # Rich 终端 UI + prompt_toolkit 输入
```

---

## 依赖

- Python 3.11+
- `rich` — 终端 UI
- `prompt_toolkit` — 交互式输入
- `httpx` — HTTP 客户端
- `tenacity` — 自动重试
- `psutil` — 硬件信息（可选，有降级方案）
- `anthropic` / `openai` — LLM SDK（至少安装一个，或通过 httpx 直接调用）
- `huggingface_hub` — 模型下载（安装时自动安装）

---

## License

MIT
