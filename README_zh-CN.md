<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/PRs-Welcome-brightgreen.svg" alt="PRs Welcome">
  <img src="https://img.shields.io/badge/Tests-55%20Passed-success.svg" alt="Tests">
</p>

<h1 align="center">🌳 TreeRAG</h1>

<p align="center">
  <strong>基于树索引的智能文档检索引擎 — 无需向量化，纯 LLM 推理驱动的下一代 RAG 方案</strong>
</p>

<p align="center">
  <a href="#-项目介绍">项目介绍</a> •
  <a href="#-核心特性">核心特性</a> •
  <a href="#-快速开始">快速开始</a> •
  <a href="#-详细使用指南">使用指南</a> •
  <a href="#-设计思路与迭代规划">设计思路</a>
</p>

<p align="center">
  <strong>简体中文</strong> |
  <a href="./README_zh-TW.md">繁體中文</a> |
  <a href="./README.md">English</a>
</p>

---

## 🎉 项目介绍

**TreeRAG** 是一款颠覆传统 RAG（Retrieval-Augmented Generation）范式的新一代文档检索引擎。它彻底抛弃了向量数据库和文本分块的固有路线，转而采用纯 LLM 推理驱动的树索引方案，为文档检索带来了全新的解题思路。

### 🔥 解决的核心痛点

传统 RAG 方案长期面临三大困境：

- **分块丢失语义**：将文档机械切割为固定长度的 chunk，破坏了段落之间的逻辑关联，上下文信息支离破碎
- **向量化精度瓶颈**：嵌入模型对专业术语、长尾概念的语义捕捉能力有限，检索精度难以突破天花板
- **检索黑盒**：无法解释"为什么检索到这段内容"，缺乏可追溯性，用户对结果难以建立信任

### 💡 自研差异化亮点

TreeRAG 模拟人类专家阅读文档的方式，形成了鲜明的差异化优势：

1. **层次化理解**：将文档组织为树状结构，从段落 → 章节 → 全文逐层构建摘要，完整保留逻辑层次
2. **推理式检索**：搜索时从根节点出发，通过 LLM 推理判断每条路径的相关性，精准定位目标内容
3. **可解释性**：完整的检索路径可追溯，每一步推理过程透明可见，检索结果可信可控
4. **增量更新**：新增文档无需全量重建索引，支持增量合并，大幅降低维护成本
5. **多 LLM 后端**：OpenAI、Claude、Ollama 本地模型自由切换，灵活适配不同场景和预算
6. **查询缓存**：内置 TTL 过期 + LRU 淘汰策略，重复查询秒级响应
7. **Web UI**：内置深色主题可视化界面，开箱即用

### 🌟 灵感来源

灵感来源于 **AlphaGo 的树搜索算法**——与其将所有信息压缩到一个平面空间（向量化），不如构建层次化的知识树，让 AI 像下棋一样"思考"每一步该走向哪个分支。

---

## ✨ 核心特性

| 特性 | 描述 |
|------|------|
| 🌳 **树索引引擎** | 递归构建层次化文档索引，LLM 自动生成节点摘要 |
| 🔍 **推理式搜索** | 深度优先树遍历 + LLM 相关性评分，精准定位答案 |
| 📄 **多格式支持** | PDF、Markdown、TXT、DOCX 全格式解析 |
| 🤖 **多 LLM 后端** | OpenAI、Claude、Ollama 本地模型自由切换 |
| ⚡ **查询缓存** | TTL 过期 + LRU 淘汰策略，重复查询秒级响应 |
| 🔄 **增量更新** | 新增文档无需全量重建索引，支持增量合并 |
| 🌐 **Web UI** | 内置深色主题可视化界面，开箱即用 |
| 💻 **CLI 工具** | Rich 美化终端输出，支持索引/搜索/服务全流程 |
| 🔌 **REST API** | FastAPI 驱动，轻松集成到现有系统 |
| 🧪 **完善测试** | 55 个单元测试覆盖核心模块，质量有保障 |

---

## 🚀 快速开始

### 环境要求

- **Python** >= 3.10
- **LLM API**：以下任一即可
  - OpenAI API Key（推荐 GPT-4 / GPT-3.5-turbo）
  - Anthropic API Key（Claude 系列）
  - Ollama 本地模型（推荐 llama3、mistral）

### 安装

```bash
# 克隆仓库
git clone https://github.com/gitstq/TreeRAG.git
cd TreeRAG

# 安装依赖
pip install -r requirements.txt

# 可选：安装特定 LLM 后端
pip install openai    # OpenAI 支持
pip install anthropic # Claude 支持
```

### 本地启动（Ollama 零成本方案）

```bash
# 1. 配置 LLM（使用 Ollama 本地模型，零成本）
export TREERAG_LLM_BACKEND=ollama
export TREERAG_LLM_MODEL=llama3
export TREERAG_LLM_BASE_URL=http://localhost:11434

# 2. 索引文档
treerag index ./docs/

# 3. 搜索查询
treerag search "什么是机器学习？"

# 4. 启动 Web 服务
treerag serve --host 0.0.0.0 --port 8000
```

### 使用 OpenAI

```bash
export TREERAG_LLM_BACKEND=openai
export TREERAG_LLM_API_KEY=sk-your-key
export TREERAG_LLM_MODEL=gpt-4

treerag index ./docs/
treerag search "你的问题"
```

### Docker 部署

```bash
docker build -t treerag .
docker run -p 8000:8000 -e TREERAG_LLM_BACKEND=ollama treerag
```

---

## 📖 详细使用指南

### CLI 完整命令

```bash
# 索引单个文件
treerag index document.pdf

# 索引整个目录（递归扫描）
treerag index ./knowledge-base/

# 搜索并指定返回结果数
treerag search "查询内容" --top-k 10

# 搜索并显示详细路径
treerag search "查询内容" --verbose

# 查看索引统计信息
treerag stats

# 管理配置
treerag config set llm.backend openai
treerag config set llm.model gpt-4
treerag config show

# 启动 Web 服务
treerag serve --port 8080 --reload
```

### REST API 使用示例

```python
import requests

# 上传并索引文件
with open("document.pdf", "rb") as f:
    requests.post("http://localhost:8000/api/index", files={"file": f})

# 搜索
response = requests.post("http://localhost:8000/api/search", json={
    "query": "什么是深度学习？",
    "top_k": 5
})
results = response.json()

# 查看索引统计
stats = requests.get("http://localhost:8000/api/stats").json()
```

### Python SDK 集成

```python
from treerag.indexer import TreeIndexer
from treerag.searcher import TreeSearcher
from treerag.parser import DocumentParser
from treerag.llm_client import LLMClient
from treerag.config import Config

# 初始化配置
config = Config(llm_backend="openai", api_key="sk-xxx", model="gpt-4")

# 解析文档
parser = DocumentParser()
documents = parser.parse("./knowledge-base/")

# 构建索引
llm = LLMClient(config)
indexer = TreeIndexer(llm_client=llm)
tree = indexer.build_index(documents)

# 搜索
searcher = TreeSearcher(tree, llm_client=llm)
results = searcher.search("你的问题", top_k=5)

for result in results:
    print(f"[Score: {result.score:.2f}] {result.content}")
    print(f"  来源: {result.source}")
    print(f"  路径: {' → '.join(result.node_path)}")
```

### 配置文件说明

在项目根目录创建 `treerag_config.json`：

```json
{
  "llm": {
    "backend": "ollama",
    "model": "llama3",
    "base_url": "http://localhost:11434",
    "temperature": 0.3,
    "max_tokens": 2000
  },
  "index": {
    "dir": "./treerag_data",
    "max_segment_length": 1000,
    "group_size": 5
  },
  "cache": {
    "enabled": true,
    "ttl": 3600
  }
}
```

配置加载优先级（后者覆盖前者）：
1. 默认配置值
2. 配置文件（`treerag_config.json`）
3. 环境变量（`TREERAG_` 前缀，如 `TREERAG_LLM_BACKEND`）
4. CLI 命令行参数

---

## 💡 设计思路与迭代规划

### 设计理念

> "让 AI 像人类专家一样阅读文档——先看目录，再翻章节，最后精读段落。"

TreeRAG 的核心设计哲学是**层次化推理**：

- **索引阶段**：自底向上构建摘要树，每一层都是下一层的"目录"
- **搜索阶段**：自顶向下推理导航，LLM 判断"这个分支值得深入吗？"
- **答案提取**：在叶节点层面精读原文，提取精准答案

### 技术选型

| 组件 | 选型 | 原因 |
|------|------|------|
| 核心语言 | Python 3.10+ | AI 生态最完善，LLM SDK 首选 |
| Web 框架 | FastAPI | 异步高性能，自动 API 文档 |
| CLI 框架 | Click + Rich | 类型安全 + 终端美化 |
| 文档解析 | PyPDF2 + python-docx | 轻量级，无重依赖 |

### 后续迭代计划

- [ ] 🗂️ **知识图谱集成**：将树索引与知识图谱结合，支持跨文档关联推理
- [ ] 📊 **可视化索引浏览器**：交互式树结构可视化，支持手动调整
- [ ] 🔗 **MCP 协议支持**：接入 Claude Desktop、Cursor 等 AI 工具
- [ ] 🌍 **多语言优化**：针对中文、日文等 CJK 语言的分段和检索优化
- [ ] 📱 **移动端适配**：响应式 Web UI + PWA 支持
- [ ] 🔐 **权限控制**：文档级别的访问控制和加密索引

---

## 📦 打包与部署指南

### pip 安装（推荐）

```bash
pip install .
```

### 从源码运行

```bash
git clone https://github.com/gitstq/TreeRAG.git
cd TreeRAG
pip install -r requirements.txt
python -m treerag.cli serve
```

### Docker 部署

```bash
# 构建镜像
docker build -t treerag .

# 运行（使用 Ollama 本地模型）
docker run -d -p 8000:8000 \
  -v ./docs:/app/docs \
  -v ./treerag_data:/app/treerag_data \
  --network host \
  treerag

# 运行（使用 OpenAI）
docker run -d -p 8000:8000 \
  -e TREERAG_LLM_BACKEND=openai \
  -e TREERAG_LLM_API_KEY=sk-xxx \
  treerag
```

### 兼容环境

| 环境 | 最低版本 | 推荐版本 |
|------|---------|---------|
| Python | 3.10 | 3.11+ |
| Node.js（Web UI 开发） | 18 | 20 LTS |
| Docker | 20.10 | 24+ |
| Ollama | 0.1 | 最新 |

---

## 🤝 贡献指南

我们欢迎所有形式的贡献！无论是 Bug 报告、功能建议还是代码提交。

### PR 提交规范

遵循 **Angular Commit Convention**：

1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/amazing-feature`
3. 提交更改（请使用规范的提交信息）：
   - `feat:` 新增功能
   - `fix:` 修复问题
   - `docs:` 文档更新
   - `refactor:` 代码重构
   - `test:` 测试相关
   - `chore:` 构建/工具链相关
4. 推送分支：`git push origin feature/amazing-feature`
5. 提交 Pull Request

### Issue 反馈规则

请使用 [GitHub Issues](https://github.com/gitstq/TreeRAG/issues) 提交 Bug 报告或功能建议，提交时请附上：

- **问题描述**：清晰描述遇到的问题或建议的功能
- **复现步骤**：详细的操作步骤，确保他人可以复现
- **期望行为**：你认为应该出现的结果
- **实际行为**：实际出现的结果（附截图或日志更佳）
- **环境信息**：Python 版本、操作系统、LLM 后端及模型

---

## 📄 开源协议

本项目基于 [MIT License](./LICENSE) 开源。

```
MIT License

Copyright (c) 2026 gitstq

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/gitstq">gitstq</a>
</p>
