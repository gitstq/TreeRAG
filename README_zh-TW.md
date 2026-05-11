<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/PRs-Welcome-brightgreen.svg" alt="PRs Welcome">
  <img src="https://img.shields.io/badge/Tests-55%20Passed-success.svg" alt="Tests">
</p>

<h1 align="center">🌳 TreeRAG</h1>

<p align="center">
  <strong>基於樹索引的智慧文件檢索引擎 — 無需向量化，純 LLM 推理驅動的下一代 RAG 方案</strong>
</p>

<p align="center">
  <a href="#-專案介紹">專案介紹</a> •
  <a href="#-核心特性">核心特性</a> •
  <a href="#-快速開始">快速開始</a> •
  <a href="#-詳細使用指南">使用指南</a> •
  <a href="#-設計思路與迭代規劃">設計思路</a>
</p>

<p align="center">
  <a href="./README_zh-CN.md">简体中文</a> |
  <strong>繁體中文</strong> |
  <a href="./README.md">English</a>
</p>

---

## 🎉 專案介紹

**TreeRAG** 是一款創新的文件檢索引擎，它徹底顛覆了傳統 RAG（Retrieval-Augmented Generation）方案中依賴向量資料庫與文字分塊的範式。

### 🔥 解決的核心痛點

傳統 RAG 方案的三大困境：

- **分塊丟失語義**：將文件機械切割為固定長度的 chunk，破壞了段落之間的邏輯關聯
- **向量化精度瓶頸**：嵌入模型對專業術語、長尾概念的語義捕捉能力有限
- **檢索黑盒**：無法解釋「為什麼檢索到這段內容」，缺乏可追溯性

### 💡 TreeRAG 的解決方案

TreeRAG 模擬人類專家閱讀文件的方式：

1. **層次化理解**：將文件組織為樹狀結構，從段落 → 章節 → 全文逐層建構摘要
2. **推理式檢索**：搜尋時從根節點出發，透過 LLM 推理判斷每條路徑的相關性，精準定位目標內容
3. **可解釋性**：完整的檢索路徑可追溯，每一步推理過程透明可見

### 🌟 自研差異化亮點

- **層次化理解**：不同於扁平化的向量空間，TreeRAG 以樹狀結構保留文件的層次語義
- **推理式檢索**：搜尋過程本身就是一次推理，而非簡單的相似度比對
- **可解釋性**：每條檢索結果都附帶完整的推理路徑，讓你清楚知道答案從何而來
- **增量更新**：新增文件無需全量重建索引，支援增量合併
- **多 LLM 後端**：OpenAI、Claude、Ollama 自由切換，不受單一供應商綁定
- **查詢快取**：TTL 過期 + LRU 淘汰策略，重複查詢秒級回應
- **Web UI**：內建深色主題視覺化介面，開箱即用

### 🌟 靈感來源

靈感來自 AlphaGo 的樹搜尋演算法 —— 與其將所有資訊壓縮到一個平面空間（向量化），不如建構層次化的知識樹，讓 AI 像下棋一樣「思考」每一步該走向哪個分支。

---

## ✨ 核心特性

| 特性 | 描述 |
|------|------|
| 🌳 **樹索引引擎** | 遞迴建構層次化文件索引，LLM 自動產生節點摘要 |
| 🔍 **推理式搜尋** | 深度優先樹走訪 + LLM 相關性評分，精準定位答案 |
| 📄 **多格式支援** | PDF、Markdown、TXT、DOCX 全格式解析 |
| 🤖 **多 LLM 後端** | OpenAI、Claude、Ollama 本地模型自由切換 |
| ⚡ **查詢快取** | TTL 過期 + LRU 淘汰策略，重複查詢秒級回應 |
| 🔄 **增量更新** | 新增文件無需全量重建索引，支援增量合併 |
| 🌐 **Web UI** | 內建深色主題視覺化介面，開箱即用 |
| 💻 **CLI 工具** | Rich 美化終端輸出，支援索引 / 搜尋 / 服務全流程 |
| 🔌 **REST API** | FastAPI 驅動，輕鬆整合到現有系統 |
| 🧪 **完善測試** | 55 個單元測試覆蓋核心模組，品質有保障 |

---

## 🚀 快速開始

### 環境需求

- **Python** >= 3.10
- **LLM API**：以下任一即可
  - OpenAI API Key（推薦 GPT-4 / GPT-3.5-turbo）
  - Anthropic API Key（Claude 系列）
  - Ollama 本地模型（推薦 llama3、mistral）

### 安裝步驟

```bash
# 複製倉庫
git clone https://github.com/gitstq/TreeRAG.git
cd TreeRAG

# 安裝依賴
pip install -r requirements.txt

# 可選：安裝特定 LLM 後端
pip install openai    # OpenAI 支援
pip install anthropic # Claude 支援
```

### 本地啟動（Ollama 零成本方案）

```bash
# 1. 設定 LLM（使用 Ollama 本地模型，零成本）
export TREERAG_LLM_BACKEND=ollama
export TREERAG_LLM_MODEL=llama3
export TREERAG_LLM_BASE_URL=http://localhost:11434

# 2. 索引文件
treerag index ./docs/

# 3. 搜尋查詢
treerag search "什麼是機器學習？"

# 4. 啟動 Web 服務
treerag serve --host 0.0.0.0 --port 8000
```

### 使用 OpenAI

```bash
export TREERAG_LLM_BACKEND=openai
export TREERAG_LLM_API_KEY=sk-your-key
export TREERAG_LLM_MODEL=gpt-4

treerag index ./docs/
treerag search "你的問題"
```

### Docker 部署

```bash
docker build -t treerag .
docker run -p 8000:8000 -e TREERAG_LLM_BACKEND=ollama treerag
```

---

## 📖 詳細使用指南

### CLI 完整命令

```bash
# 索引單個檔案
treerag index document.pdf

# 索引整個目錄（遞迴掃描）
treerag index ./knowledge-base/

# 搜尋並指定回傳結果數
treerag search "查詢內容" --top-k 10

# 搜尋並顯示詳細路徑
treerag search "查詢內容" --verbose

# 查看索引統計資訊
treerag stats

# 管理設定
treerag config set llm.backend openai
treerag config set llm.model gpt-4
treerag config show

# 啟動 Web 服務
treerag serve --port 8080 --reload
```

### REST API 使用範例

```python
import requests

# 上傳並索引檔案
with open("document.pdf", "rb") as f:
    requests.post("http://localhost:8000/api/index", files={"file": f})

# 搜尋
response = requests.post("http://localhost:8000/api/search", json={
    "query": "什麼是深度學習？",
    "top_k": 5
})
results = response.json()

# 查看索引統計
stats = requests.get("http://localhost:8000/api/stats").json()
```

### Python SDK 整合範例

```python
from treerag.indexer import TreeIndexer
from treerag.searcher import TreeSearcher
from treerag.parser import DocumentParser
from treerag.llm_client import LLMClient
from treerag.config import Config

# 初始化設定
config = Config(llm_backend="openai", api_key="sk-xxx", model="gpt-4")

# 解析文件
parser = DocumentParser()
documents = parser.parse("./knowledge-base/")

# 建構索引
llm = LLMClient(config)
indexer = TreeIndexer(llm_client=llm)
tree = indexer.build_index(documents)

# 搜尋
searcher = TreeSearcher(tree, llm_client=llm)
results = searcher.search("你的問題", top_k=5)

for result in results:
    print(f"[Score: {result.score:.2f}] {result.content}")
    print(f"  來源: {result.source}")
    print(f"  路徑: {' → '.join(result.node_path)}")
```

### 設定檔說明

在專案根目錄建立 `treerag_config.json`：

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

---

## 💡 設計思路與迭代規劃

### 設計理念

> 「讓 AI 像人類專家一樣閱讀文件 —— 先看目錄，再翻章節，最後精讀段落。」

TreeRAG 的核心設計哲學是**層次化推理**：

- **索引階段**：自底向上建構摘要樹，每一層都是下一層的「目錄」
- **搜尋階段**：自頂向下推理導航，LLM 判斷「這個分支值得深入嗎？」
- **答案提取**：在葉節點層面精讀原文，提取精準答案

### 技術選型

| 元件 | 選型 | 原因 |
|------|------|------|
| 核心語言 | Python 3.10+ | AI 生態最完善，LLM SDK 首選 |
| Web 框架 | FastAPI | 非同步高效能，自動 API 文件 |
| CLI 框架 | Click + Rich | 型別安全 + 終端美化 |
| 文件解析 | PyPDF2 + python-docx | 輕量級，無重依賴 |

### 後續迭代計畫

- [ ] 🗂️ **知識圖譜整合**：將樹索引與知識圖譜結合，支援跨文件關聯推理
- [ ] 📊 **視覺化索引瀏覽器**：互動式樹結構視覺化，支援手動調整
- [ ] 🔗 **MCP 協議支援**：接入 Claude Desktop、Cursor 等 AI 工具
- [ ] 🌍 **多語言最佳化**：針對中文、日文等 CJK 語言的分段與檢索最佳化
- [ ] 📱 **行動端適配**：響應式 Web UI + PWA 支援
- [ ] 🔐 **權限控制**：文件層級的存取控制與加密索引

---

## 📦 打包與部署指南

### pip 安裝（推薦）

```bash
pip install .
```

### 從原始碼執行

```bash
git clone https://github.com/gitstq/TreeRAG.git
cd TreeRAG
pip install -r requirements.txt
python -m treerag.cli serve
```

### Docker 部署

```bash
# 建構映像檔
docker build -t treerag .

# 執行（使用 Ollama 本地模型）
docker run -d -p 8000:8000 \
  -v ./docs:/app/docs \
  -v ./treerag_data:/app/treerag_data \
  --network host \
  treerag

# 執行（使用 OpenAI）
docker run -d -p 8000:8000 \
  -e TREERAG_LLM_BACKEND=openai \
  -e TREERAG_LLM_API_KEY=sk-xxx \
  treerag
```

### 相容環境

| 環境 | 最低版本 | 推薦版本 |
|------|---------|---------|
| Python | 3.10 | 3.11+ |
| Node.js（Web UI 開發） | 18 | 20 LTS |
| Docker | 20.10 | 24+ |
| Ollama | 0.1 | 最新 |

---

## 🤝 貢獻指南

我們歡迎所有形式的貢獻！無論是 Bug 回報、功能建議還是程式碼提交。

### 提交 PR 流程

1. Fork 本倉庫
2. 建立特性分支：`git checkout -b feature/amazing-feature`
3. 提交變更：`git commit -m 'feat: 新增某個特性'`
4. 推送分支：`git push origin feature/amazing-feature`
5. 提交 Pull Request

### 提交規範

遵循 Angular Commit Convention：

- `feat:` 新增功能
- `fix:` 修復問題
- `docs:` 文件更新
- `refactor:` 程式碼重構
- `test:` 測試相關
- `chore:` 建構 / 工具鏈相關

### Issue 回饋規則

請使用 [GitHub Issues](https://github.com/gitstq/TreeRAG/issues) 提交 Bug 回報或功能建議，提交時請附上：

- 問題描述
- 重現步驟
- 期望行為
- 實際行為
- 環境資訊（Python 版本、作業系統、LLM 後端）

---

## 📄 開源協議說明

本專案基於 [MIT License](./LICENSE) 開源。

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
