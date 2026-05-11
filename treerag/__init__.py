"""
TreeRAG - 基于树索引的智能文档检索引擎

TreeRAG 是一个独立自研的项目，核心功能是通过LLM推理构建层次化树索引来检索文档内容，
替代传统向量化RAG方案。

核心特性：
- 层次化树索引：通过LLM递归构建文档摘要树
- 智能搜索：基于LLM推理的树遍历搜索
- 多LLM后端：支持OpenAI、Claude、Ollama
- 多格式文档：支持PDF、Markdown、TXT、DOCX
- Web界面：提供现代化的Web UI
"""

__version__ = "1.0.0"
__author__ = "TreeRAG Team"
