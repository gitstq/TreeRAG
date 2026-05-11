"""
API路由模块

定义TreeRAG Web API的所有路由端点。
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from pydantic import BaseModel, Field

from treerag.config import Config
from treerag.parser import DocumentParser
from treerag.indexer import TreeIndexer
from treerag.searcher import TreeSearcher
from treerag.models import SearchResult

logger = logging.getLogger(__name__)

# 全局状态：存储索引和搜索引擎实例
_indexer: Optional[TreeIndexer] = None
_searcher: Optional[TreeSearcher] = None
_config: Optional[Config] = None


class SearchRequest(BaseModel):
    """搜索请求模型。"""
    query: str = Field(..., description="搜索查询文本", min_length=1)
    top_k: int = Field(default=5, description="返回结果数量", ge=1, le=50)


class SearchResponse(BaseModel):
    """搜索响应模型。"""
    query: str
    results: list[dict[str, Any]]
    total: int


class IndexResponse(BaseModel):
    """索引响应模型。"""
    message: str
    total_segments: int
    total_nodes: int
    max_depth: int


class StatsResponse(BaseModel):
    """统计信息响应模型。"""
    total_nodes: int = 0
    total_documents: int = 0
    max_depth: int = 0
    index_size: int = 0
    has_index: bool = False


class DocumentInfo(BaseModel):
    """文档信息模型。"""
    source: str
    segments: int
    format: str


class ConfigResponse(BaseModel):
    """配置响应模型。"""
    llm_backend: str
    model: str
    base_url: str
    index_dir: str
    cache_enabled: bool


def create_router(config: Config) -> APIRouter:
    """创建API路由。

    Args:
        config: 配置对象

    Returns:
        配置好的APIRouter实例
    """
    global _config, _indexer, _searcher
    _config = config

    router = APIRouter()

    # 确保索引目录存在
    config.ensure_directories()

    def _get_indexer() -> TreeIndexer:
        """获取或创建索引器实例。"""
        global _indexer
        if _indexer is None:
            _indexer = TreeIndexer(config=_config)
        return _indexer

    def _get_searcher() -> TreeSearcher:
        """获取搜索引擎实例。"""
        global _searcher
        indexer = _get_indexer()

        # 尝试加载已有索引
        index_path = Path(_config.index_dir) / "index.json"
        if index_path.exists():
            try:
                root = indexer.load_index(str(index_path))
                if _searcher is None or _searcher.index is None:
                    _searcher = TreeSearcher(config=_config, index=root)
            except Exception as e:
                logger.error(f"加载索引失败: {e}")

        if _searcher is None:
            _searcher = TreeSearcher(config=_config)

        return _searcher

    @router.post("/index", response_model=IndexResponse)
    async def create_index(
        file: UploadFile = File(..., description="要索引的文件"),
    ) -> IndexResponse:
        """上传文件并构建索引。

        接收一个文件上传，解析内容并构建层次化树索引。
        支持PDF、Markdown、TXT、DOCX格式。
        """
        # 验证文件格式
        allowed_extensions = {".pdf", ".md", ".markdown", ".txt", ".text", ".docx"}
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件格式: {suffix}。"
                       f"支持: {', '.join(allowed_extensions)}",
            )

        # 保存上传文件到临时位置
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix
        ) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        try:
            # 解析文档
            parser = DocumentParser(
                max_segment_length=_config.max_segment_length
            )
            documents = parser.parse(tmp_path)

            if not documents:
                raise HTTPException(
                    status_code=400,
                    detail="未能从文件中解析出任何内容",
                )

            # 构建索引
            indexer = _get_indexer()
            root = indexer.build_index(documents)

            # 保存索引
            index_path = str(Path(_config.index_dir) / "index.json")
            indexer.save_index(root, index_path)

            # 更新搜索引擎
            global _searcher
            _searcher = TreeSearcher(config=_config, index=root)

            stats = indexer.get_stats(root, index_path)

            return IndexResponse(
                message=f"索引构建成功: {file.filename}",
                total_segments=stats.total_documents,
                total_nodes=stats.total_nodes,
                max_depth=stats.max_depth,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"索引构建失败: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"索引构建失败: {str(e)}",
            )
        finally:
            # 清理临时文件
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

    @router.post("/search", response_model=SearchResponse)
    async def search(request: SearchRequest) -> SearchResponse:
        """搜索已索引的文档。

        根据查询文本在树索引中搜索相关内容。
        """
        searcher = _get_searcher()

        if searcher.index is None:
            raise HTTPException(
                status_code=400,
                detail="尚未构建索引，请先上传文件进行索引",
            )

        try:
            results = searcher.search(request.query, top_k=request.top_k)
            return SearchResponse(
                query=request.query,
                results=[r.to_dict() for r in results],
                total=len(results),
            )
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"搜索失败: {str(e)}",
            )

    @router.get("/stats", response_model=StatsResponse)
    async def get_stats() -> StatsResponse:
        """获取索引统计信息。"""
        index_path = Path(_config.index_dir) / "index.json"

        if not index_path.exists():
            return StatsResponse(has_index=False)

        try:
            indexer = _get_indexer()
            root = indexer.load_index(str(index_path))
            stats = indexer.get_stats(root, str(index_path))
            return StatsResponse(
                total_nodes=stats.total_nodes,
                total_documents=stats.total_documents,
                max_depth=stats.max_depth,
                index_size=stats.index_size,
                has_index=True,
            )
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"获取统计信息失败: {str(e)}",
            )

    @router.get("/documents", response_model=list[DocumentInfo])
    async def get_documents() -> list[DocumentInfo]:
        """获取已索引的文档列表。"""
        index_path = Path(_config.index_dir) / "index.json"

        if not index_path.exists():
            return []

        try:
            indexer = _get_indexer()
            root = indexer.load_index(str(index_path))
            leaves = root.get_all_leaves()

            # 按来源分组
            doc_map: dict[str, dict[str, int]] = {}
            for leaf in leaves:
                source = leaf.metadata.get("source", "未知来源")
                fmt = leaf.metadata.get("format", "未知")
                if source not in doc_map:
                    doc_map[source] = {"segments": 0, "format": fmt}
                doc_map[source]["segments"] += 1

            return [
                DocumentInfo(
                    source=source,
                    segments=info["segments"],
                    format=info["format"],
                )
                for source, info in doc_map.items()
            ]
        except Exception as e:
            logger.error(f"获取文档列表失败: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"获取文档列表失败: {str(e)}",
            )

    @router.get("/config", response_model=ConfigResponse)
    async def get_config() -> ConfigResponse:
        """获取当前配置信息（隐藏敏感信息）。"""
        return ConfigResponse(
            llm_backend=_config.llm_backend,
            model=_config.model,
            base_url=_config.base_url,
            index_dir=_config.index_dir,
            cache_enabled=_config.cache_enabled,
        )

    @router.delete("/cache")
    async def clear_cache() -> dict[str, str]:
        """清空搜索缓存。"""
        searcher = _get_searcher()
        searcher.clear_cache()
        return {"message": "缓存已清空"}

    return router
