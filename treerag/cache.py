"""
查询缓存模块

提供基于内存的查询缓存功能，避免重复查询消耗LLM资源。
支持TTL过期和命中率统计。
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .models import SearchResult


@dataclass
class CacheEntry:
    """缓存条目。

    Attributes:
        results: 缓存的搜索结果列表
        created_at: 创建时间戳
        ttl: 生存时间（秒）
    """
    results: list[SearchResult]
    created_at: float = field(default_factory=time.time)
    ttl: int = 3600

    def is_expired(self) -> bool:
        """检查缓存条目是否已过期。"""
        return time.time() - self.created_at > self.ttl


class QueryCache:
    """查询缓存类。

    基于字典的内存缓存，使用查询文本的hash作为键。
    支持TTL过期机制和命中率统计。

    Usage:
        >>> cache = QueryCache()
        >>> cache.set("什么是机器学习", results)
        >>> cached = cache.get("什么是机器学习")
        >>> stats = cache.get_stats()
    """

    def __init__(self, ttl: int = 3600, max_size: int = 1000) -> None:
        """初始化缓存。

        Args:
            ttl: 默认缓存过期时间（秒）
            max_size: 缓存最大条目数
        """
        self._cache: dict[str, CacheEntry] = {}
        self._default_ttl = ttl
        self._max_size = max_size
        # 统计信息
        self._hits: int = 0
        self._misses: int = 0

    @staticmethod
    def _hash_query(query: str) -> str:
        """计算查询文本的hash值。

        Args:
            query: 查询文本

        Returns:
            查询文本的SHA256 hash值
        """
        normalized = query.strip().lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def get(self, query: str) -> Optional[list[SearchResult]]:
        """从缓存中获取查询结果。

        Args:
            query: 查询文本

        Returns:
            缓存的搜索结果列表，如果缓存未命中或已过期则返回None
        """
        key = self._hash_query(query)
        entry = self._cache.get(key)

        if entry is None:
            self._misses += 1
            return None

        if entry.is_expired():
            # 清除过期条目
            del self._cache[key]
            self._misses += 1
            return None

        self._hits += 1
        return entry.results

    def set(self, query: str, results: list[SearchResult], ttl: Optional[int] = None) -> None:
        """将查询结果存入缓存。

        Args:
            query: 查询文本
            results: 搜索结果列表
            ttl: 缓存过期时间（秒），如果为None则使用默认值
        """
        # 如果缓存已满，先清除过期条目
        if len(self._cache) >= self._max_size:
            self._cleanup_expired()
            # 如果仍然已满，清除最早的条目
            if len(self._cache) >= self._max_size:
                oldest_key = min(
                    self._cache.keys(),
                    key=lambda k: self._cache[k].created_at,
                )
                del self._cache[oldest_key]

        key = self._hash_query(query)
        actual_ttl = ttl if ttl is not None else self._default_ttl
        self._cache[key] = CacheEntry(
            results=results,
            ttl=actual_ttl,
        )

    def clear(self) -> None:
        """清空所有缓存。"""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def _cleanup_expired(self) -> None:
        """清除所有过期的缓存条目。"""
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired()
        ]
        for key in expired_keys:
            del self._cache[key]

    def get_stats(self) -> dict[str, Any]:
        """获取缓存统计信息。

        Returns:
            包含命中率、缓存大小等统计信息的字典
        """
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        # 清理过期条目后统计
        self._cleanup_expired()

        return {
            "total_entries": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 4),
            "total_requests": total_requests,
            "max_size": self._max_size,
            "default_ttl": self._default_ttl,
        }

    def __len__(self) -> int:
        """返回当前缓存条目数。"""
        return len(self._cache)

    def __contains__(self, query: str) -> bool:
        """检查查询是否在缓存中（且未过期）。"""
        return self.get(query) is not None
