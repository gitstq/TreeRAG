"""
数据模型定义模块

定义TreeRAG中使用的所有核心数据结构，包括文档、树节点、搜索结果等。
使用Python dataclasses实现，提供类型安全和序列化支持。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class Document:
    """文档数据类，表示解析后的文档片段。

    Attributes:
        content: 文档文本内容
        metadata: 文档元数据，包含source、page、section等信息
    """
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """将文档序列化为字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Document:
        """从字典反序列化文档。"""
        return cls(content=data["content"], metadata=data.get("metadata", {}))

    def __str__(self) -> str:
        """返回文档内容的字符串表示，截断过长的内容。"""
        preview = self.content[:200] + "..." if len(self.content) > 200 else self.content
        return f"Document(content={preview!r}, metadata={self.metadata})"


@dataclass
class TreeNode:
    """树节点数据类，表示层次化索引中的一个节点。

    每个节点可以包含子节点，形成层次结构。叶节点存储原始文档内容，
    非叶节点存储子节点的摘要。

    Attributes:
        id: 节点唯一标识符
        title: 节点标题
        content: 节点内容（叶节点为原文，非叶节点为摘要）
        summary: 节点摘要（由LLM生成）
        children: 子节点列表
        parent_id: 父节点ID
        level: 节点在树中的层级（0为根节点）
        metadata: 节点元数据
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    content: str = ""
    summary: str = ""
    children: list[TreeNode] = field(default_factory=list)
    parent_id: Optional[str] = None
    level: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_child(self, child: TreeNode) -> None:
        """添加子节点。

        Args:
            child: 要添加的子节点
        """
        child.parent_id = self.id
        child.level = self.level + 1
        self.children.append(child)

    def is_leaf(self) -> bool:
        """判断是否为叶节点。"""
        return len(self.children) == 0

    def get_all_leaves(self) -> list[TreeNode]:
        """获取所有叶节点。"""
        if self.is_leaf():
            return [self]
        leaves: list[TreeNode] = []
        for child in self.children:
            leaves.extend(child.get_all_leaves())
        return leaves

    def get_node_count(self) -> int:
        """获取以该节点为根的子树中的节点总数。"""
        count = 1
        for child in self.children:
            count += child.get_node_count()
        return count

    def get_max_depth(self) -> int:
        """获取以该节点为根的子树的最大深度。"""
        if self.is_leaf():
            return 0
        return 1 + max(child.get_max_depth() for child in self.children)

    def to_dict(self) -> dict[str, Any]:
        """将树节点序列化为字典（递归序列化子节点）。"""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "summary": self.summary,
            "children": [child.to_dict() for child in self.children],
            "parent_id": self.parent_id,
            "level": self.level,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TreeNode:
        """从字典反序列化树节点（递归反序列化子节点）。"""
        node = cls(
            id=data["id"],
            title=data.get("title", ""),
            content=data.get("content", ""),
            summary=data.get("summary", ""),
            parent_id=data.get("parent_id"),
            level=data.get("level", 0),
            metadata=data.get("metadata", {}),
        )
        for child_data in data.get("children", []):
            child = cls.from_dict(child_data)
            node.add_child(child)
        return node

    def find_node_by_id(self, node_id: str) -> Optional[TreeNode]:
        """根据ID查找节点（深度优先搜索）。

        Args:
            node_id: 要查找的节点ID

        Returns:
            找到的节点，未找到返回None
        """
        if self.id == node_id:
            return self
        for child in self.children:
            result = child.find_node_by_id(node_id)
            if result is not None:
                return result
        return None

    def get_path_from_root(self) -> list[str]:
        """获取从根节点到当前节点的路径（节点ID列表）。"""
        # 注意：此方法需要完整的树结构，通常从根节点调用
        return [self.id]

    def __str__(self) -> str:
        """返回节点的字符串表示。"""
        content_preview = self.content[:100] + "..." if len(self.content) > 100 else self.content
        return (
            f"TreeNode(id={self.id[:8]}..., title={self.title!r}, "
            f"level={self.level}, children={len(self.children)}, "
            f"content={content_preview!r})"
        )


@dataclass
class SearchResult:
    """搜索结果数据类。

    Attributes:
        content: 匹配的文档内容
        score: 相关性分数（0-1之间）
        source: 来源文件路径
        node_path: 从根节点到匹配节点的路径
        metadata: 额外元数据
    """
    content: str
    score: float = 0.0
    source: str = ""
    node_path: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """将搜索结果序列化为字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SearchResult:
        """从字典反序列化搜索结果。"""
        return cls(
            content=data["content"],
            score=data.get("score", 0.0),
            source=data.get("source", ""),
            node_path=data.get("node_path", []),
            metadata=data.get("metadata", {}),
        )

    def __str__(self) -> str:
        """返回搜索结果的字符串表示。"""
        preview = self.content[:150] + "..." if len(self.content) > 150 else self.content
        return f"SearchResult(score={self.score:.3f}, source={self.source!r}, content={preview!r})"


@dataclass
class IndexStats:
    """索引统计数据类。

    Attributes:
        total_nodes: 树中节点总数
        total_documents: 已索引的文档总数
        max_depth: 树的最大深度
        index_size: 索引文件大小（字节）
    """
    total_nodes: int = 0
    total_documents: int = 0
    max_depth: int = 0
    index_size: int = 0

    def to_dict(self) -> dict[str, Any]:
        """将统计信息序列化为字典。"""
        return asdict(self)

    def __str__(self) -> str:
        """返回统计信息的格式化字符串。"""
        return (
            f"IndexStats(nodes={self.total_nodes}, documents={self.total_documents}, "
            f"max_depth={self.max_depth}, size={self.index_size} bytes)"
        )
