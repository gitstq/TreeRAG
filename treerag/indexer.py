"""
树索引引擎模块

核心模块，负责构建层次化树索引。通过LLM递归地将文档段落分组并生成摘要，
形成从叶节点到根节点的层次结构，实现高效的文档检索。
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Optional

from .config import Config
from .llm_client import LLMClient
from .models import Document, TreeNode, IndexStats

logger = logging.getLogger(__name__)


class TreeIndexer:
    """树索引构建器。

    将文档集合构建为层次化的树索引结构。叶节点存储原始文档内容，
    非叶节点存储由LLM生成的子节点摘要。

    构建流程：
    1. 将文档段落分组
    2. 用LLM生成每组摘要作为父节点
    3. 递归向上构建
    4. 最终形成根节点

    Usage:
        >>> indexer = TreeIndexer(config)
        >>> root = indexer.build_index(documents)
        >>> indexer.save_index(root, "index.json")
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        """初始化树索引构建器。

        Args:
            config: 配置对象，如果为None则使用默认配置
            llm_client: LLM客户端，如果为None则根据配置创建
        """
        if config is None:
            from .config import load_config
            config = load_config()

        self.config = config
        self.group_size = config.group_size

        if llm_client is not None:
            self.llm = llm_client
        else:
            self.llm = LLMClient(
                backend=config.llm_backend,
                api_key=config.api_key,
                model=config.model,
                base_url=config.base_url,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                retry_count=config.retry_count,
                retry_delay=config.retry_delay,
            )

    def build_index(self, documents: list[Document]) -> TreeNode:
        """构建层次化树索引。

        将文档列表递归地构建为树结构。

        Args:
            documents: 文档列表

        Returns:
            树的根节点
        """
        if not documents:
            raise ValueError("文档列表不能为空")

        logger.info(f"开始构建树索引，共 {len(documents)} 个文档片段")

        # 创建叶节点
        leaf_nodes: list[TreeNode] = []
        for i, doc in enumerate(documents):
            node = TreeNode(
                id=str(uuid.uuid4()),
                title=f"片段_{i + 1}",
                content=doc.content,
                summary=doc.content[:200],  # 叶节点摘要为内容前200字
                level=0,
                metadata={
                    "source": doc.metadata.get("source", ""),
                    "page": doc.metadata.get("page", ""),
                    "section": doc.metadata.get("section", ""),
                    "format": doc.metadata.get("format", ""),
                    "doc_index": i,
                },
            )
            leaf_nodes.append(node)

        # 递归构建上层节点
        root = self._build_tree_recursive(leaf_nodes)

        logger.info(
            f"树索引构建完成：共 {root.get_node_count()} 个节点，"
            f"最大深度 {root.get_max_depth()}"
        )
        return root

    def _group_segments(
        self, segments: list[TreeNode], group_size: Optional[int] = None
    ) -> list[list[TreeNode]]:
        """将节点列表分组。

        将节点按指定大小分组，最后一组可能不足group_size个。

        Args:
            segments: 节点列表
            group_size: 每组大小，如果为None则使用配置值

        Returns:
            分组后的节点列表的列表
        """
        size = group_size if group_size is not None else self.group_size
        groups: list[list[TreeNode]] = []

        for i in range(0, len(segments), size):
            group = segments[i:i + size]
            if group:  # 确保不添加空组
                groups.append(group)

        return groups

    def _generate_summary(self, nodes: list[TreeNode]) -> str:
        """调用LLM生成节点摘要。

        将一组子节点的内容发送给LLM，生成简洁的摘要。

        Args:
            nodes: 子节点列表

        Returns:
            LLM生成的摘要文本
        """
        # 构建内容文本
        contents = []
        for i, node in enumerate(nodes):
            # 截取每个节点内容的前500字，避免prompt过长
            content = node.content[:500]
            contents.append(f"[{i + 1}] {content}")

        combined = "\n\n".join(contents)

        prompt = (
            "请对以下多个文档片段进行综合摘要。"
            "摘要应该概括这些片段的核心内容和关键信息，"
            "长度控制在200字以内。\n\n"
            f"文档片段：\n{combined}\n\n"
            "摘要："
        )

        try:
            summary = self.llm.generate(prompt, temperature=0.3, max_tokens=500)
            return summary.strip()
        except Exception as e:
            logger.warning(f"LLM生成摘要失败，使用拼接摘要: {e}")
            # 降级方案：拼接各节点内容的前100字
            fallback_parts = [node.content[:100] for node in nodes]
            return " | ".join(fallback_parts)

    def _generate_title(self, nodes: list[TreeNode]) -> str:
        """调用LLM生成节点标题。

        Args:
            nodes: 子节点列表

        Returns:
            生成的标题文本
        """
        contents = []
        for node in nodes:
            content = node.content[:200]
            contents.append(content)

        combined = "\n".join(contents)

        prompt = (
            "请用简短的标题（不超过20个字）概括以下内容的核心主题：\n\n"
            f"{combined}\n\n"
            "标题："
        )

        try:
            title = self.llm.generate(prompt, temperature=0.3, max_tokens=50)
            return title.strip().strip('"').strip("'").strip("《").strip("》")
        except Exception as e:
            logger.warning(f"LLM生成标题失败，使用默认标题: {e}")
            return f"层级摘要_{len(nodes)}个片段"

    def _build_tree_recursive(self, nodes: list[TreeNode]) -> TreeNode:
        """递归构建树结构。

        将一组节点分组，为每组生成父节点，然后递归处理父节点列表，
        直到只剩一个根节点。

        Args:
            nodes: 当前层级的节点列表

        Returns:
            子树的根节点
        """
        # 基本情况：只有一个节点，直接返回
        if len(nodes) == 1:
            return nodes[0]

        # 分组
        groups = self._group_segments(nodes)

        # 如果只有一组，尝试用更小的分组
        if len(groups) == 1 and len(groups[0]) > 1:
            # 减小分组大小重新分组
            new_size = max(2, len(nodes) // 3)
            groups = self._group_segments(nodes, group_size=new_size)

        # 为每组创建父节点
        parent_nodes: list[TreeNode] = []
        for i, group in enumerate(groups):
            # 生成摘要和标题
            summary = self._generate_summary(group)
            title = self._generate_title(group)

            # 收集子节点的来源信息
            sources = list(set(
                node.metadata.get("source", "")
                for node in group
                if node.metadata.get("source")
            ))

            parent = TreeNode(
                id=str(uuid.uuid4()),
                title=title,
                content=summary,
                summary=summary,
                level=0,  # 将在add_child时更新
                metadata={
                    "child_count": len(group),
                    "sources": sources,
                    "group_index": i,
                },
            )

            for child in group:
                parent.add_child(child)

            parent_nodes.append(parent)

        # 递归构建上层
        return self._build_tree_recursive(parent_nodes)

    def save_index(self, index: TreeNode, path: str) -> None:
        """将树索引序列化保存到JSON文件。

        Args:
            index: 树的根节点
            path: 保存路径
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "1.0.0",
            "root": index.to_dict(),
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        file_size = path_obj.stat().st_size
        logger.info(f"索引已保存到 {path}（大小: {file_size} 字节）")

    def load_index(self, path: str) -> TreeNode:
        """从JSON文件加载树索引。

        Args:
            path: 索引文件路径

        Returns:
            树的根节点

        Raises:
            FileNotFoundError: 索引文件不存在
            ValueError: 索引文件格式错误
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"索引文件不存在: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "root" not in data:
            raise ValueError("索引文件格式错误：缺少root字段")

        root = TreeNode.from_dict(data["root"])
        logger.info(
            f"索引已从 {path} 加载，"
            f"共 {root.get_node_count()} 个节点"
        )
        return root

    def incremental_update(
        self, index: TreeNode, new_documents: list[Document]
    ) -> TreeNode:
        """增量更新索引。

        将新文档添加到现有索引中。策略是：
        1. 为新文档创建叶节点
        2. 将新叶节点作为新的子树
        3. 创建新的根节点包含旧根和新子树

        Args:
            index: 现有的树根节点
            new_documents: 新增的文档列表

        Returns:
            更新后的树根节点
        """
        if not new_documents:
            logger.info("没有新文档需要添加")
            return index

        logger.info(f"增量更新索引，新增 {len(new_documents)} 个文档片段")

        # 为新文档构建子树
        new_subtree = self.build_index(new_documents)

        # 创建新的根节点
        new_root = TreeNode(
            id=str(uuid.uuid4()),
            title="更新后的索引根",
            content="",
            summary="增量更新后的文档索引",
            metadata={
                "update_type": "incremental",
                "old_node_count": index.get_node_count(),
                "new_doc_count": len(new_documents),
            },
        )

        # 将旧索引和新子树作为子节点
        new_root.add_child(index)
        new_root.add_child(new_subtree)

        # 生成新根的摘要
        old_summary = index.summary or index.content[:200]
        new_summary = new_subtree.summary or new_subtree.content[:200]
        new_root.content = f"{old_summary}\n\n{new_summary}"
        new_root.summary = self._generate_summary([index, new_subtree])

        logger.info(
            f"增量更新完成，新索引共 {new_root.get_node_count()} 个节点"
        )
        return new_root

    def get_stats(self, index: TreeNode, index_path: Optional[str] = None) -> IndexStats:
        """获取索引统计信息。

        Args:
            index: 树的根节点
            index_path: 索引文件路径（可选，用于计算文件大小）

        Returns:
            IndexStats统计信息对象
        """
        index_size = 0
        if index_path and Path(index_path).exists():
            index_size = Path(index_path).stat().st_size

        # 统计文档数（叶节点数）
        total_documents = len(index.get_all_leaves())

        return IndexStats(
            total_nodes=index.get_node_count(),
            total_documents=total_documents,
            max_depth=index.get_max_depth(),
            index_size=index_size,
        )
