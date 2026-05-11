"""
搜索引擎模块

基于树索引的智能搜索引擎。通过LLM推理在层次化树结构中遍历，
评估节点与查询的相关性，最终从相关叶节点中提取答案。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .cache import QueryCache
from .config import Config
from .llm_client import LLMClient
from .models import TreeNode, SearchResult, IndexStats

logger = logging.getLogger(__name__)


class TreeSearcher:
    """树索引搜索引擎。

    搜索策略：
    1. 从根节点开始深度优先遍历
    2. 使用LLM评估每个节点与查询的相关性
    3. 选择高分路径深入遍历
    4. 到达叶节点时收集相关内容
    5. 从收集的内容中提取最终答案

    Usage:
        >>> searcher = TreeSearcher(config, root_node)
        >>> results = searcher.search("什么是机器学习？")
        >>> for r in results:
        ...     print(f"[{r.score:.2f}] {r.content[:100]}")
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        index: Optional[TreeNode] = None,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        """初始化搜索引擎。

        Args:
            config: 配置对象
            index: 树索引根节点
            llm_client: LLM客户端
        """
        if config is None:
            from .config import load_config
            config = load_config()

        self.config = config
        self.index = index

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

        # 初始化缓存
        self.cache = QueryCache(
            ttl=config.cache_ttl if config.cache_enabled else 0,
        )
        self._cache_enabled = config.cache_enabled

    def set_index(self, index: TreeNode) -> None:
        """设置搜索使用的树索引。

        Args:
            index: 树索引根节点
        """
        self.index = index
        # 设置新索引后清空缓存
        self.cache.clear()

    def search(self, query: str, top_k: Optional[int] = None) -> list[SearchResult]:
        """搜索入口。

        根据查询文本在树索引中搜索相关内容。

        Args:
            query: 搜索查询文本
            top_k: 返回结果数量，如果为None则使用配置值

        Returns:
            搜索结果列表，按相关性分数降序排列
        """
        if self.index is None:
            raise ValueError("未设置树索引，请先调用set_index()或加载索引")

        actual_top_k = top_k if top_k is not None else self.config.top_k

        # 检查缓存
        if self._cache_enabled:
            cached = self.cache.get(query)
            if cached is not None:
                logger.info("搜索结果命中缓存")
                return cached[:actual_top_k]

        logger.info(f"开始搜索: {query}")

        # 收集所有相关节点
        relevant_nodes: list[tuple[TreeNode, float, list[str]]] = []
        self._traverse_tree(self.index, query, relevant_nodes, [])

        # 按分数排序
        relevant_nodes.sort(key=lambda x: x[1], reverse=True)

        # 构建搜索结果
        results: list[SearchResult] = []
        for node, score, path in relevant_nodes[:actual_top_k]:
            result = SearchResult(
                content=node.content,
                score=score,
                source=node.metadata.get("source", ""),
                node_path=path,
                metadata={
                    "title": node.title,
                    "level": node.level,
                    "page": node.metadata.get("page", ""),
                    "section": node.metadata.get("section", ""),
                },
            )
            results.append(result)

        # 如果需要，提取综合答案
        if results and any(r.score > 0.5 for r in results):
            try:
                answer = self._extract_answer(query, results)
                if answer:
                    # 将综合答案作为第一个结果插入
                    answer_result = SearchResult(
                        content=answer,
                        score=max(r.score for r in results),
                        source="综合答案",
                        node_path=[],
                        metadata={"type": "extracted_answer"},
                    )
                    results.insert(0, answer_result)
            except Exception as e:
                logger.warning(f"提取综合答案失败: {e}")

        # 缓存结果
        if self._cache_enabled:
            self.cache.set(query, results)

        logger.info(f"搜索完成，找到 {len(results)} 个结果")
        return results

    def _traverse_tree(
        self,
        node: TreeNode,
        query: str,
        results: list[tuple[TreeNode, float, list[str]]],
        current_path: list[str],
    ) -> None:
        """深度优先遍历树，评估节点相关性。

        从给定节点开始，递归遍历子树，使用LLM评估每个节点的相关性。

        Args:
            node: 当前遍历的节点
            query: 搜索查询
            results: 收集相关节点的列表
            current_path: 从根到当前节点的路径
        """
        # 更新路径
        node_path = current_path + [node.title or node.id[:8]]

        # 评估当前节点的相关性
        score = self._score_relevance(query, node.content or node.summary)

        if score > 0.3:  # 相关性阈值
            # 如果是叶节点，直接添加到结果
            if node.is_leaf():
                results.append((node, score, node_path))
            else:
                # 非叶节点：如果有子节点，继续遍历
                if node.children:
                    for child in node.children:
                        self._traverse_tree(child, query, results, node_path)
                else:
                    # 没有子节点的非叶节点，也添加到结果
                    results.append((node, score, node_path))
        elif score > 0.1 and node.is_leaf():
            # 低相关性但为叶节点，也收集（可能包含有用信息）
            results.append((node, score, node_path))

    def _score_relevance(self, query: str, node_content: str) -> float:
        """使用LLM评估查询与节点内容的相关性。

        Args:
            query: 搜索查询
            node_content: 节点内容

        Returns:
            相关性分数（0-1之间）
        """
        if not node_content or not node_content.strip():
            return 0.0

        # 截取内容，避免prompt过长
        content = node_content[:800]

        prompt = (
            "请评估以下查询与文档内容的相关性。\n"
            "只返回一个0到1之间的数字，不要包含其他文字。\n"
            "0表示完全不相关，1表示高度相关。\n\n"
            f"查询: {query}\n\n"
            f"文档内容: {content}\n\n"
            "相关性分数:"
        )

        try:
            response = self.llm.generate(prompt, temperature=0.1, max_tokens=10)
            # 解析分数
            score_text = response.strip()
            # 提取数字
            import re
            match = re.search(r'([0-9]*\.?[0-9]+)', score_text)
            if match:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))  # 确保在0-1范围内
            return 0.0
        except Exception as e:
            logger.warning(f"相关性评估失败: {e}")
            return 0.0

    def _extract_answer(
        self, query: str, relevant_segments: list[SearchResult]
    ) -> str:
        """从相关段落中提取综合答案。

        将相关段落发送给LLM，要求基于这些内容回答查询。

        Args:
            query: 搜索查询
            relevant_segments: 相关的搜索结果列表

        Returns:
            提取的答案文本
        """
        # 构建上下文
        contexts = []
        for i, seg in enumerate(relevant_segments[:5]):  # 最多使用5个段落
            contexts.append(f"[{i + 1}] {seg.content}")

        combined_context = "\n\n".join(contexts)

        prompt = (
            f"基于以下参考内容，回答用户的问题。\n"
            f"请给出准确、简洁的回答。如果参考内容中没有足够信息，"
            f"请说明。\n\n"
            f"问题: {query}\n\n"
            f"参考内容:\n{combined_context}\n\n"
            f"回答:"
        )

        answer = self.llm.generate(prompt, temperature=0.3, max_tokens=1000)
        return answer.strip()

    def get_cache_stats(self) -> dict[str, Any]:
        """获取缓存统计信息。

        Returns:
            缓存统计字典
        """
        return self.cache.get_stats()

    def clear_cache(self) -> None:
        """清空搜索缓存。"""
        self.cache.clear()
        logger.info("搜索缓存已清空")
