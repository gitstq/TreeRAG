"""
搜索引擎测试模块

测试TreeSearcher的核心功能，包括搜索流程、相关性评分、缓存等。
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from treerag.models import TreeNode, SearchResult
from treerag.searcher import TreeSearcher
from treerag.cache import QueryCache
from treerag.config import Config


class TestQueryCache(unittest.TestCase):
    """查询缓存测试。"""

    def test_set_and_get(self) -> None:
        """测试缓存存取。"""
        cache = QueryCache(ttl=60)
        results = [
            SearchResult(content="结果1", score=0.9, source="test.txt"),
            SearchResult(content="结果2", score=0.7, source="test.txt"),
        ]
        cache.set("测试查询", results)

        cached = cache.get("测试查询")
        self.assertIsNotNone(cached)
        self.assertEqual(len(cached), 2)
        self.assertEqual(cached[0].content, "结果1")

    def test_cache_miss(self) -> None:
        """测试缓存未命中。"""
        cache = QueryCache()
        result = cache.get("不存在的查询")
        self.assertIsNone(result)

    def test_case_insensitive(self) -> None:
        """测试查询大小写不敏感。"""
        cache = QueryCache()
        results = [SearchResult(content="测试", score=0.5)]
        cache.set("Hello World", results)

        cached = cache.get("hello world")
        self.assertIsNotNone(cached)

    def test_cache_expiration(self) -> None:
        """测试缓存过期。"""
        cache = QueryCache(ttl=0)  # 立即过期
        results = [SearchResult(content="测试", score=0.5)]
        cache.set("测试查询", results)

        # TTL为0应该立即过期
        import time
        time.sleep(0.01)
        cached = cache.get("测试查询")
        self.assertIsNone(cached)

    def test_clear(self) -> None:
        """测试清空缓存。"""
        cache = QueryCache()
        cache.set("查询1", [SearchResult(content="1", score=0.5)])
        cache.set("查询2", [SearchResult(content="2", score=0.5)])

        cache.clear()
        self.assertIsNone(cache.get("查询1"))
        self.assertIsNone(cache.get("查询2"))
        self.assertEqual(len(cache), 0)

    def test_max_size(self) -> None:
        """测试缓存最大容量。"""
        cache = QueryCache(max_size=2)
        cache.set("查询1", [SearchResult(content="1", score=0.5)])
        cache.set("查询2", [SearchResult(content="2", score=0.5)])
        cache.set("查询3", [SearchResult(content="3", score=0.5)])

        # 超过最大容量后，最早的条目应被清除
        self.assertLessEqual(len(cache), 2)

    def test_get_stats(self) -> None:
        """测试缓存统计。"""
        cache = QueryCache()
        cache.set("查询1", [SearchResult(content="1", score=0.5)])
        cache.get("查询1")  # 命中
        cache.get("查询2")  # 未命中

        stats = cache.get_stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertAlmostEqual(stats["hit_rate"], 0.5)

    def test_contains(self) -> None:
        """测试缓存包含检查。"""
        cache = QueryCache()
        cache.set("测试", [SearchResult(content="测试", score=0.5)])
        self.assertIn("测试", cache)
        self.assertNotIn("不存在", cache)


class TestTreeSearcher(unittest.TestCase):
    """TreeSearcher搜索引擎测试。"""

    def setUp(self) -> None:
        """测试前准备。"""
        self.config = Config(
            llm_backend="ollama",
            model="test-model",
            cache_enabled=False,
        )

        self.mock_llm = MagicMock()

        # 创建测试用的树结构
        self.root = TreeNode(
            id="root",
            title="根节点",
            content="这是关于人工智能和机器学习的综合文档",
            summary="AI和ML综合文档",
        )

        # 第一层子节点
        self.ai_node = TreeNode(
            id="ai",
            title="人工智能",
            content="人工智能是计算机科学的一个分支",
            summary="AI概述",
        )
        self.ml_node = TreeNode(
            id="ml",
            title="机器学习",
            content="机器学习是人工智能的核心技术之一",
            summary="ML概述",
        )
        self.root.add_child(self.ai_node)
        self.root.add_child(self.ml_node)

        # 第二层叶节点
        leaf1 = TreeNode(
            id="leaf1",
            title="深度学习",
            content="深度学习是机器学习的一个子领域，使用神经网络来学习数据的层次表示",
            summary="深度学习简介",
            metadata={"source": "ai_intro.txt", "page": 1},
        )
        leaf2 = TreeNode(
            id="leaf2",
            title="自然语言处理",
            content="自然语言处理（NLP）是人工智能和语言学的交叉领域",
            summary="NLP简介",
            metadata={"source": "ai_intro.txt", "page": 2},
        )
        leaf3 = TreeNode(
            id="leaf3",
            title="监督学习",
            content="监督学习是机器学习中最常用的方法，需要标注数据来训练模型",
            summary="监督学习简介",
            metadata={"source": "ml_basics.txt", "page": 1},
        )
        leaf4 = TreeNode(
            id="leaf4",
            title="无监督学习",
            content="无监督学习不需要标注数据，主要用于发现数据中的隐藏模式",
            summary="无监督学习简介",
            metadata={"source": "ml_basics.txt", "page": 2},
        )

        self.ai_node.add_child(leaf1)
        self.ai_node.add_child(leaf2)
        self.ml_node.add_child(leaf3)
        self.ml_node.add_child(leaf4)

    def test_score_relevance(self) -> None:
        """测试相关性评分。"""
        self.mock_llm.generate.return_value = "0.85"

        searcher = TreeSearcher(
            config=self.config,
            index=self.root,
            llm_client=self.mock_llm,
        )

        score = searcher._score_relevance(
            "什么是深度学习",
            "深度学习是机器学习的一个子领域",
        )

        self.assertAlmostEqual(score, 0.85)
        self.mock_llm.generate.assert_called_once()

    def test_score_relevance_invalid_response(self) -> None:
        """测试无效响应的相关性评分。"""
        self.mock_llm.generate.return_value = "无法评估"

        searcher = TreeSearcher(
            config=self.config,
            index=self.root,
            llm_client=self.mock_llm,
        )

        score = searcher._score_relevance("查询", "内容")
        self.assertEqual(score, 0.0)

    def test_score_relevance_empty_content(self) -> None:
        """测试空内容的相关性评分。"""
        searcher = TreeSearcher(
            config=self.config,
            index=self.root,
            llm_client=self.mock_llm,
        )

        score = searcher._score_relevance("查询", "")
        self.assertEqual(score, 0.0)

    def test_search_without_index(self) -> None:
        """测试未设置索引时的搜索。"""
        searcher = TreeSearcher(config=self.config)

        with self.assertRaises(ValueError):
            searcher.search("测试查询")

    def test_search(self) -> None:
        """测试搜索流程。"""
        # 模拟LLM返回不同的相关性分数
        # 遍历顺序: root -> ai_node -> leaf1 -> leaf2 -> ml_node -> leaf3 -> leaf4
        # root需要>0.3才能继续遍历子节点
        self.mock_llm.generate.side_effect = [
            "0.6",   # root节点 - 中相关性（>0.3，继续遍历子节点）
            "0.9",   # AI节点 - 高相关性（>0.3，继续遍历子节点）
            "0.95",  # 深度学习叶节点 - 高相关性
            "0.3",   # NLP叶节点 - 低相关性（>0.1，作为叶节点仍收集）
            "0.8",   # ML节点 - 高相关性（>0.3，继续遍历子节点）
            "0.7",   # 监督学习叶节点 - 高相关性
            "0.2",   # 无监督学习叶节点 - 低相关性（>0.1，作为叶节点仍收集）
            "综合答案内容",  # 提取答案
        ]

        searcher = TreeSearcher(
            config=self.config,
            index=self.root,
            llm_client=self.mock_llm,
        )

        results = searcher.search("什么是深度学习", top_k=5)

        self.assertGreater(len(results), 0)
        # 第一个结果应该是综合答案
        self.assertEqual(results[0].metadata.get("type"), "extracted_answer")

    def test_search_with_cache(self) -> None:
        """测试带缓存的搜索。"""
        config = Config(cache_enabled=True, cache_ttl=60)
        self.mock_llm.generate.return_value = "0.5"

        searcher = TreeSearcher(
            config=config,
            index=self.root,
            llm_client=self.mock_llm,
        )

        # 第一次搜索
        results1 = searcher.search("测试查询")
        call_count = self.mock_llm.generate.call_count

        # 第二次搜索（应命中缓存）
        results2 = searcher.search("测试查询")
        self.assertEqual(self.mock_llm.generate.call_count, call_count)

        # 结果应该相同
        self.assertEqual(len(results1), len(results2))

    def test_clear_cache(self) -> None:
        """测试清空缓存。"""
        config = Config(cache_enabled=True)
        self.mock_llm.generate.return_value = "0.5"

        searcher = TreeSearcher(
            config=config,
            index=self.root,
            llm_client=self.mock_llm,
        )

        searcher.search("测试查询")
        searcher.clear_cache()

        stats = searcher.get_cache_stats()
        self.assertEqual(stats["hits"], 0)
        self.assertEqual(stats["misses"], 0)

    def test_extract_answer(self) -> None:
        """测试答案提取。"""
        self.mock_llm.generate.return_value = "深度学习是机器学习的子领域，使用神经网络学习层次表示。"

        searcher = TreeSearcher(
            config=self.config,
            index=self.root,
            llm_client=self.mock_llm,
        )

        segments = [
            SearchResult(content="深度学习是机器学习的一个子领域", score=0.9),
            SearchResult(content="神经网络是深度学习的基础", score=0.8),
        ]

        answer = searcher._extract_answer("什么是深度学习", segments)
        self.assertIn("深度学习", answer)

    def test_set_index(self) -> None:
        """测试设置索引。"""
        searcher = TreeSearcher(config=self.config)
        self.assertIsNone(searcher.index)

        searcher.set_index(self.root)
        self.assertIsNotNone(searcher.index)
        self.assertEqual(searcher.index.id, "root")


if __name__ == "__main__":
    unittest.main()
