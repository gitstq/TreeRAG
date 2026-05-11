"""
树索引引擎测试模块

测试TreeIndexer的核心功能，包括树构建、序列化/反序列化、增量更新等。
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from treerag.models import Document, TreeNode, IndexStats
from treerag.indexer import TreeIndexer
from treerag.config import Config


class TestTreeNode(unittest.TestCase):
    """TreeNode数据模型测试。"""

    def test_create_node(self) -> None:
        """测试创建树节点。"""
        node = TreeNode(
            id="test-1",
            title="测试节点",
            content="测试内容",
            summary="测试摘要",
            level=0,
        )
        self.assertEqual(node.id, "test-1")
        self.assertEqual(node.title, "测试节点")
        self.assertEqual(node.content, "测试内容")
        self.assertTrue(node.is_leaf())

    def test_add_child(self) -> None:
        """测试添加子节点。"""
        parent = TreeNode(id="parent", title="父节点")
        child = TreeNode(id="child", title="子节点")
        parent.add_child(child)

        self.assertEqual(len(parent.children), 1)
        self.assertEqual(child.parent_id, "parent")
        self.assertEqual(child.level, 1)
        self.assertFalse(parent.is_leaf())
        self.assertTrue(child.is_leaf())

    def test_get_all_leaves(self) -> None:
        """测试获取所有叶节点。"""
        root = TreeNode(id="root", title="根")
        leaf1 = TreeNode(id="leaf1", title="叶1")
        leaf2 = TreeNode(id="leaf2", title="叶2")
        root.add_child(leaf1)
        root.add_child(leaf2)

        leaves = root.get_all_leaves()
        self.assertEqual(len(leaves), 2)

    def test_get_node_count(self) -> None:
        """测试获取节点总数。"""
        root = TreeNode(id="root")
        child1 = TreeNode(id="c1")
        child2 = TreeNode(id="c2")
        root.add_child(child1)
        root.add_child(child2)

        self.assertEqual(root.get_node_count(), 3)

    def test_get_max_depth(self) -> None:
        """测试获取最大深度。"""
        root = TreeNode(id="root")
        child = TreeNode(id="child")
        grandchild = TreeNode(id="grandchild")
        root.add_child(child)
        child.add_child(grandchild)

        self.assertEqual(root.get_max_depth(), 2)

    def test_serialization(self) -> None:
        """测试节点序列化和反序列化。"""
        original = TreeNode(
            id="test-1",
            title="测试",
            content="内容",
            summary="摘要",
            level=0,
            metadata={"key": "value"},
        )
        child = TreeNode(id="child-1", title="子节点", content="子内容")
        original.add_child(child)

        # 序列化
        data = original.to_dict()
        self.assertEqual(data["id"], "test-1")
        self.assertEqual(len(data["children"]), 1)

        # 反序列化
        restored = TreeNode.from_dict(data)
        self.assertEqual(restored.id, "test-1")
        self.assertEqual(len(restored.children), 1)
        self.assertEqual(restored.children[0].parent_id, "test-1")
        self.assertEqual(restored.children[0].level, 1)

    def test_find_node_by_id(self) -> None:
        """测试按ID查找节点。"""
        root = TreeNode(id="root")
        child = TreeNode(id="child-1")
        grandchild = TreeNode(id="gc-1")
        root.add_child(child)
        child.add_child(grandchild)

        found = root.find_node_by_id("gc-1")
        self.assertIsNotNone(found)
        self.assertEqual(found.id, "gc-1")

        not_found = root.find_node_by_id("nonexistent")
        self.assertIsNone(not_found)


class TestTreeIndexer(unittest.TestCase):
    """TreeIndexer索引引擎测试。"""

    def setUp(self) -> None:
        """测试前准备：创建配置和mock LLM客户端。"""
        self.config = Config(
            llm_backend="ollama",
            model="test-model",
            group_size=3,
        )

        # 创建mock LLM客户端
        self.mock_llm = MagicMock()
        self.mock_llm.generate.return_value = "这是测试摘要内容"

        self.indexer = TreeIndexer(
            config=self.config,
            llm_client=self.mock_llm,
        )

    def test_group_segments(self) -> None:
        """测试段落分组功能。"""
        nodes = [TreeNode(id=f"n{i}", content=f"内容{i}") for i in range(10)]
        groups = self.indexer._group_segments(nodes, group_size=3)

        self.assertEqual(len(groups), 4)  # 3+3+3+1
        self.assertEqual(len(groups[0]), 3)
        self.assertEqual(len(groups[3]), 1)

    def test_group_segments_single(self) -> None:
        """测试单个段落的分组。"""
        nodes = [TreeNode(id="n1", content="内容")]
        groups = self.indexer._group_segments(nodes, group_size=3)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 1)

    def test_generate_summary(self) -> None:
        """测试摘要生成。"""
        nodes = [
            TreeNode(id="n1", content="这是第一段内容"),
            TreeNode(id="n2", content="这是第二段内容"),
        ]

        summary = self.indexer._generate_summary(nodes)
        self.assertEqual(summary, "这是测试摘要内容")
        self.mock_llm.generate.assert_called_once()

    def test_generate_summary_fallback(self) -> None:
        """测试摘要生成的降级方案。"""
        self.mock_llm.generate.side_effect = RuntimeError("LLM不可用")

        nodes = [
            TreeNode(id="n1", content="a" * 200),
            TreeNode(id="n2", content="b" * 200),
        ]

        summary = self.indexer._generate_summary(nodes)
        # 降级方案应返回拼接的内容
        self.assertIn("a", summary)

    def test_build_index(self) -> None:
        """测试索引构建。"""
        documents = [
            Document(content=f"文档片段内容 {i}", metadata={"source": "test.txt"})
            for i in range(6)
        ]

        root = self.indexer.build_index(documents)

        self.assertIsNotNone(root)
        self.assertGreater(root.get_node_count(), 0)
        self.assertGreaterEqual(root.get_max_depth(), 0)

    def test_build_index_empty(self) -> None:
        """测试空文档列表的索引构建。"""
        with self.assertRaises(ValueError):
            self.indexer.build_index([])

    def test_save_and_load_index(self) -> None:
        """测试索引的保存和加载。"""
        # 创建简单的树结构
        root = TreeNode(
            id="root",
            title="根节点",
            content="根内容",
            summary="根摘要",
        )
        child = TreeNode(
            id="child",
            title="子节点",
            content="子内容",
            summary="子摘要",
        )
        root.add_child(child)

        # 保存到临时文件
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            temp_path = f.name

        try:
            self.indexer.save_index(root, temp_path)

            # 验证文件存在且包含正确内容
            self.assertTrue(os.path.exists(temp_path))
            with open(temp_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["version"], "1.0.0")
            self.assertEqual(data["root"]["id"], "root")

            # 加载索引
            loaded_root = self.indexer.load_index(temp_path)
            self.assertEqual(loaded_root.id, "root")
            self.assertEqual(len(loaded_root.children), 1)
            self.assertEqual(loaded_root.children[0].id, "child")
            self.assertEqual(loaded_root.children[0].parent_id, "root")

        finally:
            os.unlink(temp_path)

    def test_load_index_not_found(self) -> None:
        """测试加载不存在的索引文件。"""
        with self.assertRaises(FileNotFoundError):
            self.indexer.load_index("/nonexistent/path/index.json")

    def test_incremental_update(self) -> None:
        """测试增量更新索引。"""
        # 创建初始索引
        docs1 = [
            Document(content="原始文档内容", metadata={"source": "doc1.txt"})
        ]
        root = self.indexer.build_index(docs1)
        original_count = root.get_node_count()

        # 增量更新
        docs2 = [
            Document(content="新增文档内容", metadata={"source": "doc2.txt"})
        ]
        new_root = self.indexer.incremental_update(root, docs2)

        self.assertIsNotNone(new_root)
        self.assertGreater(new_root.get_node_count(), original_count)
        self.assertEqual(len(new_root.children), 2)  # 旧根 + 新子树

    def test_incremental_update_empty(self) -> None:
        """测试空文档列表的增量更新。"""
        root = TreeNode(id="root", content="测试")
        result = self.indexer.incremental_update(root, [])
        self.assertEqual(result.id, "root")

    def test_get_stats(self) -> None:
        """测试获取索引统计信息。"""
        root = TreeNode(id="root", content="根")
        child1 = TreeNode(id="c1", content="叶1")
        child2 = TreeNode(id="c2", content="叶2")
        root.add_child(child1)
        root.add_child(child2)

        stats = self.indexer.get_stats(root)
        self.assertEqual(stats.total_nodes, 3)
        self.assertEqual(stats.total_documents, 2)
        self.assertEqual(stats.max_depth, 1)

    def test_get_stats_with_file(self) -> None:
        """测试包含文件大小信息的统计。"""
        root = TreeNode(id="root", content="根")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write('{"version": "1.0.0", "root": {"id": "root"}}')
            temp_path = f.name

        try:
            stats = self.indexer.get_stats(root, temp_path)
            self.assertGreater(stats.index_size, 0)
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    unittest.main()
