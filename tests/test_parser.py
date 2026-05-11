"""
文档解析器测试模块

测试DocumentParser的各种格式解析功能。
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from treerag.parser import DocumentParser
from treerag.models import Document


class TestDocumentParser(unittest.TestCase):
    """DocumentParser文档解析器测试。"""

    def setUp(self) -> None:
        """测试前准备：创建解析器实例。"""
        self.parser = DocumentParser(max_segment_length=500)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        """测试后清理临时文件。"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write_temp_file(self, filename: str, content: str) -> str:
        """创建临时文件并写入内容。"""
        path = os.path.join(self.temp_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    # ===== TXT解析测试 =====

    def test_parse_txt(self) -> None:
        """测试TXT文件解析。"""
        content = (
            "这是第一段内容。包含多个句子。\n\n"
            "这是第二段内容。也有多个句子。\n\n"
            "这是第三段内容。"
        )
        path = self._write_temp_file("test.txt", content)
        docs = self.parser.parse(path)

        self.assertGreater(len(docs), 0)
        for doc in docs:
            self.assertIsInstance(doc, Document)
            self.assertTrue(len(doc.content) > 0)
            self.assertEqual(doc.metadata.get("format"), "txt")

    def test_parse_txt_empty(self) -> None:
        """测试空TXT文件解析。"""
        path = self._write_temp_file("empty.txt", "")
        docs = self.parser.parse(path)
        self.assertEqual(len(docs), 0)

    def test_parse_txt_encoding(self) -> None:
        """测试不同编码的TXT文件。"""
        # 测试UTF-8编码
        content = "UTF-8编码测试内容。"
        path = self._write_temp_file("utf8.txt", content)
        docs = self.parser.parse(path)
        self.assertGreater(len(docs), 0)

    def test_parse_txt_long_paragraph(self) -> None:
        """测试长段落TXT文件的分段。"""
        # 创建一个很长的段落
        content = "这是一个很长的段落。" * 200
        path = self._write_temp_file("long.txt", content)
        docs = self.parser.parse(path)

        self.assertGreater(len(docs), 1)
        # 每个分段不应超过max_segment_length
        for doc in docs:
            self.assertLessEqual(len(doc.content), self.parser.max_segment_length + 50)

    # ===== Markdown解析测试 =====

    def test_parse_markdown(self) -> None:
        """测试Markdown文件解析。"""
        content = (
            "# 第一章\n\n"
            "这是第一章的内容。包含一些文字。\n\n"
            "## 第一节\n\n"
            "这是第一节的内容。\n\n"
            "# 第二章\n\n"
            "这是第二章的内容。"
        )
        path = self._write_temp_file("test.md", content)
        docs = self.parser.parse(path)

        self.assertGreater(len(docs), 0)
        for doc in docs:
            self.assertEqual(doc.metadata.get("format"), "markdown")

    def test_parse_markdown_sections(self) -> None:
        """测试Markdown按章节分割。"""
        content = (
            "# 简介\n\n"
            "项目简介内容。\n\n"
            "# 安装\n\n"
            "安装说明内容。\n\n"
            "# 使用\n\n"
            "使用方法内容。"
        )
        path = self._write_temp_file("sections.md", content)
        docs = self.parser.parse(path)

        # 检查section元数据
        sections = set(doc.metadata.get("section", "") for doc in docs)
        self.assertTrue(len(sections) > 0)

    def test_parse_markdown_no_headers(self) -> None:
        """测试没有标题的Markdown文件。"""
        content = "只是一些普通的文本内容。\n\n分成多个段落。"
        path = self._write_temp_file("no_headers.md", content)
        docs = self.parser.parse(path)

        self.assertGreater(len(docs), 0)

    # ===== 文件不存在测试 =====

    def test_parse_file_not_found(self) -> None:
        """测试解析不存在的文件。"""
        with self.assertRaises(FileNotFoundError):
            self.parser.parse("/nonexistent/file.txt")

    def test_parse_unsupported_format(self) -> None:
        """测试不支持的文件格式。"""
        path = self._write_temp_file("test.xyz", "内容")
        with self.assertRaises(ValueError) as ctx:
            self.parser.parse(path)
        self.assertIn("不支持的文件格式", str(ctx.exception))

    # ===== 分段测试 =====

    def test_split_into_segments_short(self) -> None:
        """测试短文本分段。"""
        text = "短文本"
        segments = self.parser._split_into_segments(text)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0], "短文本")

    def test_split_into_segments_empty(self) -> None:
        """测试空文本分段。"""
        segments = self.parser._split_into_segments("")
        self.assertEqual(len(segments), 0)

    def test_split_into_segments_paragraphs(self) -> None:
        """测试按段落分段。"""
        text = "第一段内容。\n\n第二段内容。\n\n第三段内容。"
        segments = self.parser._split_into_segments(text)

        # 应该分成多个段落
        self.assertGreater(len(segments), 0)

    def test_split_into_segments_respects_max_length(self) -> None:
        """测试分段遵守最大长度限制。"""
        # 创建超过max_length的文本
        text = "这是一句话。" * 200
        segments = self.parser._split_into_segments(text)

        for seg in segments:
            self.assertLessEqual(len(seg), self.parser.max_segment_length + 50)

    # ===== 文本清理测试 =====

    def test_clean_text(self) -> None:
        """测试文本清理功能。"""
        dirty_text = "  多余\t空白\r字符  \n\n\n多个空行  "
        cleaned = DocumentParser._clean_text(dirty_text)

        self.assertNotIn("\t", cleaned)
        self.assertNotIn("\r", cleaned)
        self.assertNotIn("\f", cleaned)
        # 不应有三个以上连续换行
        self.assertNotIn("\n\n\n", cleaned)

    # ===== DOCX解析测试（仅测试依赖缺失时的错误处理）=====

    def test_parse_docx_no_dependency(self) -> None:
        """测试DOCX解析时python-docx未安装的情况。"""
        path = self._write_temp_file("test.docx", "fake docx content")

        # 临时移除python-docx的导入
        import sys
        docx_module = sys.modules.get("docx")
        if docx_module is not None:
            sys.modules["docx"] = None
            try:
                with self.assertRaises(ImportError):
                    self.parser.parse(path)
            finally:
                sys.modules["docx"] = docx_module

    # ===== PDF解析测试（仅测试依赖缺失时的错误处理）=====

    def test_parse_pdf_no_dependency(self) -> None:
        """测试PDF解析时PyPDF2未安装的情况。"""
        path = self._write_temp_file("test.pdf", "fake pdf content")

        import sys
        pypdf2_module = sys.modules.get("PyPDF2")
        if pypdf2_module is not None:
            sys.modules["PyPDF2"] = None
            try:
                with self.assertRaises(ImportError):
                    self.parser.parse(path)
            finally:
                sys.modules["PyPDF2"] = pypdf2_module

    # ===== 自动格式检测测试 =====

    def test_auto_detect_txt(self) -> None:
        """测试自动检测TXT格式。"""
        path = self._write_temp_file("test.txt", "内容")
        docs = self.parser.parse(path)
        self.assertGreater(len(docs), 0)
        self.assertEqual(docs[0].metadata.get("format"), "txt")

    def test_auto_detect_markdown(self) -> None:
        """测试自动检测Markdown格式。"""
        path = self._write_temp_file("test.markdown", "# 标题\n\n内容")
        docs = self.parser.parse(path)
        self.assertGreater(len(docs), 0)
        self.assertEqual(docs[0].metadata.get("format"), "markdown")

    def test_auto_detect_text_extension(self) -> None:
        """测试.text扩展名检测。"""
        path = self._write_temp_file("test.text", "内容")
        docs = self.parser.parse(path)
        self.assertGreater(len(docs), 0)
        self.assertEqual(docs[0].metadata.get("format"), "txt")


if __name__ == "__main__":
    unittest.main()
