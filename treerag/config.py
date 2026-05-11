"""
配置管理模块

管理TreeRAG的所有配置项，支持从环境变量、配置文件加载，
并提供默认配置值。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional


@dataclass
class Config:
    """TreeRAG全局配置类。

    支持从环境变量和JSON配置文件加载配置。
    环境变量优先级高于配置文件。

    Attributes:
        llm_backend: LLM后端类型（openai/claude/ollama）
        api_key: API密钥
        model: 使用的模型名称
        base_url: API基础URL
        temperature: 生成温度
        max_tokens: 最大生成token数
        index_dir: 索引文件存储目录
        cache_enabled: 是否启用查询缓存
        cache_ttl: 缓存过期时间（秒）
        max_segment_length: 文档分段最大长度
        group_size: 构建树索引时的分组大小
        top_k: 搜索返回结果数量
        host: Web服务监听地址
        port: Web服务监听端口
        retry_count: LLM调用重试次数
        retry_delay: 重试间隔（秒）
    """
    # LLM配置
    llm_backend: str = "ollama"
    api_key: str = ""
    model: str = "llama3"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.3
    max_tokens: int = 2000

    # 索引配置
    index_dir: str = "./treerag_data"
    max_segment_length: int = 1000
    group_size: int = 5

    # 搜索配置
    top_k: int = 5
    cache_enabled: bool = True
    cache_ttl: int = 3600

    # Web服务配置
    host: str = "0.0.0.0"
    port: int = 8000

    # 重试配置
    retry_count: int = 3
    retry_delay: float = 1.0

    def __post_init__(self) -> None:
        """初始化后处理：确保索引目录路径存在。"""
        self.index_dir = str(Path(self.index_dir).resolve())

    def to_dict(self) -> dict[str, Any]:
        """将配置序列化为字典。"""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """将配置序列化为JSON字符串。"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        """从字典创建配置对象。

        Args:
            data: 配置字典

        Returns:
            Config实例
        """
        # 只使用Config中定义的字段，忽略未知字段
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

    def apply_env_overrides(self) -> Config:
        """从环境变量覆盖配置值。

        环境变量命名规则：TREERAG_<FIELD_NAME>，例如TREERAG_API_KEY

        Returns:
            应用环境变量后的Config实例
        """
        env_mapping = {
            "TREERAG_LLM_BACKEND": "llm_backend",
            "TREERAG_API_KEY": "api_key",
            "TREERAG_MODEL": "model",
            "TREERAG_BASE_URL": "base_url",
            "TREERAG_TEMPERATURE": "temperature",
            "TREERAG_MAX_TOKENS": "max_tokens",
            "TREERAG_INDEX_DIR": "index_dir",
            "TREERAG_CACHE_ENABLED": "cache_enabled",
            "TREERAG_CACHE_TTL": "cache_ttl",
            "TREERAG_MAX_SEGMENT_LENGTH": "max_segment_length",
            "TREERAG_GROUP_SIZE": "group_size",
            "TREERAG_TOP_K": "top_k",
            "TREERAG_HOST": "host",
            "TREERAG_PORT": "port",
            "TREERAG_RETRY_COUNT": "retry_count",
            "TREERAG_RETRY_DELAY": "retry_delay",
        }

        for env_var, field_name in env_mapping.items():
            value = os.environ.get(env_var)
            if value is not None:
                current_type = type(getattr(self, field_name))
                try:
                    if current_type == bool:
                        setattr(self, field_name, value.lower() in ("true", "1", "yes"))
                    elif current_type in (int, float):
                        setattr(self, field_name, current_type(value))
                    else:
                        setattr(self, field_name, value)
                except (ValueError, TypeError):
                    pass  # 忽略无法转换的环境变量值

        # 特殊处理：OPENAI_API_KEY 和 ANTHROPIC_API_KEY
        if not self.api_key:
            if self.llm_backend == "openai":
                self.api_key = os.environ.get("OPENAI_API_KEY", "")
            elif self.llm_backend == "claude":
                self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")

        return self

    def ensure_directories(self) -> None:
        """确保必要的目录存在。"""
        Path(self.index_dir).mkdir(parents=True, exist_ok=True)


def load_config(config_path: Optional[str] = None) -> Config:
    """加载配置。

    加载顺序（后者覆盖前者）：
    1. 默认配置
    2. 配置文件（如果指定）
    3. 环境变量

    Args:
        config_path: 配置文件路径，如果为None则使用默认路径

    Returns:
        加载完成的Config实例
    """
    config = Config()

    # 尝试从配置文件加载
    if config_path is None:
        # 查找默认配置文件
        candidates = [
            Path("treerag_config.json"),
            Path.home() / ".treerag" / "config.json",
            Path(config.index_dir) / "config.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                config_path = str(candidate)
                break

    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            file_config = json.load(f)
        config = Config.from_dict(file_config)

    # 应用环境变量覆盖
    config.apply_env_overrides()

    return config


def save_config(config: Config, path: str) -> None:
    """保存配置到文件。

    Args:
        config: 要保存的配置对象
        path: 保存路径
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(config.to_json())
