"""
CLI命令行入口模块

提供TreeRAG的命令行接口，支持索引构建、搜索、Web服务等操作。
使用Click框架实现，Rich库美化输出。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.prompt import Confirm

from . import __version__
from .config import Config, load_config, save_config
from .parser import DocumentParser
from .indexer import TreeIndexer
from .searcher import TreeSearcher

console = Console()


def _load_config_with_cli_overrides(
    config_path: Optional[str],
    backend: Optional[str],
    model: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
) -> Config:
    """加载配置并应用CLI参数覆盖。"""
    config = load_config(config_path)
    if backend:
        config.llm_backend = backend
    if model:
        config.model = model
    if base_url:
        config.base_url = base_url
    if api_key:
        config.api_key = api_key
    return config


@click.group()
@click.version_option(version=__version__, prog_name="treerag")
@click.option(
    "--config", "-c",
    default=None,
    help="配置文件路径",
)
@click.option(
    "--backend", "-b",
    type=click.Choice(["openai", "claude", "ollama"]),
    default=None,
    help="LLM后端类型",
)
@click.option(
    "--model", "-m",
    default=None,
    help="LLM模型名称",
)
@click.option(
    "--base-url",
    default=None,
    help="API基础URL",
)
@click.option(
    "--api-key",
    default=None,
    help="API密钥",
)
@click.pass_context
def cli(
    ctx: click.Context,
    config: Optional[str],
    backend: Optional[str],
    model: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
) -> None:
    """TreeRAG - 基于树索引的智能文档检索引擎

    通过LLM推理构建层次化树索引来检索文档内容，替代传统向量化RAG方案。
    """
    ctx.ensure_object(dict)
    ctx.obj["config"] = _load_config_with_cli_overrides(
        config, backend, model, base_url, api_key
    )


@cli.command()
@click.argument("file_or_dir", type=click.Path(exists=True))
@click.option(
    "--output", "-o",
    default=None,
    help="索引输出路径",
)
@click.option(
    "--group-size", "-g",
    default=None,
    type=int,
    help="分组大小",
)
@click.pass_context
def index(
    ctx: click.Context,
    file_or_dir: str,
    output: Optional[str],
    group_size: Optional[int],
) -> None:
    """索引文件或目录。

    将指定的文件或目录中的文档构建为层次化树索引。
    支持PDF、Markdown、TXT、DOCX格式。
    """
    config: Config = ctx.obj["config"]
    config.ensure_directories()

    # 确定输出路径
    if output is None:
        output = str(Path(config.index_dir) / "index.json")

    # 覆盖分组大小
    if group_size is not None:
        config.group_size = group_size

    # 收集文件
    path = Path(file_or_dir)
    files: list[Path] = []

    if path.is_file():
        files.append(path)
    elif path.is_dir():
        supported_ext = {".pdf", ".md", ".markdown", ".txt", ".text", ".docx"}
        for f in path.iterdir():
            if f.is_file() and f.suffix.lower() in supported_ext:
                files.append(f)

    if not files:
        console.print("[red]未找到可索引的文件[/red]")
        sys.exit(1)

    console.print(Panel(
        f"准备索引 {len(files)} 个文件\n"
        f"输出路径: {output}\n"
        f"LLM后端: {config.llm_backend} ({config.model})",
        title="索引任务",
        border_style="blue",
    ))

    # 解析文档
    parser = DocumentParser(max_segment_length=config.max_segment_length)
    all_documents = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("解析文档...", total=len(files))

        for file_path in files:
            progress.update(task, description=f"解析: {file_path.name}")
            try:
                docs = parser.parse(str(file_path))
                all_documents.extend(docs)
            except Exception as e:
                console.print(f"[yellow]警告: 解析 {file_path.name} 失败: {e}[/yellow]")
            progress.advance(task)

    if not all_documents:
        console.print("[red]未能解析出任何文档内容[/red]")
        sys.exit(1)

    console.print(f"[green]共解析出 {len(all_documents)} 个文档片段[/green]")

    # 构建索引
    console.print("[cyan]开始构建树索引（这可能需要一些时间）...[/cyan]")
    indexer = TreeIndexer(config=config)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("构建树索引...", total=None)
        root = indexer.build_index(all_documents)

    # 保存索引
    indexer.save_index(root, output)

    # 显示统计
    stats = indexer.get_stats(root, output)
    _display_stats(stats)

    console.print(f"\n[green]索引已成功保存到: {output}[/green]")


@cli.command()
@click.argument("query")
@click.option(
    "--top-k", "-k",
    default=None,
    type=int,
    help="返回结果数量",
)
@click.option(
    "--index-path", "-i",
    default=None,
    help="索引文件路径",
)
@click.option(
    "--no-cache",
    is_flag=True,
    help="禁用缓存",
)
@click.pass_context
def search(
    ctx: click.Context,
    query: str,
    top_k: Optional[int],
    index_path: Optional[str],
    no_cache: bool,
) -> None:
    """搜索已索引的文档。

    使用树索引搜索与查询相关的文档内容。
    """
    config: Config = ctx.obj["config"]

    # 确定索引路径
    if index_path is None:
        index_path = str(Path(config.index_dir) / "index.json")

    if not Path(index_path).exists():
        console.print(f"[red]索引文件不存在: {index_path}[/red]")
        console.print("请先使用 'treerag index' 命令构建索引")
        sys.exit(1)

    # 加载索引
    console.print(f"[cyan]加载索引: {index_path}[/cyan]")
    indexer = TreeIndexer(config=config)
    root = indexer.load_index(index_path)

    # 创建搜索引擎
    if no_cache:
        config.cache_enabled = False
    searcher = TreeSearcher(config=config, index=root)

    # 执行搜索
    actual_top_k = top_k if top_k is not None else config.top_k

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("搜索中...", total=None)
        results = searcher.search(query, top_k=actual_top_k)

    # 显示结果
    if not results:
        console.print("[yellow]未找到相关结果[/yellow]")
        return

    console.print(Panel(
        f"查询: {query}\n找到 {len(results)} 个结果",
        title="搜索结果",
        border_style="green",
    ))

    for i, result in enumerate(results, 1):
        # 判断是否为综合答案
        is_answer = result.metadata.get("type") == "extracted_answer"

        if is_answer:
            console.print(f"\n[bold cyan]综合答案:[/bold cyan]")
            console.print(Panel(result.content, border_style="cyan"))
        else:
            score_color = "green" if result.score > 0.7 else "yellow" if result.score > 0.4 else "red"
            console.print(f"\n[bold]{i}.[/bold] ", end="")
            console.print(f"[{score_color}]相关性: {result.score:.2f}[/{score_color}]")

            if result.source:
                console.print(f"   来源: {result.source}")
            if result.metadata.get("page"):
                console.print(f"   页码: {result.metadata['page']}")
            if result.metadata.get("section"):
                console.print(f"   章节: {result.metadata['section']}")

            # 显示内容预览
            preview = result.content[:300]
            if len(result.content) > 300:
                preview += "..."
            console.print(f"   内容: {preview}")

    # 显示缓存统计
    if config.cache_enabled:
        cache_stats = searcher.get_cache_stats()
        console.print(
            f"\n[dim]缓存: {cache_stats['hits']} 命中 / "
            f"{cache_stats['total_requests']} 请求 "
            f"({cache_stats['hit_rate']:.1%})[/dim]"
        )


@cli.command()
@click.option(
    "--host", "-h",
    default=None,
    help="监听地址",
)
@click.option(
    "--port", "-p",
    default=None,
    type=int,
    help="监听端口",
)
@click.pass_context
def serve(ctx: click.Context, host: Optional[str], port: Optional[int]) -> None:
    """启动Web服务。

    提供基于FastAPI的Web界面和API服务。
    """
    config: Config = ctx.obj["config"]

    if host:
        config.host = host
    if port:
        config.port = port

    console.print(Panel(
        f"Web服务启动中...\n"
        f"地址: http://{config.host}:{config.port}",
        title="TreeRAG Web",
        border_style="blue",
    ))

    try:
        import uvicorn
        from ..web.app import create_app

        app = create_app(config)
        uvicorn.run(app, host=config.host, port=config.port)
    except ImportError as e:
        console.print(f"[red]缺少依赖: {e}[/red]")
        console.print("请安装Web服务依赖: pip install fastapi uvicorn")
        sys.exit(1)


@cli.command()
@click.option(
    "--index-path", "-i",
    default=None,
    help="索引文件路径",
)
@click.pass_context
def stats(ctx: click.Context, index_path: Optional[str]) -> None:
    """显示索引统计信息。"""
    config: Config = ctx.obj["config"]

    if index_path is None:
        index_path = str(Path(config.index_dir) / "index.json")

    if not Path(index_path).exists():
        console.print(f"[red]索引文件不存在: {index_path}[/red]")
        sys.exit(1)

    indexer = TreeIndexer(config=config)
    root = indexer.load_index(index_path)
    index_stats = indexer.get_stats(root, index_path)

    _display_stats(index_stats)


def _display_stats(stats) -> None:
    """显示索引统计信息的辅助函数。"""
    table = Table(title="索引统计")
    table.add_column("指标", style="cyan")
    table.add_column("值", style="green")

    table.add_row("总节点数", str(stats.total_nodes))
    table.add_row("文档片段数", str(stats.total_documents))
    table.add_row("最大深度", str(stats.max_depth))
    table.add_row("索引大小", f"{stats.index_size:,} 字节")

    console.print(table)


@cli.command()
@click.argument("action", type=click.Choice(["show", "set", "reset"]))
@click.option(
    "--key",
    default=None,
    help="配置键名",
)
@click.option(
    "--value",
    default=None,
    help="配置值",
)
@click.option(
    "--path",
    default=None,
    help="配置文件路径",
)
@click.pass_context
def config_cmd(
    ctx: click.Context,
    action: str,
    key: Optional[str],
    value: Optional[str],
    path: Optional[str],
) -> None:
    """配置管理。

    查看、设置或重置TreeRAG配置。
    """
    config: Config = ctx.obj["config"]

    if action == "show":
        # 显示当前配置
        table = Table(title="当前配置")
        table.add_column("配置项", style="cyan")
        table.add_column("值", style="green")

        config_dict = config.to_dict()
        for k, v in config_dict.items():
            # 隐藏敏感信息
            if "key" in k.lower() and v:
                display_val = "***" + v[-4:] if len(v) > 4 else "***"
            else:
                display_val = str(v)
            table.add_row(k, display_val)

        console.print(table)

    elif action == "set":
        if not key or value is None:
            console.print("[red]请指定 --key 和 --value[/red]")
            sys.exit(1)

        if not hasattr(config, key):
            console.print(f"[red]无效的配置项: {key}[/red]")
            sys.exit(1)

        # 类型转换
        current_type = type(getattr(config, key))
        try:
            if current_type == bool:
                converted = value.lower() in ("true", "1", "yes")
            elif current_type in (int, float):
                converted = current_type(value)
            else:
                converted = value
            setattr(config, key, converted)
        except (ValueError, TypeError) as e:
            console.print(f"[red]值转换失败: {e}[/red]")
            sys.exit(1)

        # 保存配置
        save_path = path or str(Path(config.index_dir) / "config.json")
        save_config(config, save_path)
        console.print(f"[green]配置已更新: {key} = {converted}[/green]")
        console.print(f"[dim]保存到: {save_path}[/dim]")

    elif action == "reset":
        default_config = Config()
        save_path = path or str(Path(config.index_dir) / "config.json")
        save_config(default_config, save_path)
        console.print(f"[green]配置已重置为默认值[/green]")
        console.print(f"[dim]保存到: {save_path}[/dim]")


def main() -> None:
    """CLI主入口函数。"""
    cli()


if __name__ == "__main__":
    main()
