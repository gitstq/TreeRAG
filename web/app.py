"""
FastAPI Web应用模块

创建和配置FastAPI应用实例，包含CORS支持、静态文件服务等。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from treerag.config import Config


def create_app(config: Optional[Config] = None) -> FastAPI:
    """创建FastAPI应用实例。

    Args:
        config: 配置对象，如果为None则使用默认配置

    Returns:
        配置完成的FastAPI应用实例
    """
    if config is None:
        from treerag.config import load_config
        config = load_config()

    app = FastAPI(
        title="TreeRAG API",
        description="基于树索引的智能文档检索引擎 API",
        version="1.0.0",
    )

    # CORS中间件配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册路由
    from .routes import create_router
    api_router = create_router(config)
    app.include_router(api_router, prefix="/api")

    # 挂载静态文件（Web UI模板）
    templates_dir = Path(__file__).parent / "templates"
    if templates_dir.exists():
        from fastapi.staticfiles import StaticFiles
        from fastapi.responses import FileResponse

        @app.get("/")
        async def serve_index():
            """提供Web UI首页。"""
            index_path = templates_dir / "index.html"
            if index_path.exists():
                return FileResponse(str(index_path))
            return {"message": "Web UI template not found"}

    # 存储配置供路由使用
    app.state.config = config

    return app
