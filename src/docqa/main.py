"""Standalone app for local / Docker use:  uvicorn docqa.main:app --port 8000

In the shared demo-api, do not use this file: mount `docqa.api.router` under /a1 instead.
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import router


def create_app() -> FastAPI:
    app = FastAPI(title="Ledgerly Support Assistant (docqa)", version="0.3.0")
    origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
    if origins:  # only needed when a browser calls the API directly (local UI development)
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                           allow_headers=["Content-Type", "X-Gateway-Secret"])
    app.include_router(router)
    return app


app = create_app()
