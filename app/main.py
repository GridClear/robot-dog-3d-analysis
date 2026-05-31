"""Multi-image → splat reconstruction service."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import health, jobs
from app.services import worker


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await worker.start_worker()
    yield
    await worker.stop_worker()


app = FastAPI(title="robot-dog 3d splat", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(health.router)
app.include_router(jobs.router)
