"""Combines all API routers into one include point for app/main.py."""

from fastapi import APIRouter

from app.api import health, chat

router = APIRouter()
router.include_router(health.router)
router.include_router(chat.router)
