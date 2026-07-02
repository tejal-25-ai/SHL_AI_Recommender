"""GET /health — readiness check. Deliberately has zero dependencies
on catalog/model loading so it responds instantly even during a cold
start, per the doc's 2-minute wake-up allowance for the FIRST call
(this endpoint itself must not be what's slow)."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}
