from fastapi import APIRouter
from .medical_qa import router as medical_qa_router
from .sentiment import router as sentiment_router

router = APIRouter()
router.include_router(medical_qa_router, tags=["Medical QA"])
router.include_router(sentiment_router, tags=["Sentiment Analysis"])
