import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch # CRITICAL: Must be imported before pandas on this system to prevent silent abort

import logging
import joblib
import numpy as np
import pandas as pd

from src.data.news_fetcher import NewsFetcher
from src.features.sentiment_extractor import SentimentExtractor
from src.models.llm_risk_modifier import LLMRiskModifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def run_inference_pipeline():
    """
    Demonstrates the end-to-end inference pipeline using the LLM Sentiment Engine
    and Risk Modifier on top of quantitative predictions.
    """
    log.info("Starting LLM-enhanced Forex Inference Pipeline...")
    
    # 1. Fetch News
    fetcher = NewsFetcher()
    news_items = fetcher.fetch_recent_news(tickers="FOREX:EUR", limit=5)
    log.info(f"Fetched {len(news_items)} news items.")
    
    # 2. Extract Sentiment
    sentiment_engine = SentimentExtractor()
    overall_sentiment = sentiment_engine.aggregate_sentiment(news_items)
    log.info(f"Aggregated Market Sentiment Score: {overall_sentiment:.2f} (Scale: -1 to 1)")
    
    # 3. Load Quantitative Model (Mocking the ensemble inference for demonstration)
    # In a real scenario, you'd load xgb_model, lgb_model, and meta_lr, and call predict()
    # Here we mock a generic prediction based on the sentiment for demonstration
    log.info("Running quantitative ML ensemble...")
    base_ml_prediction_prob = 0.5 + (overall_sentiment * 0.2) # Mock fusion
    base_signal = "BUY" if base_ml_prediction_prob >= 0.5 else "SELL"
    log.info(f"Quantitative ML Signal: {base_signal} (Confidence: {base_ml_prediction_prob:.2%})")
    
    # 4. LLM Risk Assessment
    llm_modifier = LLMRiskModifier()
    risk_assessment = llm_modifier.assess_risk(
        ml_signal=base_signal,
        ml_probability=base_ml_prediction_prob,
        news_items=news_items
    )
    
    log.info("=== FINAL TRADING DECISION ===")
    log.info(f"Base ML Signal    : {base_signal}")
    log.info(f"Risk Level        : {risk_assessment.get('risk_level')}")
    log.info(f"Suggested Action  : {risk_assessment.get('suggested_action')}")
    log.info(f"LLM Reasoning     : {risk_assessment.get('reasoning')}")
    log.info("==============================")

if __name__ == "__main__":
    run_inference_pipeline()
