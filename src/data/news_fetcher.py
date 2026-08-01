import logging
import requests
import json
from typing import List, Dict, Optional
import os
from datetime import datetime, timedelta

log = logging.getLogger(__name__)

class NewsFetcher:
    """
    Fetches latest financial news for specific currency pairs or macroeconomic themes.
    Uses AlphaVantage or a generic mock if no API key is provided.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("NEWS_API_KEY", "demo")
        self.base_url = "https://www.alphavantage.co/query"

    def fetch_recent_news(self, tickers: str = "FOREX:EUR", limit: int = 10) -> List[Dict]:
        """
        Fetches news sentiment data.
        Returns a list of dictionaries with 'title', 'summary', and 'time_published'.
        """
        log.info(f"Fetching news for {tickers}...")
        
        # If we only have the demo key or it's empty, use mock data to prevent crashes
        if self.api_key == "demo" or not self.api_key:
            log.warning("No NEWS_API_KEY provided. Using mock news data.")
            return self._get_mock_news(tickers, limit)
            
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": tickers,
            "apikey": self.api_key,
            "limit": limit
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if "feed" in data:
                news_items = []
                for item in data["feed"][:limit]:
                    news_items.append({
                        "title": item.get("title", ""),
                        "summary": item.get("summary", ""),
                        "time_published": item.get("time_published", ""),
                        "url": item.get("url", "")
                    })
                return news_items
            else:
                log.warning(f"Unexpected API response structure: {data.keys()}")
                return self._get_mock_news(tickers, limit)
                
        except Exception as e:
            log.error(f"Failed to fetch news: {e}")
            return self._get_mock_news(tickers, limit)
            
    def _get_mock_news(self, tickers: str, limit: int) -> List[Dict]:
        """Provides fallback mock news for testing."""
        now = datetime.now()
        mock_headlines = [
            "Federal Reserve signals unexpected rate hike, sparking market volatility.",
            "European Central Bank maintains rates, citing stable inflation.",
            "US CPI data comes in higher than expected, strengthening the Dollar.",
            "Global trade tensions ease as new tariff agreements are reached.",
            "Unemployment rate drops slightly, showing a resilient labor market."
        ]
        
        results = []
        for i in range(min(limit, len(mock_headlines))):
            results.append({
                "title": mock_headlines[i],
                "summary": "Mock summary for the headline to simulate API response.",
                "time_published": (now - timedelta(hours=i)).strftime("%Y%m%dT%H%M%S"),
                "url": "https://example.com/news"
            })
        return results
