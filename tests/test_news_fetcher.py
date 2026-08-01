import unittest
from unittest.mock import patch, MagicMock
from src.data.news_fetcher import NewsFetcher

class TestNewsFetcher(unittest.TestCase):
    def setUp(self):
        self.fetcher = NewsFetcher()

    @patch('src.data.news_fetcher.requests.get')
    def test_fetch_news_mocked_api(self, mock_get):
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "feed": [
                {"title": "EUR/USD rallies heavily", "summary": "European Central Bank announces surprise rate hike."},
                {"title": "US Dollar drops", "summary": "Inflation cools down unexpectedly."}
            ]
        }
        mock_get.return_value = mock_response

        # Fetch
        self.fetcher.api_key = "FAKE_API_KEY"
        news = self.fetcher.fetch_recent_news("FOREX:EUR")
        
        # Verify
        self.assertEqual(len(news), 2)
        self.assertEqual(news[0]['title'], "EUR/USD rallies heavily")
        self.assertEqual(news[1]['title'], "US Dollar drops")
        
    def test_fetch_news_fallback_no_api_key(self):
        # Ensure it falls back to mock data if API key is not set
        self.fetcher.api_key = ""
        news = self.fetcher.fetch_recent_news("FOREX:EUR")
        
        # Verify fallback data
        self.assertEqual(len(news), 5)
        self.assertTrue(any("Federal Reserve" in item['title'] for item in news))

if __name__ == '__main__':
    unittest.main()
