import unittest
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

from src.features.sentiment_extractor import SentimentExtractor

class TestSentimentExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = SentimentExtractor()

    def test_bullish_headline(self):
        headline = "Company reports record profits and raises guidance for the year, sending shares soaring."
        score = self.extractor.aggregate_sentiment([{"title": headline, "summary": ""}])
        self.assertGreater(score, 0.5, "Expected a highly bullish sentiment score > 0.5")

    def test_bearish_headline(self):
        headline = "Unemployment skyrockets, inflation hits new highs, and the economy collapses into a severe recession."
        score = self.extractor.aggregate_sentiment([{"title": headline, "summary": ""}])
        self.assertLess(score, -0.5, "Expected a highly bearish sentiment score < -0.5")

    def test_neutral_headline(self):
        headline = "The bank will release its regular quarterly earnings report tomorrow."
        score = self.extractor.aggregate_sentiment([{"title": headline, "summary": ""}])
        self.assertTrue(-0.2 <= score <= 0.2, f"Expected a neutral sentiment score, got {score}")

if __name__ == '__main__':
    unittest.main()
