import logging
from typing import List, Dict, Tuple
import numpy as np

log = logging.getLogger(__name__)

class SentimentExtractor:
    """
    Extracts sentiment from financial news text using FinBERT.
    Gracefully falls back to neutral scores if transformers is not installed.
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self._load_model()

    def _load_model(self):
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            log.info("Loading FinBERT model for sentiment extraction...")
            model_name = "ProsusAI/finbert"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.is_loaded = True
            log.info(f"FinBERT loaded on {self.device}")
            
        except ImportError:
            log.warning("transformers or torch not installed. Sentiment will return neutral (0).")
        except Exception as e:
            log.error(f"Failed to load FinBERT: {e}")

    def analyze_sentiment(self, texts: List[str]) -> Tuple[List[float], List[str]]:
        """
        Analyzes a list of texts and returns a list of sentiment scores [-1 to 1]
        and their labels (negative, neutral, positive).
        """
        if not texts:
            return [], []
            
        if not self.is_loaded:
            # Fallback to neutral if model fails to load
            return [0.0] * len(texts), ["neutral"] * len(texts)
            
        import torch
        
        scores = []
        labels = []
        label_map = {0: "positive", 1: "negative", 2: "neutral"}
        
        try:
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            for pred in predictions:
                pos_score = pred[0].item()
                neg_score = pred[1].item()
                
                # Calculate a continuous score: +1 for 100% positive, -1 for 100% negative
                continuous_score = pos_score - neg_score
                scores.append(continuous_score)
                
                class_idx = torch.argmax(pred).item()
                labels.append(label_map.get(class_idx, "neutral"))
                
        except Exception as e:
            log.error(f"Error during sentiment analysis: {e}")
            scores = [0.0] * len(texts)
            labels = ["neutral"] * len(texts)
            
        return scores, labels

    def aggregate_sentiment(self, news_items: List[Dict]) -> float:
        """
        Takes a list of news dictionaries (from news_fetcher) and returns a single
        aggregated sentiment score between -1 and 1.
        """
        if not news_items:
            return 0.0
            
        texts = [f"{item.get('title', '')} {item.get('summary', '')}".strip() for item in news_items]
        texts = [t for t in texts if t] # Filter empty
        
        if not texts:
            return 0.0
            
        scores, _ = self.analyze_sentiment(texts)
        
        # Simple average, could be weighted by time
        return float(np.mean(scores))
