import os
import logging
import json
from typing import Dict, List

log = logging.getLogger(__name__)

class LLMRiskModifier:
    """
    Tier 2 Engine: Uses an LLM to evaluate the macroeconomic environment 
    based on news, and adjusts the purely technical ML prediction.
    """
    
    def __init__(self, provider: str = "openai", model_name: str = "gpt-4-turbo"):
        self.provider = provider
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        
        if not self.api_key:
            log.warning("OPENAI_API_KEY not found. LLM risk modifier will run in mock mode.")
            self.mock_mode = True
        else:
            self.mock_mode = False
            
    def assess_risk(self, ml_signal: str, ml_probability: float, news_items: List[Dict]) -> Dict:
        """
        Assess risk and suggest trading modifiers based on ML signal and recent news.
        
        Returns a dictionary:
        {
            "risk_level": "Low" | "Medium" | "High",
            "suggested_action": "Confirm Signal" | "Reduce Position Size" | "Invalidate Signal",
            "reasoning": "Explanation string"
        }
        """
        if self.mock_mode:
            return self._mock_assessment(ml_signal, ml_probability, news_items)
            
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            
            # Format news for prompt
            news_text = "\n".join([f"- {item['title']}: {item['summary']}" for item in news_items[:5]])
            if not news_text:
                news_text = "No significant news recently."
                
            prompt = f"""
You are a senior Forex quantitative risk manager. 
Our technical ML model has output the following signal:
- Signal: {ml_signal}
- Confidence: {ml_probability:.2%}

Here are the most recent macroeconomic news headlines:
{news_text}

Based on this news, assess the fundamental risk of executing this technical trade. 
Respond in strict JSON format with exactly these keys:
- "risk_level": strictly one of ["Low", "Medium", "High"]
- "suggested_action": strictly one of ["Confirm Signal", "Reduce Position Size", "Invalidate Signal"]
- "reasoning": a brief 1-2 sentence explanation of your decision.
"""
            
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a Forex risk API. You only output valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={ "type": "json_object" },
                temperature=0.2
            )
            
            content = response.choices[0].message.content
            result = json.loads(content)
            
            # Validate output
            if result.get("risk_level") not in ["Low", "Medium", "High"]:
                result["risk_level"] = "Medium"
            if result.get("suggested_action") not in ["Confirm Signal", "Reduce Position Size", "Invalidate Signal"]:
                result["suggested_action"] = "Confirm Signal"
                
            return result
            
        except ImportError:
            log.error("openai package not installed. Running in mock mode.")
            return self._mock_assessment(ml_signal, ml_probability, news_items)
        except Exception as e:
            log.error(f"LLM API call failed: {e}")
            return self._mock_assessment(ml_signal, ml_probability, news_items)
            
    def _mock_assessment(self, ml_signal: str, ml_probability: float, news_items: List[Dict]) -> Dict:
        """Fallback mock response when API key is missing or call fails."""
        if ml_probability < 0.55:
            return {
                "risk_level": "High",
                "suggested_action": "Invalidate Signal",
                "reasoning": "ML confidence is low and simulated news is volatile."
            }
        else:
            return {
                "risk_level": "Medium",
                "suggested_action": "Reduce Position Size",
                "reasoning": "Simulated environment suggests caution due to upcoming rate decisions."
            }
