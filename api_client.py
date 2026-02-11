"""
SGLang API client for video inference
"""

import requests
import json
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class SGLangAPIClient:
    """Client for communicating with SGLang server"""
    
    def __init__(self, host: str, port: int, model_path: str, timeout: int = 60):
        """
        Initialize SGLang API client
        
        Args:
            host: Server host address
            port: Server port
            model_path: Model identifier
            timeout: Request timeout in seconds
        """
        self.base_url = f"http://{host}:{port}"
        self.model_path = model_path
        self.timeout = timeout
        
    def health_check(self) -> bool:
        """Check if SGLang server is running"""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=5
            )
            is_healthy = response.status_code == 200
            if is_healthy:
                logger.info(f"✓ Server at {self.base_url} is healthy")
            else:
                logger.warning(f"✗ Server at {self.base_url} returned status {response.status_code}")
            return is_healthy
        except requests.exceptions.RequestException as e:
            logger.error(f"✗ Cannot reach server at {self.base_url}: {e}")
            return False
    
    def infer_video(
        self,
        video_url: str,
        system_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0
    ) -> Optional[str]:
        """
        Run inference on a video
        
        Args:
            video_url: URL to the video file
            system_prompt: The prompt/instruction to give to the model
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            
        Returns:
            Model response text or None if failed
        """
        try:
            payload = {
                "model": self.model_path,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": system_prompt
                            },
                            {
                                "type": "video_url",
                                "video_url": {"url": video_url}
                            }
                        ]
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            logger.debug(f"Sending request to {self.base_url}/v1/chat/completions")
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                logger.info(f"✓ Inference successful for video")
                return content
            else:
                logger.error(f"✗ API returned status {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error(f"✗ Request timeout after {self.timeout}s")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"✗ Request failed: {e}")
            return None
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"✗ Failed to parse response: {e}")
            return None
