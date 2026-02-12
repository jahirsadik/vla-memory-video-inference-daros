"""
CSV handler for saving inference results with thread-safe file locking
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import threading

logger = logging.getLogger(__name__)


class CSVResultHandler:
    """Handles saving inference results to CSV files with thread-safe locking"""
    
    def __init__(self, results_dir: str):
        """
        Initialize CSV handler
        
        Args:
            results_dir: Directory where CSV files will be saved
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # File locks per CSV file to prevent concurrent writes
        self._locks: Dict[str, threading.Lock] = {}
        self._lock_manager = threading.Lock()
    
    def _get_lock(self, csv_path: str) -> threading.Lock:
        """
        Get or create a lock for a specific CSV file
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            Threading lock for that file
        """
        with self._lock_manager:
            if csv_path not in self._locks:
                self._locks[csv_path] = threading.Lock()
            return self._locks[csv_path]
    
    def get_csv_path(self, video_name: str) -> Path:
        """
        Get the CSV file path for a given video
        
        Args:
            video_name: Name of the video file
            
        Returns:
            Path to the CSV file
        """
        video_stem = Path(video_name).stem
        csv_filename = f"{video_stem}_results.csv"
        return self.results_dir / csv_filename
    
    def save_result(
        self,
        video_name: str,
        model_name: str,
        model_path: str,
        response: str,
        status: str = "success",
        error_message: str = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        inference_time_seconds: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save a single inference result to CSV (thread-safe)
        
        Args:
            video_name: Name of the video file
            model_name: Display name of the model
            model_path: Model path/identifier
            response: Model's response text
            status: Status of the inference (success/error)
            error_message: Error message if status is error
            temperature: Sampling temperature used
            max_tokens: Maximum tokens used
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            inference_time_seconds: Time taken for inference
            metadata: Additional metadata as JSON string or dict
            
        Returns:
            True if successful, False otherwise
        """
        try:
            csv_path = self.get_csv_path(video_name)
            file_exists = csv_path.exists()
            
            # Use file lock to prevent concurrent writes
            lock = self._get_lock(str(csv_path))
            
            with lock:
                with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                    fieldnames = [
                        'timestamp',
                        'video_name',
                        'model_name',
                        'model_path',
                        'status',
                        'temperature',
                        'max_tokens',
                        'top_k',
                        'top_p',
                        'inference_time_seconds',
                        'response',
                        'error_message',
                        'metadata'
                    ]
                    
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                    # Write header only if file is new
                    if not file_exists:
                        writer.writeheader()
                        logger.info(f"Created new CSV file: {csv_path}")
                    
                    # Handle metadata
                    metadata_str = ''
                    if metadata:
                        if isinstance(metadata, dict):
                            metadata_str = json.dumps(metadata)
                        else:
                            metadata_str = str(metadata)
                    
                    # Write the result row
                    writer.writerow({
                        'timestamp': datetime.now().isoformat(),
                        'video_name': video_name,
                        'model_name': model_name,
                        'model_path': model_path,
                        'status': status,
                        'temperature': temperature if temperature is not None else '',
                        'max_tokens': max_tokens if max_tokens is not None else '',
                        'top_k': top_k if top_k is not None else '',
                        'top_p': top_p if top_p is not None else '',
                        'inference_time_seconds': inference_time_seconds if inference_time_seconds is not None else '',
                        'response': response if response else '',
                        'error_message': error_message if error_message else '',
                        'metadata': metadata_str
                    })
                
            logger.info(f"✓ Result saved to {csv_path}")
            return True
            
        except IOError as e:
            logger.error(f"✗ Failed to write to CSV: {e}")
            return False
        except Exception as e:
            logger.error(f"✗ Unexpected error while saving result: {e}")
            return False
    
    def get_all_csv_files(self) -> List[Path]:
        """Get all CSV files in the results directory"""
        return list(self.results_dir.glob("*_results.csv"))
    
    def summary(self) -> Dict[str, Any]:
        """Get a summary of all results"""
        summary = {
            'total_csv_files': 0,
            'total_results': 0,
            'files': {}
        }
        
        for csv_path in self.get_all_csv_files():
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    summary['total_csv_files'] += 1
                    summary['total_results'] += len(rows)
                    summary['files'][csv_path.name] = len(rows)
            except Exception as e:
                logger.error(f"Error reading {csv_path}: {e}")
        
        return summary
