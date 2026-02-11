"""
CSV handler for saving inference results
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class CSVResultHandler:
    """Handles saving inference results to CSV files"""
    
    def __init__(self, results_dir: str):
        """
        Initialize CSV handler
        
        Args:
            results_dir: Directory where CSV files will be saved
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
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
        error_message: str = None
    ) -> bool:
        """
        Save a single inference result to CSV
        
        Args:
            video_name: Name of the video file
            model_name: Display name of the model
            model_path: Model path/identifier
            response: Model's response text
            status: Status of the inference (success/error)
            error_message: Error message if status is error
            
        Returns:
            True if successful, False otherwise
        """
        try:
            csv_path = self.get_csv_path(video_name)
            file_exists = csv_path.exists()
            
            with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'timestamp',
                    'video_name',
                    'model_name',
                    'model_path',
                    'status',
                    'response',
                    'error_message'
                ]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # Write header only if file is new
                if not file_exists:
                    writer.writeheader()
                    logger.info(f"Created new CSV file: {csv_path}")
                
                # Write the result row
                writer.writerow({
                    'timestamp': datetime.now().isoformat(),
                    'video_name': video_name,
                    'model_name': model_name,
                    'model_path': model_path,
                    'status': status,
                    'response': response if response else '',
                    'error_message': error_message if error_message else ''
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
