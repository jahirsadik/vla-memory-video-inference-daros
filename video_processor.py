"""
Video processor for orchestrating multi-model inference
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml

from api_client import SGLangAPIClient
from csv_handler import CSVResultHandler

logger = logging.getLogger(__name__)


class VideoInferenceProcessor:
    """Main processor for video inference on multiple models"""
    
    # Default system prompt - customize this for your use case
    DEFAULT_SYSTEM_PROMPT = (
        "Act as a precise spatial analyst. Watch the robot start from a table on one end of the corridor, "
        "move down the corridor to the other end, where there is another table, and move back to the starting position again. "
        "Your task is to count unique cubes on the corridor floor that you see along the way. Note that, colored cubes may repeat, "
        "for example, there may be two Red colored cubes, in this case you count both. But you do not count the exact same cube "
        "(where both the color and placement in the corridor is the same) twice. Be concise with your thinking process and answer.\n"
        "Output format:\n"
        "Reasoning: [Identify of unique cubes described with relative spatiotemporal context]\n"
        "Final Count: [Integer]"
    )
    
    def __init__(self, config_path: str):
        """
        Initialize the processor
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.csv_handler = CSVResultHandler(self.config['directories']['results'])
        self.models: List[Dict[str, Any]] = []
        self.api_clients: Dict[str, SGLangAPIClient] = {}
        self._initialize_models()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"✓ Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"✗ Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"✗ Error parsing YAML: {e}")
            raise
    
    def _initialize_models(self):
        """Initialize API clients for enabled models"""
        self.models = [m for m in self.config['models'] if m.get('enabled', True)]
        
        logger.info(f"Initializing {len(self.models)} model(s)...")
        
        for model in self.models:
            client = SGLangAPIClient(
                host=model['host'],
                port=model['port'],
                model_path=model['model_path'],
                timeout=self.config['inference'].get('timeout', 60)
            )
            self.api_clients[model['name']] = client
        
        logger.info(f"✓ {len(self.api_clients)} model client(s) initialized")
    
    def get_video_files(self) -> List[Path]:
        """Get all video files from the videos directory"""
        videos_dir = Path(self.config['directories']['videos'])
        
        if not videos_dir.exists():
            logger.warning(f"Videos directory does not exist: {videos_dir}")
            return []
        
        # Common video extensions
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        
        video_files = [
            f for f in videos_dir.glob('*')
            if f.is_file() and f.suffix.lower() in video_extensions
        ]
        
        logger.info(f"Found {len(video_files)} video file(s)")
        return sorted(video_files)
    
    def check_all_servers(self) -> Dict[str, bool]:
        """Check health of all configured servers"""
        logger.info("Checking server health...")
        health_status = {}
        
        for model_name, client in self.api_clients.items():
            is_healthy = client.health_check()
            health_status[model_name] = is_healthy
        
        return health_status
    
    def process_video(
        self,
        video_path: Path,
        video_url: str,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a single video on all enabled models
        
        Args:
            video_path: Local path to video file
            video_url: URL to video file (for API)
            system_prompt: Custom prompt (uses default if None)
            
        Returns:
            Dictionary with results for each model
        """
        if system_prompt is None:
            system_prompt = self.DEFAULT_SYSTEM_PROMPT
        
        results = {
            'video_name': video_path.name,
            'models': {}
        }
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing video: {video_path.name}")
        logger.info(f"{'='*60}")
        
        for model in self.models:
            model_name = model['name']
            logger.info(f"\n▶ Running inference with {model_name}...")
            
            try:
                client = self.api_clients[model_name]
                response = client.infer_video(
                    video_url=video_url,
                    system_prompt=system_prompt,
                    max_tokens=self.config['inference'].get('max_tokens', 1024),
                    temperature=self.config['inference'].get('temperature', 0.0)
                )
                
                if response:
                    results['models'][model_name] = {
                        'status': 'success',
                        'response': response
                    }
                    self.csv_handler.save_result(
                        video_name=video_path.name,
                        model_name=model_name,
                        model_path=model['model_path'],
                        response=response,
                        status='success'
                    )
                    logger.info(f"✓ {model_name} completed successfully")
                else:
                    results['models'][model_name] = {
                        'status': 'error',
                        'response': None
                    }
                    self.csv_handler.save_result(
                        video_name=video_path.name,
                        model_name=model_name,
                        model_path=model['model_path'],
                        response=None,
                        status='error',
                        error_message='API returned no response'
                    )
                    logger.error(f"✗ {model_name} failed to process video")
                    
            except Exception as e:
                logger.error(f"✗ Unexpected error with {model_name}: {e}")
                results['models'][model_name] = {
                    'status': 'error',
                    'response': None
                }
                self.csv_handler.save_result(
                    video_name=video_path.name,
                    model_name=model_name,
                    model_path=model['model_path'],
                    response=None,
                    status='error',
                    error_message=str(e)
                )
        
        return results
    
    def process_all_videos(
        self,
        video_url_base: str = "http://localhost:8000",
        system_prompt: Optional[str] = None
    ):
        """
        Process all videos in the videos directory
        
        Args:
            video_url_base: Base URL for video hosting (e.g., http://localhost:8000)
            system_prompt: Custom prompt for all videos
        """
        logger.info("\n" + "="*60)
        logger.info("STARTING VIDEO INFERENCE PIPELINE")
        logger.info("="*60)
        
        # Check server health first
        health = self.check_all_servers()
        unhealthy = [name for name, healthy in health.items() if not healthy]
        
        if unhealthy:
            logger.warning(f"\n⚠ Warning: The following servers are not healthy: {unhealthy}")
            logger.warning("Proceeding anyway, but some models may fail...")
        
        # Get all videos
        video_files = self.get_video_files()
        if not video_files:
            logger.error("No video files found. Exiting.")
            return
        
        # Process each video
        all_results = []
        for idx, video_path in enumerate(video_files, 1):
            logger.info(f"\n[{idx}/{len(video_files)}]")
            
            # Construct video URL
            video_url = f"{video_url_base}/{video_path.name}"
            
            # Process video on all models
            result = self.process_video(
                video_path=video_path,
                video_url=video_url,
                system_prompt=system_prompt
            )
            all_results.append(result)
        
        # Print summary
        self._print_summary(all_results)
    
    def _print_summary(self, results: List[Dict[str, Any]]):
        """Print summary of inference results"""
        logger.info("\n" + "="*60)
        logger.info("INFERENCE SUMMARY")
        logger.info("="*60)
        
        for result in results:
            logger.info(f"\nVideo: {result['video_name']}")
            for model_name, model_result in result['models'].items():
                status = model_result['status'].upper()
                logger.info(f"  {model_name}: {status}")
        
        # CSV summary
        csv_summary = self.csv_handler.summary()
        logger.info(f"\n📊 Results saved:")
        logger.info(f"  Total CSV files: {csv_summary['total_csv_files']}")
        logger.info(f"  Total results: {csv_summary['total_results']}")
        logger.info(f"  Output directory: {self.config['directories']['results']}")
        
        logger.info("\n" + "="*60)
        logger.info("✓ PIPELINE COMPLETED")
        logger.info("="*60)
