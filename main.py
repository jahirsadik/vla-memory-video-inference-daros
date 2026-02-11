"""
Main entry point for video inference pipeline
"""

import argparse
import sys
from pathlib import Path

from logger import setup_logging
from video_processor import VideoInferenceProcessor


def main():
    parser = argparse.ArgumentParser(
        description="Automated video inference on multiple Vision Language Models using SGLang"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/models.yaml',
        help='Path to configuration file (default: config/models.yaml)'
    )
    
    parser.add_argument(
        '--video-url-base',
        type=str,
        default='http://localhost:8000',
        help='Base URL for video hosting (default: http://localhost:8000)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        default='inference.log',
        help='Path to log file (default: inference.log)'
    )
    
    parser.add_argument(
        '--custom-prompt',
        type=str,
        default=None,
        help='Custom system prompt for all videos (optional)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(log_level=args.log_level, log_file=args.log_file)
    
    # Check if config file exists
    if not Path(args.config).exists():
        logger.error(f"Configuration file not found: {args.config}")
        logger.info("Please create a config file at the specified path or use --config to specify a different path")
        sys.exit(1)
    
    try:
        # Initialize processor
        processor = VideoInferenceProcessor(args.config)
        
        # Process all videos
        processor.process_all_videos(
            video_url_base=args.video_url_base,
            system_prompt=args.custom_prompt
        )
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
