#!/usr/bin/env python3
"""
Setup script for video inference automation project
Creates necessary directories and validates configuration
"""

import os
import sys
from pathlib import Path
import shutil


def create_directories():
    """Create necessary project directories"""
    dirs = [
        'config',
        'videos',
        'results',
        'logs'
    ]
    
    print("Creating directories...")
    for dir_name in dirs:
        path = Path(dir_name)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ Created {dir_name}/")
        else:
            print(f"  - {dir_name}/ already exists")
    
    # Create .gitkeep files
    for dir_name in ['videos', 'results']:
        gitkeep = Path(dir_name) / '.gitkeep'
        if not gitkeep.exists():
            gitkeep.touch()


def check_dependencies():
    """Check if required packages are installed"""
    print("\nChecking dependencies...")
    
    required = {
        'requests': 'HTTP client library',
        'yaml': 'YAML parser'
    }
    
    missing = []
    for package, description in required.items():
        try:
            __import__(package)
            print(f"  ✓ {package} installed")
        except ImportError:
            print(f"  ✗ {package} NOT installed")
            missing.append(package)
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True


def check_config():
    """Check if config file exists"""
    print("\nChecking configuration...")
    
    config_path = Path('config/models.yaml')
    if config_path.exists():
        print(f"  ✓ Configuration file found at {config_path}")
        return True
    else:
        print(f"  ✗ Configuration file not found at {config_path}")
        
        # Check if config_models.yaml exists in current directory
        if Path('config_models.yaml').exists():
            print("\n  Found config_models.yaml in current directory")
            response = input("  Move it to config/models.yaml? (y/n): ")
            if response.lower() == 'y':
                shutil.move('config_models.yaml', 'config/models.yaml')
                print("  ✓ Moved to config/models.yaml")
                return True
        
        print("\n  Please create config/models.yaml")
        print("  You can start with the example in the README")
        return False


def verify_project_structure():
    """Verify essential files exist"""
    print("\nVerifying project structure...")
    
    essential_files = [
        'main.py',
        'api_client.py',
        'csv_handler.py',
        'logger.py',
        'video_processor.py',
        'README.md'
    ]
    
    missing_files = []
    for file_name in essential_files:
        if Path(file_name).exists():
            print(f"  ✓ {file_name}")
        else:
            print(f"  ✗ {file_name}")
            missing_files.append(file_name)
    
    return len(missing_files) == 0


def main():
    """Run setup"""
    print("="*60)
    print("Video Inference Automation - Project Setup")
    print("="*60)
    
    # Check if we're in the right directory
    if not Path('main.py').exists():
        print("\n✗ Error: main.py not found in current directory")
        print("Please run this script from the project root directory")
        sys.exit(1)
    
    print("\n✓ Running from project root directory\n")
    
    # Run checks
    create_directories()
    has_dependencies = check_dependencies()
    has_config = check_config()
    has_structure = verify_project_structure()
    
    # Summary
    print("\n" + "="*60)
    print("SETUP SUMMARY")
    print("="*60)
    
    if has_structure and has_dependencies:
        print("\n✓ Project structure is valid")
        print("✓ Dependencies are installed")
        
        if has_config:
            print("✓ Configuration is ready")
            print("\n🎉 Project is ready!")
            print("\nNext steps:")
            print("  1. Start SGLang server (see README.md)")
            print("  2. Place videos in videos/ directory")
            print("  3. Run: python3 main.py")
        else:
            print("⚠ Configuration file missing")
            print("\nNext steps:")
            print("  1. Create config/models.yaml")
            print("  2. Start SGLang server (see README.md)")
            print("  3. Place videos in videos/ directory")
            print("  4. Run: python3 main.py")
    else:
        print("\n✗ Project setup incomplete")
        if not has_structure:
            print("  - Missing project files")
        if not has_dependencies:
            print("  - Missing dependencies")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
