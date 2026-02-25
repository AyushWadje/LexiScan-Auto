#!/usr/bin/env python
"""
LexiScan Main Launcher
Dispatches to CLI or REST API based on arguments.

Usage:
    python main.py extract <pdf_path>          # Run CLI extract command
    python main.py batch <folder_path>          # Run CLI batch command
    python main.py train --data <file>         # Run CLI train command
    python main.py info                         # Run CLI info command
    python main.py serve --host 0.0.0.0 --port 8000  # Run REST API
"""

import sys
import os
import argparse


def main():
    """Main entry point."""
    # Add project root to sys.path
    root_dir = os.path.abspath(os.path.dirname(__file__))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    # Check if first argument is "serve"
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        # Start REST API
        args_dict = {}
        
        # Parse serve-specific arguments
        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == "--host" and i + 1 < len(sys.argv):
                args_dict["host"] = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--port" and i + 1 < len(sys.argv):
                args_dict["port"] = int(sys.argv[i + 1])
                i += 2
            elif sys.argv[i] == "--reload":
                args_dict["reload"] = True
                i += 1
            else:
                i += 1
        
        host = args_dict.get("host", "0.0.0.0")
        port = args_dict.get("port", 8000)
        reload = args_dict.get("reload", False)
        
        try:
            import uvicorn
            from lexiscan.api import app

            print(f"\n{'='*60}")
            print("Starting LexiScan REST API")
            print(f"{'='*60}")
            print(f"Host: {host}")
            print(f"Port: {port}")
            print(f"Docs: http://{host if host != '0.0.0.0' else 'localhost'}:{port}/docs")
            print(f"{'='*60}\n")

            uvicorn.run(
                "lexiscan.api:app",
                host=host,
                port=port,
                reload=reload,
            )
        except ImportError as e:
            print(f"Error: FastAPI/Uvicorn not installed: {e}")
            print("Install with: pip install fastapi uvicorn")
            sys.exit(1)
    else:
        # Delegate to CLI for all other commands
        from lexiscan.cli import main as cli_main
        cli_main()


if __name__ == "__main__":
    main()
