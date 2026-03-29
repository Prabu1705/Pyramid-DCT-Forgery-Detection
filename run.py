#!/usr/bin/env python3
"""
run.py
------
Entry point to launch the Pyramid DCT Forgery Detection server.

Usage:
    python run.py [--host HOST] [--port PORT] [--reload]

Defaults:
    host = 0.0.0.0
    port = 8000
"""

import argparse
import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pyramid DCT Forgery Detection Server")
    parser.add_argument("--host",   default="0.0.0.0",  help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port",   default=8000, type=int, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true",  help="Enable hot-reload (dev mode)")
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════╗
║      Pyramid DCT Forgery Detection System  v1.0          ║
║      http://{args.host}:{args.port}                      ║
║      API docs → http://{args.host}:{args.port}/api/docs  ║
╚══════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
