import argparse

import uvicorn

from .config import Settings
from .server import create_app


def main():
    parser = argparse.ArgumentParser(description="Run the local voice gateway")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    settings = Settings.from_env()
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not settings.token:
        parser.error("Set SPEED_TOKEN before listening outside loopback")
    uvicorn.run(
        create_app(settings), host=args.host, port=args.port, ws_max_size=65536, ws_max_queue=8
    )
