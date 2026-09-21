"""``python -m winnow`` / ``winnow`` - run the development server."""

from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(prog="winnow", description="Run the Winnow API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    # Pass the import string, not the app object: uvicorn's reloader needs to
    # re-import the module in the child process, and an object cannot be.
    uvicorn.run("winnow.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
