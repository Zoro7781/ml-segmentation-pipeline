#!/usr/bin/env bash
set -e

exec uvicorn segserve.api.app:app --host 0.0.0.0 --port 8000
