"""Entrypoint module for Docker deployment."""

from server.app import app

__all__ = ["app"]
