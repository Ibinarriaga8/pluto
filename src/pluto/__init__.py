# __init__.py
"""
pluto – Offline Deep Reinforcement Learning for RAG optimisation.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pluto")
except PackageNotFoundError:
    __version__ = "0.1.0"
