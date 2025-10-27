"""Utility functions for embedding_tools."""

from .device_detection import detect_best_backend, detect_best_device, get_device_info

__all__ = [
    'detect_best_backend',
    'detect_best_device',
    'get_device_info',
]
