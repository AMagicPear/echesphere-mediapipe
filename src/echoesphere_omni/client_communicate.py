"""Backward-compatibility shim.

``TcpClient`` has been moved to ``echoesphere_omni.net.client``.
"""

from echoesphere_omni.net.client import TcpClient

__all__ = ["TcpClient"]
