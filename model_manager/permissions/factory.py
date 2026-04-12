"""Permission manager factory."""

import sys
from model_manager.permissions.base import PermissionManager


def get_permission_manager() -> PermissionManager:
    if sys.platform == "win32":
        from model_manager.permissions.windows import WindowsPermissionManager
        return WindowsPermissionManager()
    if sys.platform == "darwin":
        from model_manager.permissions.macos import MacOSPermissionManager
        return MacOSPermissionManager()
    from model_manager.permissions.linux import LinuxPermissionManager
    return LinuxPermissionManager()
