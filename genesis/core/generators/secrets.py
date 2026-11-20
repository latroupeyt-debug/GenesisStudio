"""Stockage facultatif de clés via DPAPI Windows (pas de clé en clair sur disque).

Sur les autres plateformes, la clé reste uniquement en mémoire le temps de la session.
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
from ctypes import wintypes
from pathlib import Path

from .. import paths


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_byte))]


def _blob(data: bytes):
    buffer = ctypes.create_string_buffer(data)
    return _DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_byte))), buffer


def _protect(data: bytes) -> bytes:
    if sys.platform != "win32":
        raise OSError("Le stockage chiffré DPAPI n'est disponible que sous Windows.")
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    blob_ptr = ctypes.POINTER(_DataBlob)
    crypt32.CryptProtectData.argtypes = [blob_ptr, wintypes.LPCWSTR, blob_ptr,
                                         ctypes.c_void_p, ctypes.c_void_p,
                                         wintypes.DWORD, blob_ptr]
    crypt32.CryptProtectData.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    source, keepalive = _blob(data)
    target = _DataBlob()
    ok = crypt32.CryptProtectData(
        ctypes.byref(source), "Genesis Studio API key", None, None, None,
        0x1, ctypes.byref(target),  # CRYPTPROTECT_UI_FORBIDDEN
    )
    if not ok:
        raise ctypes.WinError()
    try:
        return ctypes.string_at(target.pbData, target.cbData)
    finally:
        kernel32.LocalFree(target.pbData)


def _unprotect(data: bytes) -> bytes:
    if sys.platform != "win32":
        raise OSError("Le stockage chiffré DPAPI n'est disponible que sous Windows.")
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    blob_ptr = ctypes.POINTER(_DataBlob)
    crypt32.CryptUnprotectData.argtypes = [blob_ptr, ctypes.c_void_p, blob_ptr,
                                           ctypes.c_void_p, ctypes.c_void_p,
                                           wintypes.DWORD, blob_ptr]
    crypt32.CryptUnprotectData.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    source, keepalive = _blob(data)
    target = _DataBlob()
    ok = crypt32.CryptUnprotectData(
        ctypes.byref(source), None, None, None, None, 0x1, ctypes.byref(target)
    )
    if not ok:
        raise ctypes.WinError()
    try:
        return ctypes.string_at(target.pbData, target.cbData)
    finally:
        kernel32.LocalFree(target.pbData)


def _path() -> Path:
    return paths.home() / "api-keys.dpapi.json"


def load_api_key(provider: str) -> str:
    if sys.platform != "win32":
        return ""
    try:
        store = json.loads(_path().read_text(encoding="utf-8"))
        encrypted = store.get(provider, "")
        if encrypted:
            import base64
            return _unprotect(base64.b64decode(encrypted)).decode("utf-8")
    except Exception:
        return ""
    return ""


def save_api_key(provider: str, value: str) -> None:
    if sys.platform != "win32":
        raise OSError("La mémorisation sécurisée des clés est disponible sous Windows uniquement.")
    import base64

    path = _path()
    store: dict = {}
    try:
        store = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    if value:
        store[provider] = base64.b64encode(_protect(value.encode("utf-8"))).decode("ascii")
    else:
        store.pop(provider, None)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(store, indent=2), encoding="utf-8")
    temp.replace(path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def delete_api_key(provider: str) -> None:
    save_api_key(provider, "")
