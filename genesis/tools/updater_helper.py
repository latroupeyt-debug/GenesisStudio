"""Updater minimal et indépendant de Genesis/PyTorch, exécuté après fermeture de l'app."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
import zipfile
from pathlib import Path, PurePosixPath

MAX_FILE_BYTES = 4 * 1024 ** 3
MAX_PATCH_TOTAL_BYTES = 8 * 1024 ** 3


class PatchError(RuntimeError):
    pass


def safe_relative(name: str) -> Path:
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise PatchError("Chemin de fichier invalide dans le patch.")
    posix = PurePosixPath(name)
    if posix.is_absolute() or any(part in ("", ".", "..") for part in posix.parts) or ":" in name:
        raise PatchError(f"Chemin refusé dans le patch : {name!r}")
    return Path(*posix.parts)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _wait_for_pid(pid: int, timeout_seconds: int = 300) -> None:
    if os.name != "nt":
        # Utilisé seulement pour tests/dev; sur Windows on attend réellement l'app.
        return
    import ctypes
    from ctypes import wintypes

    SYNCHRONIZE = 0x00100000
    WAIT_OBJECT_0 = 0
    WAIT_TIMEOUT = 0x00000102
    kernel32 = ctypes.windll.kernel32
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
    if not handle:
        return  # le processus est probablement déjà fermé
    try:
        result = kernel32.WaitForSingleObject(handle, timeout_seconds * 1000)
        if result == WAIT_TIMEOUT:
            raise PatchError("L'application est toujours ouverte après 5 minutes ; mise à jour annulée.")
        if result != WAIT_OBJECT_0:
            raise PatchError("Impossible d'attendre la fermeture de l'application.")
    finally:
        kernel32.CloseHandle(handle)


def apply_patches(install_dir: Path, data_dir: Path, patch_paths: list[Path],
                  from_version: str = "", progress=None) -> str:
    """Applique une chaîne de patches ; annule tout le lot si un seul échoue."""
    install_dir = Path(install_dir).resolve()
    data_dir = Path(data_dir).resolve()
    backup_dir = data_dir / "updates" / "backups" / time.strftime("%Y%m%d-%H%M%S-") / uuid.uuid4().hex
    backup_dir.mkdir(parents=True, exist_ok=True)
    originals: dict[str, Path | None] = {}
    touched: list[str] = []
    latest = ""
    total_files = 0
    expected_from = from_version.lstrip("vV") or None
    for patch_path in patch_paths:
        with zipfile.ZipFile(patch_path, "r") as archive:
            try:
                manifest = json.loads(archive.read("patch.json").decode("utf-8"))
            except Exception as exc:
                raise PatchError(f"Manifest du patch illisible : {exc}") from None
            if manifest.get("format") != 1 or not manifest.get("to_version"):
                raise PatchError("Version de format du patch inconnue.")
            patch_from = str(manifest.get("from_version", "")).lstrip("vV")
            if expected_from is not None and patch_from != expected_from:
                raise PatchError(f"Chaîne de patches incorrecte : {patch_from} au lieu de {expected_from}.")
            expected_from = str(manifest["to_version"]).lstrip("vV")
            files = manifest.get("files")
            deleted = manifest.get("delete", [])
            if not isinstance(files, list) or not isinstance(deleted, list):
                raise PatchError("Manifest du patch mal formé.")
            total_files += len(files) + len(deleted)
    done = 0
    try:
        for patch_index, patch_path in enumerate(patch_paths, 1):
            with zipfile.ZipFile(patch_path, "r") as archive:
                manifest = json.loads(archive.read("patch.json").decode("utf-8"))
                latest = str(manifest["to_version"])
                for item in manifest["files"]:
                    rel = safe_relative(item.get("path"))
                    size = int(item.get("size", -1))
                    expected = str(item.get("sha256", ""))
                    if size < 0 or size > MAX_FILE_BYTES:
                        raise PatchError(f"Taille refusée pour {rel}.")
                    member = "files/" + rel.as_posix()
                    info = archive.getinfo(member)
                    if info.file_size != size:
                        raise PatchError(f"Taille incohérente dans le patch pour {rel}.")
                    if info.file_size > MAX_PATCH_TOTAL_BYTES:
                        raise PatchError("Fichier trop volumineux dans un patch.")
                    content = archive.read(member)
                    if len(content) != size or _sha256(content) != expected:
                        raise PatchError(f"Vérification SHA-256 échouée pour {rel}.")
                    target = (install_dir / rel).resolve()
                    if install_dir not in target.parents:
                        raise PatchError(f"Chemin hors du dossier d'installation : {rel}.")
                    key = rel.as_posix()
                    if key not in originals:
                        if target.exists():
                            saved = backup_dir / rel
                            saved.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(target, saved)
                            originals[key] = saved
                        else:
                            originals[key] = None
                        touched.append(key)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temp = target.with_name(target.name + ".genesis-new")
                    with temp.open("wb") as output:
                        output.write(content)
                        output.flush()
                        os.fsync(output.fileno())
                    os.replace(temp, target)
                    done += 1
                    if progress:
                        progress(done / max(1, total_files), f"Installé : {rel}")

                for name in manifest.get("delete", []):
                    rel = safe_relative(name)
                    target = (install_dir / rel).resolve()
                    if install_dir not in target.parents:
                        raise PatchError(f"Chemin hors du dossier d'installation : {rel}.")
                    key = rel.as_posix()
                    if key not in originals:
                        if target.is_file():
                            saved = backup_dir / rel
                            saved.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(target, saved)
                            originals[key] = saved
                        else:
                            originals[key] = None
                        touched.append(key)
                    if target.is_file():
                        target.unlink()
                    done += 1
                    if progress:
                        progress(done / max(1, total_files), f"Retiré : {rel}")
    except Exception as exc:
        rollback_errors = []
        for key in reversed(touched):
            target = install_dir / safe_relative(key)
            saved = originals[key]
            try:
                if saved is None:
                    target.unlink(missing_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temp = target.with_name(target.name + ".genesis-rollback")
                    shutil.copy2(saved, temp)
                    os.replace(temp, target)
            except Exception as rollback_exc:
                rollback_errors.append(f"{key}: {rollback_exc}")
        detail = (" Rollback incomplet : " + "; ".join(rollback_errors)) if rollback_errors else ""
        raise PatchError(f"Mise à jour annulée et fichiers précédents restaurés : {exc}.{detail}") from None
    if progress:
        progress(1.0, f"Mise à jour vers {latest} terminée.")
    return latest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--install-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--from-version", required=True)
    parser.add_argument("--relaunch", type=Path, required=True)
    parser.add_argument("--patch", type=Path, action="append", required=True)
    args = parser.parse_args()
    try:
        _wait_for_pid(args.wait_pid)
        latest = apply_patches(args.install_dir, args.data_dir, args.patch,
                               from_version=args.from_version)
        if not args.relaunch.is_file():
            raise PatchError("Exécutable Genesis Studio introuvable après la mise à jour.")
        subprocess.Popen([str(args.relaunch)], cwd=str(args.install_dir), close_fds=True)
        return 0
    except Exception as exc:
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, str(exc), "Mise à jour Genesis Studio", 0x10)
        except Exception:
            print(str(exc), file=sys.stderr)
        # Après le rollback, relance la version précédente plutôt que laisser l'utilisateur
        # sur un bureau sans application.
        try:
            if args.relaunch.is_file():
                subprocess.Popen([str(args.relaunch)], cwd=str(args.install_dir), close_fds=True)
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
