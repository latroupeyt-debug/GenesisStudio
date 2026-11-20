"""Construit un patch ZIP différentiel entre deux bundles PyInstaller.

Usage:
  python tools/build_update_patch.py --current "dist/Genesis Studio" \\
      --previous previous/Genesis-Studio --from v1.1.0 --to v1.1.1 \\
      --out dist/GenesisStudio-Windows-x64-update.zip
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path

EXCLUDES = {".DS_Store", "Thumbs.db", "GenesisUpdater.exe"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files_under(root: Path) -> dict[str, Path]:
    result = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.name in EXCLUDES:
            continue
        rel = path.relative_to(root).as_posix()
        result[rel] = path
    return result


def build_patch(current: Path, previous: Path, from_version: str, to_version: str,
                output: Path) -> dict:
    current, previous, output = current.resolve(), previous.resolve(), output.resolve()
    current_files = files_under(current)
    previous_files = files_under(previous)
    changed = []
    for rel, path in sorted(current_files.items()):
        old = previous_files.get(rel)
        if old is None or path.stat().st_size != old.stat().st_size or sha256(path) != sha256(old):
            changed.append({"path": rel, "size": path.stat().st_size, "sha256": sha256(path)})
    deleted = sorted(set(previous_files) - set(current_files))
    manifest = {
        "format": 1,
        "from_version": from_version.lstrip("vV"),
        "to_version": to_version.lstrip("vV"),
        "files": changed,
        "delete": deleted,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("patch.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        for item in changed:
            archive.write(current_files[item["path"]], "files/" + item["path"])
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Génère un patch incrémental Genesis Studio")
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--from", dest="from_version", required=True)
    parser.add_argument("--to", dest="to_version", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.current.is_dir() or not args.previous.is_dir():
        parser.error("--current et --previous doivent être des dossiers PyInstaller existants")
    manifest = build_patch(args.current, args.previous, args.from_version, args.to_version, args.out)
    print(f"Patch {manifest['from_version']} → {manifest['to_version']} : "
          f"{len(manifest['files'])} fichier(s) modifié(s), {len(manifest['delete'])} supprimé(s), "
          f"{args.out.stat().st_size:,} octets".replace(",", " "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
