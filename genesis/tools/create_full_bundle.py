"""Archive un dossier PyInstaller sous le nom stable attendu par l'updater."""
from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

MAX_GITHUB_ASSET = 2 * 1024 * 1024 * 1024 - 16 * 1024 * 1024


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    if not source.is_dir():
        parser.error(f"Dossier PyInstaller introuvable : {source}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True) as archive:
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            rel = Path(source.name) / path.relative_to(source)
            archive.write(path, rel.as_posix())
            if archive.fp and archive.fp.tell() > MAX_GITHUB_ASSET:
                raise SystemExit(
                    "Le ZIP complet dépasse la limite de 2 Gio d'une release GitHub. "
                    "Publiez le ZIP complet sur un hébergement externe ou produisez une build CPU initiale."
                )
    print(f"Bundle complet : {args.output} ({args.output.stat().st_size:,} octets)".replace(",", " "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
