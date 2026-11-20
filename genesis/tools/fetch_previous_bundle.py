"""Télécharge le bundle complet de la release précédente pour produire le patch suivant.

Code retour 3 : aucun bundle précédent (cas première publication), pas une erreur.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

REPO = "latroupeyt-debug/GenesisStudio"
ASSET = "GenesisStudio-Windows-x64-full.zip"


def ver(tag: str) -> tuple[int, ...]:
    pieces = re.findall(r"\d+", tag.lstrip("vV").split("-", 1)[0])
    return tuple(map(int, pieces)) if pieces else (0,)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--current-tag", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag-file", type=Path, required=True)
    args = parser.parse_args()
    req = urllib.request.Request(
        f"https://api.github.com/repos/{args.repo}/releases?per_page=100",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "GenesisStudio-ReleaseBuilder"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            releases = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise SystemExit(f"Impossible de consulter les releases précédentes : {exc}")
    candidates = []
    for release in releases if isinstance(releases, list) else []:
        if release.get("draft") or release.get("prerelease"):
            continue
        tag = str(release.get("tag_name", ""))
        if ver(tag) >= ver(args.current_tag):
            continue
        asset = next((a for a in release.get("assets", []) if a.get("name") == ASSET), None)
        if asset and asset.get("browser_download_url"):
            candidates.append((ver(tag), tag, asset["browser_download_url"]))
    if not candidates:
        print("Aucune release antérieure avec bundle complet : première release.")
        return 3
    _v, tag, url = max(candidates, key=lambda x: x[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    archive_path = args.output.with_suffix(".zip")
    request = urllib.request.Request(url, headers={"User-Agent": "GenesisStudio-ReleaseBuilder"})
    try:
        with urllib.request.urlopen(request, timeout=180) as response, archive_path.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
    except Exception as exc:
        raise SystemExit(f"Téléchargement du bundle précédent échoué : {exc}")
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                relative = PurePosixPath(member.filename)
                if relative.is_absolute() or ".." in relative.parts:
                    raise SystemExit("Chemin dangereux dans le bundle précédent.")
                target = args.output.joinpath(*relative.parts)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
    finally:
        archive_path.unlink(missing_ok=True)
    args.tag_file.parent.mkdir(parents=True, exist_ok=True)
    args.tag_file.write_text(tag, encoding="utf-8")
    print(f"Bundle antérieur extrait : {tag} → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
