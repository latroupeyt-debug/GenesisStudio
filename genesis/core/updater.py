"""Vérification des versions GitHub et téléchargement de patches différentiels.

Les projets utilisateur vivent dans %LOCALAPPDATA% et ne sont jamais inclus dans
les patches. Le patch est appliqué par GenesisUpdater.exe après fermeture de l'app.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

UPDATE_REPOSITORY = "latroupeyt-debug/GenesisStudio"
UPDATE_ASSET_NAME = "GenesisStudio-Windows-x64-update.zip"
FULL_ASSET_NAME = "GenesisStudio-Windows-x64-full.zip"
API_ROOT = "https://api.github.com/repos/" + UPDATE_REPOSITORY
USER_AGENT = "GenesisStudio-Updater/1"
MAX_RELEASES = 100
MAX_PATCH_BYTES = 2 * 1024 * 1024 * 1024


class UpdateError(RuntimeError):
    pass


@dataclass
class ReleasePatch:
    tag: str
    version: str
    url: str
    size: int


@dataclass
class UpdatePlan:
    current_version: str
    latest_version: str
    patches: list[ReleasePatch] = field(default_factory=list)
    available: bool = False
    release_url: str = ""

    @property
    def total_bytes(self) -> int:
        return sum(patch.size for patch in self.patches)

    @property
    def summary(self) -> str:
        if not self.available:
            return f"Genesis Studio {self.current_version} est à jour."
        total = self.total_bytes
        if total >= 1024 ** 3:
            size = f"{total / (1024 ** 3):.2f} Go"
        elif total >= 1024 ** 2:
            size = f"{total / (1024 ** 2):.0f} Mo"
        else:
            size = f"{total / 1024:.0f} Ko"
        return (f"Version {self.latest_version} disponible · {len(self.patches)} patch(s) · "
                f"environ {size} à télécharger")


def _version_tuple(value: str) -> tuple[int, ...]:
    value = value.strip().lstrip("vV")
    value = value.split("-", 1)[0]
    parts = re.findall(r"\d+", value)
    if not parts:
        raise UpdateError(f"Version GitHub invalide : {value!r}")
    return tuple(int(part) for part in parts)


def _asset(release: dict, name: str) -> dict | None:
    for item in release.get("assets", []):
        if item.get("name") == name:
            return item
    return None


def build_update_plan(current_version: str, releases: list[dict],
                      repo: str = UPDATE_REPOSITORY) -> UpdatePlan:
    """Construit la chaîne de patches, oldest-first, pour supporter les versions sautées."""
    stable: list[tuple[tuple[int, ...], dict, str]] = []
    for release in releases:
        if release.get("draft") or release.get("prerelease"):
            continue
        tag = str(release.get("tag_name", ""))
        if not tag:
            continue
        try:
            version = _version_tuple(tag)
        except UpdateError:
            continue
        stable.append((version, release, tag.lstrip("vV")))
    stable.sort(key=lambda item: item[0])

    current_tuple = _version_tuple(current_version)
    current_index = next((i for i, (version, _release, _tag) in enumerate(stable)
                          if version == current_tuple), None)
    if current_index is None:
        raise UpdateError(
            f"La version {current_version} n'a pas de release de référence dans ce dépôt. "
            "Installez la version complète publiée sur GitHub, puis les prochaines mises à jour seront différentielles."
        )
    newer = stable[current_index + 1:]
    if not newer:
        current_release = stable[current_index][1]
        return UpdatePlan(
            current_version=current_version,
            latest_version=stable[-1][2],
            available=False,
            release_url=current_release.get("html_url", f"https://github.com/{repo}/releases"),
        )

    patches: list[ReleasePatch] = []
    for _version_key, release, version in newer:
        asset = _asset(release, UPDATE_ASSET_NAME)
        if not asset or not asset.get("browser_download_url"):
            raise UpdateError(
                f"Le patch de la version {version} est absent. Téléchargez le ZIP complet depuis GitHub."
            )
        size = int(asset.get("size", 0) or 0)
        if size < 0 or size > MAX_PATCH_BYTES:
            raise UpdateError(f"Taille de patch invalide pour {version}.")
        patches.append(ReleasePatch(
            tag=str(release["tag_name"]), version=version,
            url=str(asset["browser_download_url"]), size=size,
        ))
    latest_release = newer[-1][1]
    return UpdatePlan(
        current_version=current_version,
        latest_version=newer[-1][2],
        patches=patches,
        available=True,
        release_url=latest_release.get("html_url", f"https://github.com/{repo}/releases/latest"),
    )


def _github_json(url: str) -> dict | list:
    request = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": USER_AGENT,
    })
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            data = response.read(8 * 1024 * 1024 + 1)
            if len(data) > 8 * 1024 * 1024:
                raise UpdateError("Réponse GitHub trop volumineuse.")
            return json.loads(data.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise UpdateError("Aucune release publique n'a encore été publiée sur GitHub.") from None
        if exc.code == 403:
            raise UpdateError("GitHub limite temporairement les vérifications. Réessayez plus tard.") from None
        raise UpdateError(f"GitHub a répondu HTTP {exc.code}.") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise UpdateError(f"Impossible de joindre GitHub : {getattr(exc, 'reason', str(exc))}") from None
    except json.JSONDecodeError:
        raise UpdateError("GitHub a renvoyé une réponse invalide.") from None


def check_for_updates(current_version: str, repo: str = UPDATE_REPOSITORY,
                      fetch_json: Callable[[str], dict | list] | None = None) -> UpdatePlan:
    fetch = fetch_json or _github_json
    releases = fetch(f"https://api.github.com/repos/{repo}/releases?per_page={MAX_RELEASES}")
    if not isinstance(releases, list):
        raise UpdateError("Réponse de la liste des releases GitHub inattendue.")
    return build_update_plan(current_version, releases, repo)


def download_update_chain(plan: UpdatePlan, data_dir: Path, *,
                          progress: Callable[[float, str], None] | None = None,
                          should_stop: Callable[[], bool] | None = None) -> list[Path]:
    """Télécharge en flux la chaîne de patches dans le dossier de données utilisateur."""
    if not plan.available or not plan.patches:
        return []
    total = sum(max(patch.size, 1) for patch in plan.patches)
    done_before = 0
    staging = Path(data_dir) / "updates" / "staging" / uuid.uuid4().hex
    staging.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for index, patch in enumerate(plan.patches, 1):
        if should_stop and should_stop():
            raise UpdateError("Téléchargement annulé.")
        if not patch.url.startswith("https://github.com/") and not patch.url.startswith("https://objects.githubusercontent.com/"):
            raise UpdateError("URL de patch GitHub non sécurisée/refusée.")
        request = urllib.request.Request(patch.url, headers={"User-Agent": USER_AGENT})
        target = staging / f"patch-{index:03d}-{patch.version}.zip"
        temp = target.with_suffix(".part")
        received = 0
        try:
            with urllib.request.urlopen(request, timeout=60) as response, temp.open("wb") as output:
                while True:
                    if should_stop and should_stop():
                        raise UpdateError("Téléchargement annulé.")
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > MAX_PATCH_BYTES:
                        raise UpdateError(f"Le patch {patch.version} dépasse la taille maximale autorisée.")
                    output.write(chunk)
                    if progress:
                        fraction = min(0.99, (done_before + received) / total)
                        progress(fraction, f"Téléchargement {index}/{len(plan.patches)} · "
                                            f"{received / (1024 ** 2):.1f} Mo · {patch.version}")
            if patch.size and received != patch.size:
                raise UpdateError(f"Taille du patch {patch.version} incorrecte : {received} octets reçus.")
            temp.replace(target)
            downloaded.append(target)
            done_before += max(patch.size, received)
        except UpdateError:
            temp.unlink(missing_ok=True)
            raise
        except Exception as exc:
            temp.unlink(missing_ok=True)
            raise UpdateError(f"Échec du téléchargement {patch.version} : {exc}") from None
    if progress:
        progress(1.0, "Tous les patches sont téléchargés et prêts à être vérifiés.")
    return downloaded


def verify_patch_archives(patches: list[Path], current_version: str | None = None) -> list[dict]:
    """Vérifie intégralement les ZIPs téléchargés avant de fermer l'application."""
    import hashlib
    import zipfile
    from pathlib import PurePosixPath

    verified = []
    total_uncompressed = 0
    expected_from = current_version.lstrip("vV") if current_version else None
    for patch_path in patches:
        try:
            with zipfile.ZipFile(patch_path, "r") as archive:
                manifest = json.loads(archive.read("patch.json").decode("utf-8"))
                if manifest.get("format") != 1 or not manifest.get("to_version"):
                    raise UpdateError("Format de patch inconnu.")
                from_version = str(manifest.get("from_version", "")).lstrip("vV")
                to_version = str(manifest.get("to_version", "")).lstrip("vV")
                if expected_from is not None and from_version != expected_from:
                    raise UpdateError(f"Chaîne de patches inattendue : {from_version} au lieu de {expected_from}.")
                expected_from = to_version
                file_items = manifest.get("files", [])
                deleted = manifest.get("delete", [])
                if not isinstance(file_items, list) or not isinstance(deleted, list):
                    raise UpdateError("Manifest du patch incorrect.")
                for name in deleted:
                    _validate_relpath(name)
                for item in file_items:
                    name = _validate_relpath(item.get("path"))
                    size = int(item.get("size", -1))
                    if size < 0 or size > MAX_PATCH_BYTES:
                        raise UpdateError(f"Taille refusée pour {name}.")
                    member = "files/" + name
                    info = archive.getinfo(member)
                    if info.file_size != size:
                        raise UpdateError(f"Taille différente de celle annoncée pour {name}.")
                    total_uncompressed += size
                    if total_uncompressed > MAX_PATCH_BYTES * 2:
                        raise UpdateError("Patch trop volumineux après décompression.")
                    digest = hashlib.sha256()
                    read = 0
                    with archive.open(member) as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            read += len(chunk)
                            digest.update(chunk)
                    if read != size or digest.hexdigest() != item.get("sha256"):
                        raise UpdateError(f"Vérification SHA-256 échouée pour {name}.")
                verified.append(manifest)
        except UpdateError:
            raise
        except (OSError, zipfile.BadZipFile, KeyError, ValueError, json.JSONDecodeError) as exc:
            raise UpdateError(f"Patch illisible ({Path(patch_path).name}) : {exc}") from None
    return verified


def _validate_relpath(value: object) -> str:
    from pathlib import PurePosixPath

    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise UpdateError("Chemin invalide dans le patch.")
    path = PurePosixPath(value)
    if path.is_absolute() or ":" in value or any(part in ("", ".", "..") for part in path.parts):
        raise UpdateError(f"Chemin dangereux dans le patch : {value!r}")
    return path.as_posix()
