"""Tests du plan de mise à jour et de l'application atomique des patches."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from genesis.core.updater import UpdateError, build_update_plan, verify_patch_archives  # noqa: E402
from build_update_patch import build_patch  # noqa: E402
from updater_helper import PatchError, apply_patches, safe_relative  # noqa: E402


def release(tag, version, size=123, include=True):
    assets = ([{"name": "GenesisStudio-Windows-x64-update.zip",
               "browser_download_url": f"https://github.com/latroupeyt-debug/GenesisStudio/releases/download/{tag}/patch.zip",
               "size": size}] if include else [])
    return {"tag_name": tag, "draft": False, "prerelease": "-" in tag,
            "html_url": "https://github.com/latroupeyt-debug/GenesisStudio/releases/tag/" + tag,
            "assets": assets}


def main() -> int:
    releases = [release("v1.2.0", "1.2.0"), release("v1.1.0", "1.1.0", include=False),
                release("v1.1.1", "1.1.1"), release("v2.0.0-beta", "2", include=False)]
    plan = build_update_plan("1.1.0", releases)
    assert plan.available and plan.latest_version == "1.2.0"
    assert [x.version for x in plan.patches] == ["1.1.1", "1.2.0"]
    assert "2 patch(s)" in plan.summary
    try:
        build_update_plan("1.0.0", releases)
        raise AssertionError("une version sans release doit demander une installation de base")
    except UpdateError as exc:
        assert "version complète" in str(exc)
    for unsafe in ("../outside", "/absolute", "C:/Windows", "a\\b"):
        try:
            safe_relative(unsafe)
            raise AssertionError(f"chemin dangereux accepté: {unsafe}")
        except PatchError:
            pass

    with tempfile.TemporaryDirectory(prefix="genesis-update-test-") as tmp:
        base = Path(tmp)
        previous = base / "previous"
        current = base / "current"
        install = base / "install"
        data = base / "user-data"
        for root in (previous, current, install):
            (root / "_internal").mkdir(parents=True)
        (previous / "Genesis Studio.exe").write_bytes(b"old executable")
        (previous / "_internal" / "old.dll").write_bytes(b"remove me")
        (current / "Genesis Studio.exe").write_bytes(b"new executable with changes")
        (current / "_internal" / "new.dll").write_bytes(b"new library")
        (install / "Genesis Studio.exe").write_bytes(b"old executable")
        (install / "_internal" / "old.dll").write_bytes(b"remove me")
        patch = base / "patch.zip"
        manifest = build_patch(current, previous, "v1.1.0", "v1.1.1", patch)
        assert len(manifest["files"]) == 2 and manifest["delete"] == ["_internal/old.dll"]
        assert verify_patch_archives([patch], "1.1.0")[0]["to_version"] == "1.1.1"
        version = apply_patches(install, data, [patch], from_version="1.1.0")
        assert version == "1.1.1"
        assert (install / "Genesis Studio.exe").read_bytes() == b"new executable with changes"
        assert (install / "_internal" / "new.dll").read_bytes() == b"new library"
        assert not (install / "_internal" / "old.dll").exists()
        assert list((data / "updates" / "backups").rglob("old.dll"))

        # Un patch au hash altéré ne laisse pas l'installation partiellement modifiée.
        broken = base / "broken.zip"
        with zipfile.ZipFile(broken, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("patch.json", json.dumps({"format": 1, "from_version": "1.1.1",
                                               "to_version": "1.1.2", "files": [
                                                   {"path": "Genesis Studio.exe", "size": 10,
                                                    "sha256": hashlib.sha256(b"temporary").hexdigest()},
                                                   {"path": "_internal/new.dll", "size": 3,
                                                    "sha256": "0" * 64}], "delete": []}))
            z.writestr("files/Genesis Studio.exe", b"temporary")
            z.writestr("files/_internal/new.dll", b"bad")
        try:
            apply_patches(install, data, [broken], from_version="1.1.1")
            raise AssertionError("patch SHA incorrect accepté")
        except PatchError as exc:
            assert "restaurés" in str(exc)
        assert (install / "Genesis Studio.exe").read_bytes() == b"new executable with changes"

    print("Plan GitHub, patch différentiel, suppression, hash et rollback : OK")
    print("TOUS LES TESTS DE MISE À JOUR SONT PASSÉS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
