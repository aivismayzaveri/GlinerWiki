"""Jujutsu (jj) version control integration for OpenKB wiki."""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def _jj_bin() -> str | None:
    """Return the path to the jj binary, or None if not installed."""
    return shutil.which("jj")


def is_available() -> bool:
    """Return True if jj is installed and the wiki/ dir has a jj repo."""
    return _jj_bin() is not None


def is_initialized(wiki_dir: Path) -> bool:
    """Return True if wiki_dir has a jj repo (.jj/ directory)."""
    return (wiki_dir / ".jj").is_dir()


def describe(wiki_dir: Path, message: str) -> bool:
    """Run `jj describe` in wiki_dir to label the current snapshot.

    jj auto-snapshots file changes before any command, so calling
    ``describe`` is enough to capture and label the current state.

    Returns True on success, False on failure.
    """
    jj = _jj_bin()
    if jj is None or not is_initialized(wiki_dir):
        return False
    try:
        subprocess.run(
            [jj, "describe", "-m", message],
            cwd=str(wiki_dir), capture_output=True, text=True, check=True,
        )
        return True
    except subprocess.CalledProcessError as exc:
        logger.debug("jj describe failed: %s", exc.stderr.strip())
        return False


def new(wiki_dir: Path) -> bool:
    """Run `jj new` in wiki_dir to start a fresh working copy commit.

    Returns True on success, False on failure.
    """
    jj = _jj_bin()
    if jj is None or not is_initialized(wiki_dir):
        return False
    try:
        subprocess.run(
            [jj, "new"],
            cwd=str(wiki_dir), capture_output=True, text=True, check=True,
        )
        return True
    except subprocess.CalledProcessError as exc:
        logger.debug("jj new failed: %s", exc.stderr.strip())
        return False


def log(wiki_dir: Path, limit: int = 20, file_path: str | None = None) -> str | None:
    """Return the jj log output for the wiki repo, or None on failure."""
    jj = _jj_bin()
    if jj is None or not is_initialized(wiki_dir):
        return None
    tmpl = 'change_id.short() ++ " " ++ committer.timestamp().ago() ++ " " ++ description.first_line() ++ "\n"'
    # Try latest() revset (jj 0.18+), fall back to @~N..@ for older versions
    if file_path:
        revsets = [
            f"latest(files('{file_path}'), {limit})",
            f"files('{file_path}') & @~{limit}..@",
        ]
    else:
        revsets = [
            f"latest(::, {limit})",
            f"@~{limit}..@",
        ]
    for revset in revsets:
        try:
            cmd = [jj, "log", "--no-pager", "-T", tmpl, "-r", revset]
            result = subprocess.run(
                cmd, cwd=str(wiki_dir), capture_output=True, text=True,
            )
            if result.returncode == 0:
                return result.stdout.strip() or "(no history)"
        except Exception as exc:
            logger.debug("jj log failed with revset %r: %s", revset, exc)
    return None


def diff(wiki_dir: Path, revision: str = "@") -> str | None:
    """Return the jj diff output, or None on failure."""
    jj = _jj_bin()
    if jj is None or not is_initialized(wiki_dir):
        return None
    try:
        result = subprocess.run(
            [jj, "diff", "--no-pager", "-r", revision],
            cwd=str(wiki_dir), capture_output=True, text=True,
        )
        return result.stdout.strip() or "(no changes)"
    except Exception as exc:
        logger.debug("jj diff failed: %s", exc)
        return None
