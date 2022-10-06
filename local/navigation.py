from pathlib import Path


def is_repo_root(path: Path) -> bool:
    """Returns `True` if `path` is a directory with a `.git` entry; `False` otherwise."""

    if not path.is_dir():
        return False

    for entry in path.iterdir():
        if entry.stem == ".git":
            return True

    return False


def get_repo_root(path: Path) -> Path:
    """Goes up the filesystem until it finds a repo root. Raises an exception if none is found."""

    current_path = path.resolve()
    filesystem_root = path.root

    while (current_path != filesystem_root):
        if is_repo_root(current_path):
            return current_path
        current_path = current_path.parent

    raise Exception(f"No repo root found for path: {path}")
