"""
The check run_paper.py makes before any step: the programs' environment is the one
uv.lock describes.  Run as `uv run --locked python -m jetcorrml.environment` in the
tree; it prints

    IMPORTED <file of this module>     (the tree's code is the code that runs)
    PYTHON <version>
    PACKAGE <name> <version installed> <version in uv.lock> ok|MISMATCH

A package matches when its version is one uv.lock gives it, without the build's
local tag (torch 2.10.0+cpu and 2.10.0 are the same torch).  The runner stops if
the Python is not the one of .python-version or any package does not match.
"""

import re
import sys
import tomllib
from importlib import metadata
from pathlib import Path

TREE = Path(__file__).resolve().parents[2]


def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def public(version):
    return version.split("+", 1)[0]


def main():
    print("IMPORTED", Path(__file__).resolve(), flush=True)
    print("PYTHON", ".".join(map(str, sys.version_info[:3])))
    lock = tomllib.loads((TREE / "uv.lock").read_text())
    pins = {}
    for p in lock["package"]:
        if p.get("source", {}).get("editable") or p.get("source", {}).get("virtual"):
            continue                                   # this project itself
        pins.setdefault(norm(p["name"]), set()).add(public(p["version"]))
    installed = {}
    for d in metadata.distributions():
        installed.setdefault(norm(d.metadata["Name"]), public(d.version))
    for name in sorted(pins):
        have = installed.get(name)
        if have is None:
            continue                                   # not used on this platform
        want = " or ".join(sorted(pins[name]))
        print("PACKAGE", name, have, want, "ok" if have in pins[name] else "MISMATCH")


if __name__ == "__main__":
    main()
