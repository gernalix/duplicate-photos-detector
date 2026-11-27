#!/usr/bin/python3 -I
"""Resolve the pinned CPU wheels and check the embedding stack using synthetic input."""
import argparse
from hashlib import file_digest
from html.parser import HTMLParser
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urljoin, urlsplit
from urllib.request import urlopen

from packaging.requirements import Requirement
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename
from packaging.version import Version


ROOT = Path(__file__).resolve().parents[1]
INDEX = "https://download.pytorch.org/whl/cpu/"
WHEEL_HOSTS = {"download.pytorch.org", "download-r2.pytorch.org"}
CPU_NAMES = {"torch", "torchvision"}
MODEL = Path("hub/models--laion--CLIP-ViT-B-32-laion2B-s34B-b79K/snapshots/"
             "1a25a446712ba5ee05982a381eed697ef9b435cf/open_clip_model.safetensors")
MODEL_SHA256 = "ac4f8c4b88af6d963118cbf40ad93176d092abbedfcb752601ae1866352656e6"


class WheelLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.extend(value for name, value in attrs if name == "href" and value)


def trusted_url(url):
    parsed = urlsplit(url)
    return parsed.scheme == "https" and parsed.netloc in WHEEL_HOSTS and not parsed.query


def cpu_reference(requirement):
    pin, = requirement.specifier
    version = Version(pin.version)
    if pin.operator != "==" or version.local != "cpu":
        raise ValueError("PyTorch packages need an exact CPU version")
    index = INDEX + requirement.name + "/"
    with urlopen(index, timeout=15) as response:
        if not trusted_url(response.url):
            raise ValueError("Unexpected redirect from the PyTorch wheel index")
        payload = response.read(4 * 1024 * 1024 + 1)
    if len(payload) > 4 * 1024 * 1024:
        raise ValueError("PyTorch wheel index exceeds the size limit")
    table = WheelLinks()
    table.feed(payload.decode("utf-8"))
    priorities = {tag: priority for priority, tag in enumerate(sys_tags())}
    matches = []
    for link in table.links:
        url = urljoin(index, link)
        ref = urlsplit(url)
        filename = unquote(ref.path.rsplit("/", 1)[-1])
        if (not trusted_url(url) or not filename.startswith(f"{requirement.name}-{version}-")
                or not filename.endswith(".whl")
                or not re.fullmatch(r"sha256=[0-9a-f]{64}", ref.fragment)):
            continue
        name, release, _, tags = parse_wheel_filename(filename)
        compatible = tags & priorities.keys()
        if name == requirement.name and release == version and compatible:
            matches.append((min(priorities[tag] for tag in compatible), url))
    if not matches:
        raise ValueError(f"No compatible official wheel for {requirement}")
    return f"{requirement.name} @ {min(matches)[1]}"


def verify(packages, cache):
    path = cache / MODEL
    with path.open("rb") as model:
        if file_digest(model, "sha256").hexdigest() != MODEL_SHA256:
            raise ValueError("OpenCLIP model checksum mismatch")
    subprocess.run([
        "/usr/bin/bwrap", "--die-with-parent", "--unshare-net", "--unshare-pid",
        "--ro-bind", "/usr", "/usr", "--ro-bind", "/etc/ld.so.cache", "/etc/ld.so.cache",
        "--symlink", "usr/lib64", "/lib64", "--symlink", "usr/lib", "/lib",
        "--symlink", "usr/bin", "/bin", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
        "--ro-bind", str(ROOT / "src"), "/tmp/source", "--ro-bind", str(packages), "/tmp/packages",
        "--ro-bind", str(cache), "/tmp/model-cache",
        "--ro-bind", str(ROOT / "tools/check_embedding_stack.py"), "/tmp/check.py",
        "--clearenv", "--setenv", "HOME", "/tmp", "--setenv", "PATH", "/usr/bin:/bin",
        "--setenv", "PYTHONPATH", "/tmp/source:/tmp/packages", "--setenv", "HF_HOME", "/tmp/model-cache",
        "--setenv", "HF_HUB_OFFLINE", "1", "--setenv", "OMP_NUM_THREADS", "2",
        "--setenv", "OPENBLAS_NUM_THREADS", "1", "--chdir", "/tmp",
        "/usr/bin/python3", "-S", "-P", "-B", "/tmp/check.py",
    ], check=True, timeout=120)


def stage(candidate, pins, active):
    if candidate.exists() or candidate.parent == active.parent:
        raise ValueError("Expected a separate staging directory")
    requirements = [Requirement(line) for line in pins.read_text(encoding="utf-8").splitlines() if line]
    resolved = [cpu_reference(req) if req.name in CPU_NAMES else str(req) for req in requirements]
    lock = pins.with_name("embedding-resolution.txt")
    lock.write_text("\n".join(resolved) + "\n", encoding="utf-8")
    subprocess.run([
        "/usr/bin/uv", "--no-config", "--no-cache", "pip", "install", "--python", "/usr/bin/python3",
        "--index-url", "https://pypi.org/simple", "--only-binary", ":all:",
        "--target", str(candidate), "-r", str(lock),
    ], check=True, timeout=180)
    if CPU_NAMES & {requirement.name for requirement in requirements}:
        verify(candidate, active.parent / "model-cache")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    probe = commands.add_parser("verify")
    probe.add_argument("packages", type=Path)
    probe.add_argument("cache", type=Path)
    update = commands.add_parser("stage")
    for name in ("candidate", "pins", "active"):
        update.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        verify(args.packages.resolve(), args.cache.resolve())
    elif args.command == "stage":
        stage(args.candidate.resolve(), args.pins.resolve(), args.active.resolve())
    else:
        parser.error("choose verify or stage")


if __name__ == "__main__":
    main()
