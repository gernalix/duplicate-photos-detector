"""Linux file access for private local indexes and archive images."""
from __future__ import annotations

import os
import stat
import uuid
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def directory_fd(path: Path, *, private: bool = False):
    path = path.absolute()
    if ".." in path.parts:
        raise ValueError("parent traversal is not allowed")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open("/", flags)
    try:
        for depth, part in enumerate(path.parts[1:]):
            if private:
                info = os.fstat(fd)
                if ((depth and info.st_uid not in (0, os.geteuid()))
                        or (info.st_mode & 0o022 and not info.st_mode & stat.S_ISVTX)):
                    raise PermissionError("untrusted ancestor of output directory")
                try:
                    os.mkdir(part, mode=0o700, dir_fd=fd)
                except FileExistsError:
                    pass
            child = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        yield fd
    finally:
        os.close(fd)


@contextmanager
def archive_image(root: Path, path: Path):
    if not root.is_absolute() or not path.is_absolute():
        raise ValueError("archive paths must be absolute")
    path.relative_to(root)
    if path == root or ".." in path.parts:
        raise ValueError("invalid archive image path")
    with directory_fd(path.parent) as parent:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                     dir_fd=parent)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("archive image is not a regular file")
            yield Path(f"/proc/self/fd/{fd}"), info
        finally:
            os.close(fd)


@contextmanager
def private_directory(path: Path):
    with directory_fd(path, private=True) as fd:
        info = os.fstat(fd)
        if info.st_uid != os.geteuid() or info.st_mode & 0o022:
            raise PermissionError("output directory must be owned by the user and protected against other writers")
        yield fd


def write_private_file(path: Path, data: bytes) -> None:
    with private_directory(path.parent) as parent:
        temporary = ".diagnostic-" + uuid.uuid4().hex
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
            os.replace(temporary, path.name, src_dir_fd=parent, dst_dir_fd=parent)
        finally:
            try:
                os.unlink(temporary, dir_fd=parent)
            except FileNotFoundError:
                pass
