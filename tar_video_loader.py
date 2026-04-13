from __future__ import annotations

import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


@dataclass(frozen=True)
class TarVideoMember:
    """Metadata for a video entry inside a tar shard."""

    name: str
    stem: str
    safe_stem: str


class TarShardReader:
    """Reader for one tar shard that can list and extract video members."""

    def __init__(self, tar_path: Path, video_suffixes: set[str]):
        self.tar_path = tar_path
        self.video_suffixes = video_suffixes
        self._archive: tarfile.TarFile | None = None

    def __enter__(self) -> TarShardReader:
        self._archive = tarfile.open(self.tar_path, "r")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._archive is not None:
            self._archive.close()
            self._archive = None

    def video_members(self) -> list[TarVideoMember]:
        if self._archive is None:
            raise RuntimeError("Tar shard is not open")
        members: list[TarVideoMember] = []
        for member in self._archive.getmembers():
            if not member.isfile():
                continue
            if Path(member.name).suffix.lower() not in self.video_suffixes:
                continue
            stem = Path(member.name).stem
            members.append(
                TarVideoMember(
                    name=member.name,
                    stem=stem,
                    safe_stem=stem.replace("/", "__"),
                )
            )
        return members

    def extract_video(self, member: TarVideoMember, work_dir: Path) -> Path:
        if self._archive is None:
            raise RuntimeError("Tar shard is not open")
        work_dir.mkdir(parents=True, exist_ok=True)
        output_path = work_dir / Path(member.name).name
        extracted = self._archive.extractfile(member.name)
        if extracted is None:
            raise RuntimeError(f"Failed to extract member stream: {member.name}")
        with output_path.open("wb") as handle:
            shutil.copyfileobj(extracted, handle)
        return output_path


class TarVideoLoader:
    """Loader that scans tar shards and exposes per-shard readers."""

    def __init__(self, data_root: Path, work_root: Path, video_suffixes: set[str] | None = None):
        self.data_root = data_root
        self.work_root = work_root
        self.video_suffixes = video_suffixes or VIDEO_SUFFIXES

    def list_tar_paths(self) -> list[Path]:
        return sorted(self.data_root.rglob("*.tar"))

    def open_shard(self, tar_path: Path) -> TarShardReader:
        return TarShardReader(tar_path, self.video_suffixes)
