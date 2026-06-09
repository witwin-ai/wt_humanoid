"""SMPL/AMASS motion source loading for wt_humanoid.

The editor stores a source path and frame metadata. Large pose tensors stay in
the source file or runtime caches rather than being expanded into scene data.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np


_POSE_KEYS = ("poses", "pose", "smpl_pose", "body_pose")
_BETAS_KEYS = ("betas", "shape", "shapes")
_TRANS_KEYS = ("trans", "transl", "translation", "root_translation")
_FPS_KEYS = ("fps", "mocap_framerate", "frame_rate", "framerate")


@dataclass(frozen=True)
class SMPLMotionData:
    """Normalized SMPL motion data.

    ``poses`` is always shaped ``(frames, pose_size)``. ``betas`` is global shape
    data. ``translations`` is always ``(frames, 3)`` and defaults to zeros.
    """

    poses: np.ndarray
    betas: np.ndarray
    translations: np.ndarray
    fps: float
    source_path: str

    @property
    def frame_count(self) -> int:
        return int(self.poses.shape[0])

    @property
    def pose_size(self) -> int:
        return int(self.poses.shape[1])

    def frame(self, index: int, *, loop: bool = False) -> dict:
        """Return a JSON-serializable frame payload."""
        if self.frame_count <= 0:
            raise ValueError("SMPL motion contains no frames.")
        frame_index = int(index)
        if loop:
            frame_index %= self.frame_count
        else:
            frame_index = max(0, min(frame_index, self.frame_count - 1))
        return {
            "frame_index": frame_index,
            "time": float(frame_index / self.fps),
            "pose": self.poses[frame_index].astype(np.float32).tolist(),
            "shape": self.betas.astype(np.float32).tolist(),
            "translation": self.translations[frame_index].astype(np.float32).tolist(),
            "source_path": self.source_path,
        }


def load_smpl_motion(
    path: str | Path,
    *,
    pose_key: Optional[str] = None,
    betas_key: Optional[str] = None,
    trans_key: Optional[str] = None,
    fps: Optional[float] = None,
) -> SMPLMotionData:
    """Load an AMASS/SMPL-style ``.npz`` file into normalized frame data."""
    source_path = str(path)
    with np.load(source_path, allow_pickle=False) as npz:
        data = {key: npz[key] for key in npz.files}

    poses = _normalize_poses(_require_array(data, pose_key, _POSE_KEYS, "pose"))
    frame_count = int(poses.shape[0])
    betas = _normalize_betas(_optional_array(data, betas_key, _BETAS_KEYS))
    translations = _normalize_translations(
        _optional_array(data, trans_key, _TRANS_KEYS),
        frame_count,
    )
    motion_fps = _resolve_fps(data, fps)
    return SMPLMotionData(
        poses=poses,
        betas=betas,
        translations=translations,
        fps=motion_fps,
        source_path=source_path,
    )


def _require_array(
    data: Mapping[str, np.ndarray],
    requested_key: Optional[str],
    candidate_keys: tuple[str, ...],
    label: str,
) -> np.ndarray:
    value = _optional_array(data, requested_key, candidate_keys)
    if value is None:
        keys = ", ".join(candidate_keys)
        raise ValueError(f"SMPL motion file is missing {label} data; expected one of: {keys}.")
    return value


def _optional_array(
    data: Mapping[str, np.ndarray],
    requested_key: Optional[str],
    candidate_keys: tuple[str, ...],
) -> Optional[np.ndarray]:
    if requested_key:
        if requested_key not in data:
            raise ValueError(f"SMPL motion file has no key {requested_key!r}.")
        return data[requested_key]
    for key in candidate_keys:
        if key in data:
            return data[key]
    return None


def _normalize_poses(value: np.ndarray) -> np.ndarray:
    poses = np.asarray(value, dtype=np.float32)
    if poses.ndim == 1:
        poses = poses.reshape(1, -1)
    elif poses.ndim == 3 and poses.shape[-1] == 3:
        poses = poses.reshape(poses.shape[0], -1)
    elif poses.ndim != 2:
        raise ValueError(
            "SMPL pose data must have shape (frames, pose_size), (frames, joints, 3), or (pose_size,)."
        )
    if poses.shape[0] <= 0 or poses.shape[1] <= 0:
        raise ValueError("SMPL pose data must contain at least one frame and one pose value.")
    return np.ascontiguousarray(poses, dtype=np.float32)


def _normalize_betas(value: Optional[np.ndarray]) -> np.ndarray:
    if value is None:
        return np.zeros(10, dtype=np.float32)
    betas = np.asarray(value, dtype=np.float32)
    if betas.ndim == 0:
        raise ValueError("SMPL betas/shape data must be a vector or a frame-major array.")
    if betas.ndim > 1:
        betas = betas.reshape(betas.shape[0], -1)[0]
    return np.ascontiguousarray(betas.reshape(-1), dtype=np.float32)


def _normalize_translations(value: Optional[np.ndarray], frame_count: int) -> np.ndarray:
    if value is None:
        return np.zeros((frame_count, 3), dtype=np.float32)
    trans = np.asarray(value, dtype=np.float32)
    if trans.ndim == 1:
        if trans.size != 3:
            raise ValueError("SMPL translation vector must contain exactly 3 values.")
        trans = np.repeat(trans.reshape(1, 3), frame_count, axis=0)
    elif trans.ndim == 2 and trans.shape[1] == 3:
        if trans.shape[0] == 1 and frame_count > 1:
            trans = np.repeat(trans, frame_count, axis=0)
        elif trans.shape[0] != frame_count:
            raise ValueError(
                f"SMPL translation frame count {trans.shape[0]} does not match poses {frame_count}."
            )
    else:
        raise ValueError("SMPL translation data must have shape (3,) or (frames, 3).")
    return np.ascontiguousarray(trans, dtype=np.float32)


def _resolve_fps(data: Mapping[str, np.ndarray], explicit_fps: Optional[float]) -> float:
    if explicit_fps is not None:
        fps = float(explicit_fps)
    else:
        fps = 30.0
        for key in _FPS_KEYS:
            if key in data:
                fps = float(np.asarray(data[key]).reshape(-1)[0])
                break
    if fps <= 0.0:
        raise ValueError("SMPL motion fps must be positive.")
    return fps
