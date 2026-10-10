"""SMPL SkinnedMesh construction utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from witwin.genesis.smpl_pickle import load_smpl_model_data
from witwin_server.core import SceneObject
from witwin_server.core.components import SkinnedMeshComponent


@dataclass(frozen=True)
class SMPLSkinAsset:
    """Bind-pose SMPL mesh and skinning data ready for SkinnedMeshComponent."""

    vertices: np.ndarray
    faces: np.ndarray
    skin_indices: np.ndarray
    skin_weights: np.ndarray
    joint_positions: np.ndarray
    parent_indices: list[int]
    inverse_bind_matrices: np.ndarray


def build_smpl_skin_asset(
    model_root: str | Path,
    *,
    gender: str = "neutral",
    betas: Sequence[float] | None = None,
) -> SMPLSkinAsset:
    """Load an SMPL model and return bind-pose mesh data for GPU skinning."""
    try:
        import torch
        import smplx
    except Exception as exc:  # noqa: BLE001 - callers surface this as tool status.
        raise RuntimeError("SMPL SkinnedMesh requires torch and smplx to load model files.") from exc

    if not str(model_root).strip():
        raise ValueError("model_root is required for SMPL SkinnedMesh construction.")
    root = Path(model_root).expanduser()
    gender_value = str(gender or "neutral").lower()

    model = smplx.create(
        str(root),
        model_type="smpl",
        gender=gender_value,
        ext="pkl",
        batch_size=1,
        data_struct=_load_smpl_data_struct(root, gender_value, smplx),
    )
    betas_tensor = torch.tensor([_normalize_betas(betas)], dtype=torch.float32)
    zero_orient = torch.zeros((1, 3), dtype=torch.float32)
    zero_body = torch.zeros((1, 69), dtype=torch.float32)
    output = model(global_orient=zero_orient, body_pose=zero_body, betas=betas_tensor)

    vertices = output.vertices[0].detach().cpu().numpy().astype(np.float32)
    joints = output.joints[0, :24].detach().cpu().numpy().astype(np.float32)
    faces = np.asarray(model.faces, dtype=np.uint32).reshape(-1, 3)
    weights = _as_numpy(model.lbs_weights)[:, : joints.shape[0]]
    skin_indices, skin_weights = _top_four_skinning(weights)
    parent_indices = _extract_parent_indices(model, joints.shape[0])
    inverse_bind_matrices = _inverse_bind_matrices(joints, parent_indices)

    return SMPLSkinAsset(
        vertices=vertices,
        faces=faces,
        skin_indices=skin_indices,
        skin_weights=skin_weights,
        joint_positions=joints,
        parent_indices=parent_indices,
        inverse_bind_matrices=inverse_bind_matrices,
    )


def _load_smpl_data_struct(root: Path, gender: str, smplx: Any) -> Any:
    model_path = root / "smpl" / f"SMPL_{gender.upper()}.pkl"
    return smplx.utils.Struct(**load_smpl_model_data(model_path))


def attach_smpl_skinned_mesh(
    scene: Any,
    body_obj: SceneObject,
    body: Any,
    *,
    model_root: str | Path | None = None,
    gender: str | None = None,
    shape: Sequence[float] | None = None,
    asset: SMPLSkinAsset | None = None,
) -> dict[str, Any]:
    """Attach a SkinnedMesh component and SMPL bone objects to ``body_obj``."""
    if scene is None:
        raise ValueError("scene is required")
    if not body_obj.id:
        raise ValueError("body_obj must be added to the scene before attaching skinning data")

    skin_asset = asset or build_smpl_skin_asset(
        model_root or getattr(body, "model_root", ""),
        gender=gender or getattr(body, "gender", "neutral"),
        betas=shape if shape is not None else getattr(body, "shape", []),
    )
    bone_ids = _ensure_bones(scene, body_obj, skin_asset)

    skinned = body_obj.get_component("SkinnedMesh")
    if skinned is None:
        skinned = SkinnedMeshComponent()
        body_obj.add_component(skinned)
    skinned.set_mesh_data(skin_asset.vertices, skin_asset.faces, notify=False)
    skinned.set_skinning_data(
        bone_ids,
        skin_asset.skin_indices,
        skin_asset.skin_weights,
        skin_asset.inverse_bind_matrices,
        root_bone_id=bone_ids[0] if bone_ids else None,
    )
    skinned.show_bones = False
    skinned.set_generated_mesh_source_runtime(f"{body_obj.id}:smpl")
    skinned.start()

    return {
        "ok": True,
        "mode": "skinned_mesh",
        "object_id": body_obj.id,
        "bone_ids": bone_ids,
        "root_bone_id": bone_ids[0] if bone_ids else None,
        "bone_count": len(bone_ids),
    }


def _ensure_bones(scene: Any, body_obj: SceneObject, asset: SMPLSkinAsset) -> list[str]:
    bone_ids: list[str] = []
    scene.begin_batch()
    try:
        for index, joint in enumerate(np.asarray(asset.joint_positions, dtype=np.float32)):
            bone_id = f"{body_obj.id}-smpl-bone-{index:02d}"
            bone_ids.append(bone_id)
            bone_obj = scene.get_object(bone_id)
            if bone_obj is None:
                bone_obj = SceneObject(id=bone_id, name=f"{body_obj.name} Bone {index:02d}", mesh_type="Empty")
                scene.add_object(bone_obj)
            transform = bone_obj.get_component("Transform")
            parent_index = asset.parent_indices[index] if index < len(asset.parent_indices) else -1
            if parent_index >= 0 and parent_index < len(asset.joint_positions):
                local_position = joint - asset.joint_positions[parent_index]
            else:
                local_position = joint
            transform.position = np.asarray(local_position, dtype=np.float32).tolist()
            transform.rotation = [0.0, 0.0, 0.0]
            transform.scale = [1.0, 1.0, 1.0]
            transform.freeze_position = True
    finally:
        scene.end_batch()

    for index, bone_id in enumerate(bone_ids):
        parent_index = asset.parent_indices[index] if index < len(asset.parent_indices) else -1
        parent_id = bone_ids[parent_index] if 0 <= parent_index < len(bone_ids) else body_obj.id
        scene.set_parent(bone_id, parent_id, keep_local_transform=True)

    return bone_ids


def _normalize_betas(betas: Sequence[float] | None) -> list[float]:
    values = [float(item) for item in (betas or [])][:10]
    return values + [0.0] * (10 - len(values))


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().astype(np.float32)
    return np.asarray(value, dtype=np.float32)


def _top_four_skinning(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if weights.ndim != 2:
        raise ValueError(f"SMPL lbs_weights must be 2D, got shape {weights.shape}")
    top_count = min(4, weights.shape[1])
    order = np.argsort(weights, axis=1)[:, -top_count:][:, ::-1]
    skin_indices = np.zeros((weights.shape[0], 4), dtype=np.int64)
    skin_weights = np.zeros((weights.shape[0], 4), dtype=np.float32)
    skin_indices[:, :top_count] = order
    skin_weights[:, :top_count] = np.take_along_axis(weights, order, axis=1)
    sums = np.maximum(skin_weights.sum(axis=1, keepdims=True), 1e-8)
    skin_weights = (skin_weights / sums).astype(np.float32)
    return skin_indices, skin_weights


def _extract_parent_indices(model: Any, joint_count: int) -> list[int]:
    parents = _as_numpy(getattr(model, "parents")).astype(np.int64).reshape(-1).tolist()
    result: list[int] = []
    for index in range(joint_count):
        parent = int(parents[index]) if index < len(parents) else -1
        result.append(parent if 0 <= parent < joint_count and parent != index else -1)
    if result:
        result[0] = -1
    return result


def _inverse_bind_matrices(joints: np.ndarray, parent_indices: Sequence[int]) -> np.ndarray:
    matrices = []
    for index, joint in enumerate(joints):
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, 3] = joint
        matrices.append(np.linalg.inv(matrix).astype(np.float32))
    return np.stack(matrices, axis=0)
