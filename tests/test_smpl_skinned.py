"""SMPL SkinnedMesh binding and pose synchronization tests."""
from __future__ import annotations

import numpy as np
import pytest


def test_smpl_skin_loader_uses_shared_chumpy_free_model_reader(monkeypatch, tmp_path):
    import types
    import wt_humanoid.adapter.smpl_skinned as smpl_skinned

    captured = {}

    def load(path):
        captured["path"] = path
        return {"shapedirs": np.zeros((1, 1, 1))}

    class Struct:
        def __init__(self, **kwargs):
            self.values = kwargs

    monkeypatch.setattr(smpl_skinned, "load_smpl_model_data", load)
    result = smpl_skinned._load_smpl_data_struct(
        tmp_path, "neutral", types.SimpleNamespace(utils=types.SimpleNamespace(Struct=Struct))
    )

    assert captured["path"] == tmp_path / "smpl" / "SMPL_NEUTRAL.pkl"
    assert result.values["shapedirs"].shape == (1, 1, 1)


def _tiny_skin_asset():
    from wt_humanoid.adapter.smpl_skinned import SMPLSkinAsset

    return SMPLSkinAsset(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.uint32),
        skin_indices=np.array(
            [
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
            ],
            dtype=np.int64,
        ),
        skin_weights=np.array(
            [
                [0.75, 0.25, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        joint_positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        parent_indices=[-1, 0],
        inverse_bind_matrices=np.array(
            [
                np.eye(4, dtype=np.float32),
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, -1.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            ],
            dtype=np.float32,
        ),
    )


def test_attach_smpl_skinned_mesh_creates_bones_and_skinning_data():
    from witwin_server.api import SceneObject, Scene
    from wt_humanoid.adapter.smpl_skinned import attach_smpl_skinned_mesh
    from wt_humanoid.components.smpl import SMPLBodyComponent

    scene = Scene(scene_id="main")
    body_obj = SceneObject(name="SMPL Body", mesh_type="Empty")
    body = SMPLBodyComponent()
    body_obj.add_component(body)
    scene.add_object(body_obj)

    result = attach_smpl_skinned_mesh(scene, body_obj, body, asset=_tiny_skin_asset())

    skinned = body_obj.get_component("SkinnedMesh")
    assert result["ok"] is True
    assert skinned is not None
    assert skinned.bone_count == 2
    assert skinned.bone_ids == result["bone_ids"]
    assert skinned.root_bone_id == result["bone_ids"][0]
    assert skinned.get_mesh_data()["faces"] == [[0, 1, 2]]
    assert skinned.source["type"] == "generated_mesh"
    assert skinned.source["storage"] == "runtime"
    assert skinned.show_bones is False

    root_bone = scene.get_object(result["bone_ids"][0])
    child_bone = scene.get_object(result["bone_ids"][1])
    assert root_bone is not None
    assert child_bone is not None
    assert root_bone.parent_id == body_obj.id
    assert child_bone.parent_id == root_bone.id
    assert child_bone.get_component("Transform").position.tolist() == pytest.approx([0.0, 1.0, 0.0])


def test_smpl_body_frame_updates_skinned_bone_rotations():
    from witwin_server.api import SceneObject, Scene
    from wt_humanoid.adapter.smpl_skinned import attach_smpl_skinned_mesh
    from wt_humanoid.components.smpl import SMPLBodyComponent

    scene = Scene(scene_id="main")
    body_obj = SceneObject(name="SMPL Body", mesh_type="Empty")
    body = SMPLBodyComponent()
    body_obj.add_component(body)
    scene.add_object(body_obj)
    result = attach_smpl_skinned_mesh(scene, body_obj, body, asset=_tiny_skin_asset())

    pose = [0.0] * 72
    pose[3:6] = [0.0, 0.0, np.pi / 2.0]
    body.apply_frame(
        {
            "pose": pose,
            "shape": [0.0] * 10,
            "translation": [0.0, 0.0, 0.0],
            "source_path": "test.npz",
        }
    )

    root_bone = scene.get_object(result["bone_ids"][0])
    child_bone = scene.get_object(result["bone_ids"][1])
    assert root_bone.get_component("Transform").rotation.tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert child_bone.get_component("Transform").get_quaternion().tolist() == pytest.approx(
        [0.0, 0.0, 0.70710677, 0.70710677],
        abs=1e-5,
    )
