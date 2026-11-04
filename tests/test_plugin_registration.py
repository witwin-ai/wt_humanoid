"""wt_humanoid registers SMPL components and exports platform-compatible params."""
from pathlib import Path

from witwin_server.api import testing


def test_manifest_is_enabled_by_default():
    manifest = testing.read_manifest(Path(__file__).resolve().parents[1] / "witwin.toml")

    assert manifest.enabled is True


def test_plugin_import_registers_smpl_components():
    import wt_humanoid  # noqa: F401
    registry = testing.component_types()

    assert "SMPLBody" in registry
    assert "SMPLMotion" in registry


def test_plugin_registers_humanoid_library_items():
    import wt_humanoid  # noqa: F401
    item_ids = testing.library_item_ids()

    assert "humanoid" in item_ids
    assert "smpl_body" in item_ids
    assert "smpl_motion" in item_ids


def test_smpl_body_exports_platform_geometry_params():
    from wt_humanoid.components.smpl import SMPLBodyComponent

    body = SMPLBodyComponent()
    body.gender = "female"
    body.model_root = "E:/models/smpl"
    body.pose = [0.1, 0.2, 0.3]
    body.shape = [1.0, 2.0]

    assert body.to_platform_geometry_params() == {
        "kind": "smpl",
        "pose": [0.1, 0.2, 0.3],
        "shape": [1.0, 2.0],
        "gender": "female",
        "model_root": "E:/models/smpl",
    }


def test_smpl_body_syncs_sibling_platform_geometry():
    from witwin_server.api import SceneObject, PlatformGeometryComponent
    from wt_humanoid.components.smpl import SMPLBodyComponent

    obj = SceneObject(name="Human", mesh_type="Empty")
    body = SMPLBodyComponent()
    geom = PlatformGeometryComponent()
    obj.add_component(body)
    obj.add_component(geom)

    body.gender = "male"
    body.model_root = "E:/models/smpl"
    body.apply_frame(
        {
            "pose": [0.1, 0.2],
            "shape": [1.0],
            "translation": [0.0, 1.0, 2.0],
            "source_path": "walk.npz",
        }
    )

    assert geom.kind == "smpl"
    assert geom.pose == [0.1, 0.2]
    assert geom.shape == [1.0]
    assert geom.gender == "male"
    assert geom.model_root == "E:/models/smpl"
