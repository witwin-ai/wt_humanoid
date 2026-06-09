"""Integration test for wt_humanoid against real licensed SMPL model files."""
import inspect
import os

import numpy as np
import pytest


def _patch_chumpy_py311():
    # chumpy still calls inspect.getargspec on modern Python. Patch only inside
    # this integration boundary so normal application code stays untouched.
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec


@pytest.mark.integration
def test_smpl_motion_frame_bakes_with_real_smpl_model_files(tmp_path):
    model_root = os.environ.get("WITWIN_SMPL_MODEL_ROOT")
    if not model_root:
        pytest.skip("Set WITWIN_SMPL_MODEL_ROOT to a directory containing SMPL .pkl model files.")

    _patch_chumpy_py311()

    from witwin_server.core.components import PlatformGeometryComponent
    from witwin_server.features.platform.bridge import GeometryMap
    from wt_humanoid.components.smpl import SMPLBodyComponent, SMPLMotionComponent

    motion_path = tmp_path / "real_model_smoke_motion.npz"
    np.savez(
        motion_path,
        poses=np.zeros((2, 72), dtype=np.float32),
        betas=np.zeros(10, dtype=np.float32),
        trans=np.zeros((2, 3), dtype=np.float32),
        fps=30.0,
    )

    body = SMPLBodyComponent()
    body.gender = "male"
    body.model_root = model_root
    geom = PlatformGeometryComponent()
    geom.kind = "smpl"

    class _Owner:
        def get_component(self, name):
            return geom if name == "PlatformGeometry" else None

    body._owner = _Owner()

    motion = SMPLMotionComponent()
    motion.motion_path = str(motion_path)
    motion.current_frame = 1
    motion.apply_current_frame(body)

    platform_body = GeometryMap.to_platform(
        geom,
        position=[0.0, 0.0, 0.0],
        rotation_quat=[1.0, 0.0, 0.0, 0.0],
    )
    vertices, faces = platform_body.to_mesh()

    assert tuple(vertices.shape) == (6890, 3)
    assert tuple(faces.shape) == (13776, 3)
