"""SMPL motion loading and component application for wt_humanoid."""
import numpy as np


def test_loads_amass_style_npz_and_returns_frame(tmp_path):
    from wt_humanoid.adapter.smpl_motion import load_smpl_motion

    path = tmp_path / "walk.npz"
    np.savez(
        path,
        poses=np.arange(3 * 72, dtype=np.float32).reshape(3, 72),
        betas=np.linspace(0.0, 0.9, 10, dtype=np.float32),
        trans=np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [2.0, 4.0, 6.0]], dtype=np.float32),
        mocap_framerate=np.array(60.0, dtype=np.float32),
    )

    motion = load_smpl_motion(path)
    frame = motion.frame(1)

    assert motion.frame_count == 3
    assert motion.fps == 60.0
    assert frame["frame_index"] == 1
    assert frame["time"] == 1.0 / 60.0
    assert frame["pose"] == np.arange(72, 144, dtype=np.float32).tolist()
    assert frame["shape"] == np.linspace(0.0, 0.9, 10, dtype=np.float32).tolist()
    assert frame["translation"] == [1.0, 2.0, 3.0]


def test_frame_selection_clamps_or_loops(tmp_path):
    from wt_humanoid.adapter.smpl_motion import load_smpl_motion

    path = tmp_path / "short.npz"
    np.savez(path, poses=np.arange(2 * 6, dtype=np.float32).reshape(2, 6), fps=10.0)

    motion = load_smpl_motion(path)

    assert motion.frame(-5)["frame_index"] == 0
    assert motion.frame(99)["frame_index"] == 1
    assert motion.frame(3, loop=True)["frame_index"] == 1


def test_accepts_custom_key_names_and_joint_axis_shape(tmp_path):
    from wt_humanoid.adapter.smpl_motion import load_smpl_motion

    path = tmp_path / "custom.npz"
    poses = np.arange(2 * 24 * 3, dtype=np.float32).reshape(2, 24, 3)
    np.savez(path, smpl_pose=poses, shape=np.ones(10, dtype=np.float32), root=np.ones((2, 3), dtype=np.float32))

    motion = load_smpl_motion(path, pose_key="smpl_pose", betas_key="shape", trans_key="root", fps=24.0)

    assert motion.pose_size == 72
    assert motion.fps == 24.0
    assert motion.frame(0)["pose"] == poses.reshape(2, 72)[0].tolist()
    assert motion.frame(1)["translation"] == [1.0, 1.0, 1.0]


def test_component_loads_motion_and_applies_current_frame_to_body(tmp_path):
    from wt_humanoid.components.smpl import SMPLBodyComponent, SMPLMotionComponent

    path = tmp_path / "walk.npz"
    np.savez(
        path,
        poses=np.arange(2 * 72, dtype=np.float32).reshape(2, 72),
        betas=np.arange(10, dtype=np.float32),
        trans=np.array([[0.0, 0.0, 0.0], [0.5, 0.25, -1.0]], dtype=np.float32),
        fps=20.0,
    )

    body = SMPLBodyComponent()
    motion = SMPLMotionComponent()
    motion.motion_path = str(path)
    motion.current_frame = 1

    motion.load_motion()
    applied = motion.apply_current_frame(body)

    assert motion.frame_count == 2
    assert motion.fps == 20.0
    assert applied["frame_index"] == 1
    assert body.pose == np.arange(72, 144, dtype=np.float32).tolist()
    assert body.shape == np.arange(10, dtype=np.float32).tolist()
    assert body.source_path == str(path)
    assert np.allclose(motion.current_translation, [0.5, 0.25, -1.0])
