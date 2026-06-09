"""SMPL body and motion components owned by wt_humanoid."""
from __future__ import annotations

from typing import Any, Optional

from witwin_server.core.components import (
    Component,
    button,
    component,
    component_field,
    float_field,
    int_field,
    list_field,
    string_field,
    vector3_field,
)

from ..adapter.smpl_motion import load_smpl_motion


_CAT = "Human/SMPL"


@component(name="SMPLBody", display_name="SMPL Body", category=_CAT)
class SMPLBodyComponent(Component):
    """Single-frame SMPL body parameters.

    The component is the SMPL source of truth in Studio. Domain-specific SMPL
    state lives here; platform adapters may mirror it into generic geometry when
    exporting to solver runtimes.
    """

    gender = string_field("neutral", options=["male", "female", "neutral"], enum_toggle=True)
    model_root = string_field("", description="Directory containing SMPL model files")
    source_path = string_field("", description="Source SMPL motion/static file")
    pose = list_field(float_field(), default=[], hidden=True, description="Current SMPL pose vector")
    shape = list_field(float_field(), default=[], hidden=True, description="SMPL shape/betas vector")
    root_translation = vector3_field([0.0, 0.0, 0.0], description="Current root translation from motion")

    def to_platform_geometry_params(self) -> dict:
        """Return parameters compatible with generic ``PlatformGeometry(kind='smpl')``."""
        return {
            "kind": "smpl",
            "pose": [float(x) for x in self.pose],
            "shape": [float(x) for x in self.shape],
            "gender": str(self.gender),
            "model_root": str(self.model_root),
        }

    def apply_frame(self, frame: dict) -> None:
        """Apply a normalized SMPL motion frame to this body."""
        self.pose = frame["pose"]
        self.shape = frame["shape"]
        self.root_translation = frame["translation"]
        self.source_path = frame.get("source_path", self.source_path)
        self.sync_platform_geometry()

    def sync_platform_geometry(self) -> None:
        """Mirror current SMPL params to a sibling generic PlatformGeometry if present."""
        owner = getattr(self, "owner", None)
        if owner is None:
            return
        geom = owner.get_component("PlatformGeometry")
        if geom is None:
            return
        params = self.to_platform_geometry_params()
        geom.kind = "smpl"
        geom.pose = params["pose"]
        geom.shape = params["shape"]
        geom.gender = params["gender"]
        geom.model_root = params["model_root"]


@component(name="SMPLMotion", display_name="SMPL Motion", category=_CAT)
class SMPLMotionComponent(Component):
    """SMPL pose sequence source controlled by frame index."""

    target_body = component_field("SMPLBody", description="SMPL body to receive the current frame")
    motion_path = string_field("", description="AMASS/SMPL .npz motion file")
    pose_key = string_field("", description="Optional pose key override")
    betas_key = string_field("", description="Optional betas/shape key override")
    trans_key = string_field("", description="Optional translation key override")
    fps = float_field(30.0, min=0.001, description="Frames per second")
    frame_count = int_field(0, min=0, readonly=True, description="Loaded frame count")
    current_frame = int_field(0, min=0, description="Frame to apply")
    loop = string_field("off", options=["off", "on"], enum_toggle=True, description="Loop frame selection")
    status = string_field("", readonly=True, description="Load/apply status")
    current_pose = list_field(float_field(), default=[], hidden=True, readonly=True)
    current_shape = list_field(float_field(), default=[], hidden=True, readonly=True)
    current_translation = vector3_field([0.0, 0.0, 0.0], readonly=True)

    _motion_cache: Optional[Any] = None
    _motion_cache_key: Optional[tuple] = None

    @button(display_name="Load Motion")
    def load_motion(self) -> dict:
        """Load motion metadata and current frame from ``motion_path``."""
        motion = self._load_motion()
        self.frame_count = motion.frame_count
        self.fps = motion.fps
        frame = motion.frame(int(self.current_frame), loop=self.loop == "on")
        self._set_current_frame_payload(frame)
        self.status = f"Loaded {motion.frame_count} frames at {motion.fps:g} fps"
        return {"ok": True, "frame_count": motion.frame_count, "fps": motion.fps}

    @button(display_name="Apply Current Frame")
    def apply_current_frame_button(self) -> dict:
        """Apply ``current_frame`` to the referenced body."""
        body = self.get_referenced_component("target_body")
        if body is None:
            self.status = "No target SMPLBody selected"
            return {"ok": False, "error": self.status}
        frame = self.apply_current_frame(body)
        return {"ok": True, **frame}

    def apply_current_frame(self, body: Optional[SMPLBodyComponent] = None) -> dict:
        """Apply ``current_frame`` to ``body`` or the referenced target body."""
        motion = self._load_motion()
        frame = motion.frame(int(self.current_frame), loop=self.loop == "on")
        self.frame_count = motion.frame_count
        self.fps = motion.fps
        self._set_current_frame_payload(frame)
        target = body or self.get_referenced_component("target_body")
        if target is not None:
            target.apply_frame(frame)
        self.status = f"Applied frame {frame['frame_index']}"
        return frame

    def on_value_change(self, field_name, old_val, new_val):
        """Generic timeline hook: applying a keyed current_frame updates the body."""
        if field_name != "current_frame" or not self.motion_path:
            return
        try:
            self.apply_current_frame()
        except Exception as exc:  # noqa: BLE001 - component UI should show the error, not crash playback
            self.status = f"Failed to apply frame: {exc}"

    def on_timeline_apply(self, *, field_name, value, time, frame, timeline):
        """Apply frame-index tracks after the generic Timeline writes component fields."""
        if field_name == "current_frame" and self.motion_path:
            self.apply_current_frame()

    def _load_motion(self):
        cache_key = (
            str(self.motion_path),
            str(self.pose_key).strip() or None,
            str(self.betas_key).strip() or None,
            str(self.trans_key).strip() or None,
        )
        if not cache_key[0]:
            raise ValueError("SMPL motion_path is empty.")
        if self._motion_cache is None or self._motion_cache_key != cache_key:
            self._motion_cache = load_smpl_motion(
                cache_key[0],
                pose_key=cache_key[1],
                betas_key=cache_key[2],
                trans_key=cache_key[3],
            )
            self._motion_cache_key = cache_key
        return self._motion_cache

    def _set_current_frame_payload(self, frame: dict) -> None:
        self.current_pose = frame["pose"]
        self.current_shape = frame["shape"]
        self.current_translation = frame["translation"]
