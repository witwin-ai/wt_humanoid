"""
Library items for the Humanoid plugin.

Registers the humanoid model as a library item that can be
dragged into the scene from the Library panel.
"""
from pathlib import Path
from witwin_server.api import Library, SceneObject

# Get the path to the model file (relative to this plugin)
PLUGIN_DIR = Path(__file__).parent
MODEL_PATH = PLUGIN_DIR / "models" / "humanbody.glb"
_CATEGORY = "basic"


def _make_smpl_body(ctx):
    from witwin_server.api import PlatformGeometryComponent

    from .components.smpl import SMPLBodyComponent

    obj = SceneObject(name=ctx.name, mesh_type="Empty")
    body = SMPLBodyComponent()
    geom = PlatformGeometryComponent()
    geom.kind = "smpl"
    obj.add_component(body)
    obj.add_component(geom)
    body.sync_platform_geometry()
    obj.get_component("Transform").position = list(ctx.position)
    return obj


def _make_smpl_motion(ctx):
    from .components.smpl import SMPLMotionComponent

    obj = SceneObject(name=ctx.name, mesh_type="Empty")
    obj.add_component(SMPLMotionComponent())
    obj.get_component("Transform").position = list(ctx.position)
    return obj

# Register the humanoid prefab in the basic category
Library.register_item(
    id="humanoid",
    name="Humanoid",
    category=_CATEGORY,
    icon="person-standing",
    description="A rigged humanoid model with skeleton",
    model_path=str(MODEL_PATH),
    object_type="mesh"
)

Library.register_item(
    id="smpl_body",
    name="SMPL Body",
    category=_CATEGORY,
    icon="person-standing",
    description="SMPL body parameters owned by wt_humanoid",
    object_type="empty",
    factory=_make_smpl_body,
)

Library.register_item(
    id="smpl_motion",
    name="SMPL Motion",
    category=_CATEGORY,
    icon="activity",
    description="AMASS/SMPL .npz motion source that drives an SMPL Body",
    object_type="empty",
    factory=_make_smpl_motion,
)
