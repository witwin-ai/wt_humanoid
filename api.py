"""Public Python API of wt_humanoid for other Plugins (D16 layer C).

Other Plugins import only this module, and only after listing ``wt_humanoid``
in their ``pluginDependencies``. Every other module here is private and may
change without notice.
"""
from __future__ import annotations

from .adapter.smpl_motion import SMPLMotionData, load_smpl_motion
from .adapter.smpl_skinned import SMPLSkinAsset, attach_smpl_skinned_mesh, build_smpl_skin_asset
from .components.smpl import SMPLBodyComponent, SMPLMotionComponent

__all__ = [
    "SMPLBodyComponent",
    "SMPLMotionComponent",
    "SMPLMotionData",
    "SMPLSkinAsset",
    "attach_smpl_skinned_mesh",
    "build_smpl_skin_asset",
    "load_smpl_motion",
]
