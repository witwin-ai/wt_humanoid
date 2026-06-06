"""
Library items for the Humanoid plugin.

Registers the humanoid model as a library item that can be
dragged into the scene from the Library panel.
"""
from pathlib import Path
from witwin import Library

# Get the path to the model file (relative to this plugin)
PLUGIN_DIR = Path(__file__).parent
MODEL_PATH = PLUGIN_DIR / "models" / "humanbody.glb"

# Register the humanoid prefab in the basic category
Library.register_item(
    id="humanoid",
    name="Humanoid",
    category="basic",
    icon="person-standing",
    description="A rigged humanoid model with skeleton",
    model_path=str(MODEL_PATH),
    object_type="mesh"
)
