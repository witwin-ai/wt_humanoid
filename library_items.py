"""
Library items for the Human Body extension.

Registers the human body model as a library item that can be
dragged into the scene from the Library panel.
"""
from pathlib import Path
from witwin import Library

# Get the path to the model file (relative to this extension)
EXTENSION_DIR = Path(__file__).parent
MODEL_PATH = EXTENSION_DIR / "models" / "humanbody.glb"

# Register the human body prefab in the basic category
Library.register_item(
    id="humanbody",
    name="Human Body",
    category="basic",
    icon="person-standing",
    description="A rigged human body model with skeleton",
    model_path=str(MODEL_PATH),
    object_type="mesh"
)
