import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("omni_carver_description/description/omni_carver.xml")
print("✓ Model loaded!")
mujoco.viewer.launch(model)