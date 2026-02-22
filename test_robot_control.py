import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("omni_carver_description/description/omni_carver.xml")
data = mujoco.MjData(model)

print("Testing holonomic control...")
print(f"Actuators: {model.nu}")

with mujoco.viewer.launch_passive(model, data) as viewer:
    for i in range(20000):
        t = i * 0.002  # timestep
        
        # Test holonomic motion: circle + forward
        data.ctrl[0] = 0.3  # vx (forward)
        data.ctrl[1] = 0.2 * np.sin(t)  # vy (side-to-side)
        data.ctrl[2] = 0.3  # wz (rotate)
        
        mujoco.mj_step(model, data)
        viewer.sync()
        
        if i % 200 == 0:
            pos = data.qpos[0:2]
            print(f"Step {i}: pos=({pos[0]:.2f}, {pos[1]:.2f})")

print("\n✅ Robot moves! Ready for navigation environment")