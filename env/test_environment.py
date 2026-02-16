# test_with_render.py
from dynamic_nav_env import DynamicObstacleNavEnv
import numpy as np
import time

env = DynamicObstacleNavEnv(n_obstacles=5, render_mode="human")

print("="*60)
print("Visual Navigation Test - Watch the robot!")
print("="*60)

for episode in range(3):
    obs, info = env.reset()
    
    print(f"\n--- Episode {episode+1} ---")
    print(f"Robot (blue/red/green wheels): {info['robot_pos']}")
    print(f"Goal: {info['goal_pos']}")
    print(f"Distance: {info['goal_distance']:.2f}m")
    
    for step in range(1000):
        # Controller
        robot_pos = info['robot_pos']
        goal_pos = info['goal_pos']
        goal_vec = goal_pos - robot_pos
        goal_dist = np.linalg.norm(goal_vec)
        
        if goal_dist > 0.1:
            direction = goal_vec / goal_dist
            action = np.array([1.5 * direction[0], 1.5 * direction[1], 0.0])
        else:
            action = np.array([0.0, 0.0, 0.0])
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render
        env.render()
        time.sleep(0.01)  # Slow down for viewing
        
        if step % 100 == 0:
            print(f"  Step {step}: dist={info['goal_distance']:.2f}m")
        
        if terminated or truncated:
            print(f"\n  Episode ended at step {step}")
            print(f"  Goal reached: {info['goal_reached']}")
            print(f"  Collision: {info['collision']}")
            time.sleep(2)  # Pause before next episode
            break

env.close()
print("\n✅ Visual test complete!")