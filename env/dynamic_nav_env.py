"""
Dynamic Obstacle Navigation Environment for MuJoCo
Supports K-GRU prediction + TD3 training
"""

import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple

# Use third-party viewer
try:
    import mujoco_viewer
    VIEWER_AVAILABLE = True
except ImportError:
    VIEWER_AVAILABLE = False
    print("Install viewer: pip install mujoco-python-viewer")

class DynamicObstacleNavEnv(gym.Env):
    """
    Omni-directional robot navigation with dynamic obstacles
    
    Features:
    - Speed-grouped obstacles (low/high speed)
    - Collision detection
    - Goal reaching
    - State logging for prediction training
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(
        self,
        model_path: str = "omni_carver_description/description/omni_carver.xml",
        n_obstacles: int = 5,
        low_speed_ratio: float = 0.5,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 1000,
    ):
        super().__init__()
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        
        # Environment parameters
        self.n_obstacles = n_obstacles
        self.low_speed_ratio = low_speed_ratio
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        
        # Speed groups
        self.low_speed_range = (0.1, 0.3)
        self.high_speed_range = (0.5, 1.0)
        
        # Arena boundaries
        self.arena_size = 10.0
        
        # Robot parameters
        self.robot_radius = 0.25
        self.goal_threshold = 0.3
        
        # Action space
        self.action_space = spaces.Box(
            low=np.array([-0.5, -0.5, -1.5]),
            high=np.array([0.5, 0.5, 1.5]),
            dtype=np.float32
        )
        
        # Observation space
        obs_dim = 10 + n_obstacles * 5
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Initialize obstacles
        self.obstacles = self._init_obstacles()
        
        # Viewer - INITIALIZE THIS FIRST!
        self.viewer = None
        
        # Data logging
        self.episode_data = {
            'robot_states': [],
            'obstacle_states': [],
            'actions': [],
            'rewards': []
        }
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = Noneal_angle,
        #  obs1_rel_x, obs1_rel_y, obs1_vx, obs1_vy, obs1_speed,
        #  ... (repeat for each obstacle)]
        obs_dim = 10 + n_obstacles * 5
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Initialize obstacles
        self.obstacles = self._init_obstacles()
        
        # Viewer
        self.viewer = None
        
        # Data logging for prediction training
        self.episode_data = {
            'robot_states': [],
            'obstacle_states': [],
            'actions': [],
            'rewards': []
        }
        
    def _init_obstacles(self):
        """Initialize obstacles with speed groups"""
        n_low = int(self.n_obstacles * self.low_speed_ratio)
        n_high = self.n_obstacles - n_low
        
        obstacles = []
        
        # Low-speed obstacles
        for i in range(n_low):
            obs = {
                'pos': np.random.uniform(-self.arena_size/2, self.arena_size/2, 2),
                'vel': np.random.uniform(*self.low_speed_range) * self._random_direction(),
                'radius': 0.3,
                'speed_group': 'low'
            }
            obstacles.append(obs)
        
        # High-speed obstacles
        for i in range(n_high):
            obs = {
                'pos': np.random.uniform(-self.arena_size/2, self.arena_size/2, 2),
                'vel': np.random.uniform(*self.high_speed_range) * self._random_direction(),
                'radius': 0.3,
                'speed_group': 'high'
            }
            obstacles.append(obs)
        
        return obstacles
    
    def _random_direction(self):
        """Generate random unit direction vector"""
        angle = np.random.uniform(0, 2*np.pi)
        return np.array([np.cos(angle), np.sin(angle)])
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        
        # Reset MuJoCo simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Reset robot position (random start)
        robot_start = np.random.uniform(-3, 3, 2)
        self.data.qpos[0:2] = robot_start
        self.data.qpos[3:7] = [1, 0, 0, 0]  # quaternion
        
        # Set goal (random, but far from start)
        while True:
            self.goal = np.random.uniform(-4, 4, 2)
            if np.linalg.norm(self.goal - robot_start) > 2.0:  # At least 2m away
                break
        
        # Reset obstacles (ensure minimum distance from robot)
        self.obstacles = []
        while len(self.obstacles) < self.n_obstacles:
            obs_pos = np.random.uniform(-self.arena_size/2, self.arena_size/2, 2)
            
            # Check minimum distance from robot (1.5m)
            if np.linalg.norm(obs_pos - robot_start) > 1.5:
                # Determine speed group
                is_low_speed = len(self.obstacles) < int(self.n_obstacles * self.low_speed_ratio)
                speed_range = self.low_speed_range if is_low_speed else self.high_speed_range
                
                obs = {
                    'pos': obs_pos,
                    'vel': np.random.uniform(*speed_range) * self._random_direction(),
                    'radius': 0.3,
                    'speed_group': 'low' if is_low_speed else 'high'
                }
                self.obstacles.append(obs)
        
        # Rest of reset...
        self.current_step = 0
        self.episode_data = {'robot_states': [], 'obstacle_states': [], 'actions': [], 'rewards': []}
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action):
        # Apply action to robot
        self.data.ctrl[:] = action
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        # Update obstacles
        self._update_obstacles()
        
        # Get observation
        obs = self._get_obs()
        
        # Calculate reward
        reward, reward_info = self._calculate_reward()
        
        # Check termination
        terminated = self._check_collision() or self._check_goal_reached()
        truncated = self.current_step >= self.max_episode_steps
        
        # Log data
        self._log_step(obs, action, reward)
        
        self.current_step += 1
        
        info = self._get_info()
        info.update(reward_info)
        info['collision'] = self._check_collision()
        info['goal_reached'] = self._check_goal_reached()
        
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self):
        """Get observation"""
        # Robot state
        robot_pos = self.data.qpos[0:2]
        robot_theta = self._get_robot_theta()
        robot_vel = self.data.qvel[0:3]  # vx, vy, vtheta
        
        # Goal info (relative)
        goal_vec = self.goal - robot_pos
        goal_dist = np.linalg.norm(goal_vec)
        goal_angle = np.arctan2(goal_vec[1], goal_vec[0]) - robot_theta
        
        # Normalize angle to [-pi, pi]
        goal_angle = (goal_angle + np.pi) % (2*np.pi) - np.pi
        
        obs = np.concatenate([
            robot_pos,
            [robot_theta],
            robot_vel,
            self.goal,
            [goal_dist, goal_angle]
        ])
        
        # Obstacle info (relative)
        for obstacle in self.obstacles:
            rel_pos = obstacle['pos'] - robot_pos
            speed = np.linalg.norm(obstacle['vel'])
            
            obs = np.concatenate([
                obs,
                rel_pos,
                obstacle['vel'],
                [speed]
            ])
        
        return obs.astype(np.float32)
    
    def _get_robot_theta(self):
        """Extract robot orientation from quaternion"""
        quat = self.data.qpos[3:7]
        # Simple 2D rotation extraction
        return np.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]),
                         1 - 2*(quat[2]**2 + quat[3]**2))
    
    def _update_obstacles(self):
        """Update obstacle positions"""
        dt = self.model.opt.timestep
        
        for obs in self.obstacles:
            # Update position
            obs['pos'] += obs['vel'] * dt
            
            # Bounce off walls
            for i in range(2):
                if abs(obs['pos'][i]) > self.arena_size/2:
                    obs['pos'][i] = np.sign(obs['pos'][i]) * self.arena_size/2
                    obs['vel'][i] *= -1
            
            # Occasional direction change (adds unpredictability)
            if np.random.random() < 0.005:  # 0.5% chance per step
                obs['vel'] = np.linalg.norm(obs['vel']) * self._random_direction()
    
    def _calculate_reward(self):
        """Calculate reward"""
        robot_pos = self.data.qpos[0:2]
        goal_dist = np.linalg.norm(self.goal - robot_pos)
        
        reward = 0.0
        reward_info = {}
        
        # Goal progress reward
        reward += -0.1 * goal_dist
        
        # Goal reached bonus
        if goal_dist < self.goal_threshold:
            reward += 100.0
            reward_info['goal_bonus'] = 100.0
        
        # Collision penalty
        if self._check_collision():
            reward += -50.0
            reward_info['collision_penalty'] = -50.0
        
        # Step penalty (encourage efficiency)
        reward += -0.01
        
        # Risk penalty (anticipatory - for later use with prediction)
        # This is where your K-GRU predictions will plug in
        risk_penalty = self._calculate_risk_penalty()
        reward += risk_penalty
        reward_info['risk_penalty'] = risk_penalty
        
        return reward, reward_info
    
    def _calculate_risk_penalty(self):
        """Calculate risk based on proximity to obstacles"""
        robot_pos = self.data.qpos[0:2]
        robot_vel = self.data.qvel[0:2]
        
        risk = 0.0
        safety_dist = 1.0  # meters
        
        for obs in self.obstacles:
            dist = np.linalg.norm(obs['pos'] - robot_pos)
            
            if dist < safety_dist:
                # Higher penalty for closer obstacles
                risk -= 0.1 * (1.0 - dist/safety_dist)
                
                # Extra penalty for high-speed obstacles
                if obs['speed_group'] == 'high':
                    risk -= 0.05
        
        return risk
    
    def _check_collision(self):
        """Check if robot collides with obstacles"""
        robot_pos = self.data.qpos[0:2]
        
        for obs in self.obstacles:
            dist = np.linalg.norm(obs['pos'] - robot_pos)
            if dist < (self.robot_radius + obs['radius']):
                return True
        
        return False
    
    def _check_goal_reached(self):
        """Check if robot reached goal"""
        robot_pos = self.data.qpos[0:2]
        return np.linalg.norm(self.goal - robot_pos) < self.goal_threshold
    
    def _log_step(self, obs, action, reward):
        """Log data for prediction training"""
        self.episode_data['robot_states'].append(obs[:6].copy())
        self.episode_data['obstacle_states'].append(
            [{'pos': o['pos'].copy(), 'vel': o['vel'].copy(), 
              'speed': np.linalg.norm(o['vel']), 'group': o['speed_group']} 
             for o in self.obstacles]
        )
        self.episode_data['actions'].append(action.copy())
        self.episode_data['rewards'].append(reward)
    
    def _get_info(self):
        """Get additional info"""
        robot_pos = self.data.qpos[0:2]
        return {
            'robot_pos': robot_pos.copy(),
            'goal_pos': self.goal.copy(),
            'goal_distance': np.linalg.norm(self.goal - robot_pos),
            'episode_step': self.current_step,
            'obstacles': [{'pos': o['pos'].copy(), 'vel': o['vel'].copy(), 
                          'speed_group': o['speed_group']} for o in self.obstacles]
        }
    
    def render(self):
        if self.render_mode == "human" and VIEWER_AVAILABLE:
            if self.viewer is None:
                self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
                # Camera settings
                self.viewer.cam.distance = 15
                self.viewer.cam.elevation = -30
                self.viewer.cam.azimuth = 90
            
            self.viewer.render()
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None