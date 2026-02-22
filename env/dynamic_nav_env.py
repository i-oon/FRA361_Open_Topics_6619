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
    - NO automatic episode termination (controlled by caller)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(
        self,
        model_path: str = "omni_carver_description/description/omni_carver.xml",
        n_obstacles: int = 5,
        low_speed_ratio: float = 0.5,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize environment
        
        Args:
            model_path: Path to MuJoCo XML model
            n_obstacles: Number of dynamic obstacles
            low_speed_ratio: Ratio of low-speed to total obstacles
            render_mode: Rendering mode ('human', 'rgb_array', or None)
        
        Note: NO max_episode_steps parameter - caller controls episode length!
        """
        super().__init__()
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        
        # Environment parameters
        self.n_obstacles = n_obstacles
        self.low_speed_ratio = low_speed_ratio
        self.current_step = 0
        
        # Speed groups
        self.low_speed_range = (0.5, 1.5)
        self.high_speed_range = (2.5, 4.5)
        
        # Arena boundaries
        self.arena_size = 10.0
        
        # Robot parameters
        self.robot_radius = 0.25
        self.goal_threshold = 0.3
        
        # Action space: [vx, vy, omega]
        self.action_space = spaces.Box(
            low=np.array([-0.5, -0.5, -1.5]),
            high=np.array([0.5, 0.5, 1.5]),
            dtype=np.float32
        )
        
        # Observation space
        # [robot_x, robot_y, robot_theta, robot_vx, robot_vy, robot_omega,
        #  goal_x, goal_y, goal_dist, goal_angle,
        #  obs1_rel_x, obs1_rel_y, obs1_vx, obs1_vy, obs1_speed,
        #  ... (repeat for each obstacle)]
        obs_dim = 10 + n_obstacles * 5
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Initialize obstacles (will be reset properly in reset())
        self.obstacles = []
        self.goal = np.zeros(2)
        
        # Viewer
        self.viewer = None
        
        # Data logging for prediction training
        self.episode_data = {
            'robot_states': [],
            'obstacle_states': [],
            'actions': [],
            'rewards': []
        }
    
    def _init_obstacles(self, robot_pos: np.ndarray):
        """
        Initialize obstacles with speed groups
        Ensures minimum distance from robot
        
        Args:
            robot_pos: Current robot position to avoid spawning on top of robot
        """
        n_low = int(self.n_obstacles * self.low_speed_ratio)
        n_high = self.n_obstacles - n_low
        
        obstacles = []
        min_dist_from_robot = 1.5  # meters
        
        # Low-speed obstacles
        for i in range(n_low):
            # Find position away from robot
            while True:
                pos = np.random.uniform(-self.arena_size/2, self.arena_size/2, 2)
                if np.linalg.norm(pos - robot_pos) > min_dist_from_robot:
                    break
            
            obs = {
                'pos': pos,
                'vel': np.random.uniform(*self.low_speed_range) * self._random_direction(),
                'radius': 0.3,
                'speed_group': 'low'
            }
            obstacles.append(obs)
        
        # High-speed obstacles
        for i in range(n_high):
            # Find position away from robot
            while True:
                pos = np.random.uniform(-self.arena_size/2, self.arena_size/2, 2)
                if np.linalg.norm(pos - robot_pos) > min_dist_from_robot:
                    break
            
            obs = {
                'pos': pos,
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
        """
        Reset environment
        
        Returns:
            obs: Initial observation
            info: Additional info
        """
        super().reset(seed=seed)
        
        # Reset MuJoCo simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Reset robot position (random start)
        robot_start = np.random.uniform(-3, 3, 2)
        self.data.qpos[0:2] = robot_start
        self.data.qpos[2] = 0.0  # z position
        self.data.qpos[3:7] = [1, 0, 0, 0]  # quaternion (no rotation)
        
        # Reset velocities
        self.data.qvel[:] = 0.0
        
        # Set goal (random, but far from start)
        while True:
            self.goal = np.random.uniform(-4, 4, 2)
            if np.linalg.norm(self.goal - robot_start) > 2.0:  # At least 2m away
                break
        
        # Initialize obstacles (with minimum distance from robot)
        self.obstacles = self._init_obstacles(robot_start)
        
        # Reset step counter
        self.current_step = 0
        
        # Reset data logging
        self.episode_data = {
            'robot_states': [],
            'obstacle_states': [],
            'actions': [],
            'rewards': []
        }
        
        # Forward simulation to stabilize
        mujoco.mj_forward(self.model, self.data)
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action):
        """
        Step simulation
        
        Args:
            action: [vx, vy, omega] robot control
        
        Returns:
            obs: Observation
            reward: Reward
            terminated: Whether episode ended (collision or goal)
            truncated: Whether episode was truncated (NOT USED - caller controls length)
            info: Additional info
        """
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
        
        # Check termination (ONLY collision or goal - NO time limit!)
        collision = self._check_collision()
        goal_reached = self._check_goal_reached()
        terminated = collision or goal_reached
        
        # NO truncation - caller controls episode length
        truncated = False
        
        # Log data
        self._log_step(obs, action, reward)
        
        self.current_step += 1
        
        # Build info
        info = self._get_info()
        info.update(reward_info)
        info['collision'] = collision
        info['goal_reached'] = goal_reached
        
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self):
        """
        Get observation
        
        Returns:
            obs: [robot_state (10), obstacle_states (n_obstacles × 5)]
        """
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
        risk_penalty = self._calculate_risk_penalty()
        reward += risk_penalty
        reward_info['risk_penalty'] = risk_penalty
        
        return reward, reward_info
    
    def _calculate_risk_penalty(self):
        """Calculate risk based on proximity to obstacles"""
        robot_pos = self.data.qpos[0:2]
        
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
        """Render environment"""
        if self.render_mode == "human" and VIEWER_AVAILABLE:
            if self.viewer is None:
                self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
                # Camera settings
                self.viewer.cam.distance = 15
                self.viewer.cam.elevation = -30
                self.viewer.cam.azimuth = 90
            
            self.viewer.render()
    
    def close(self):
        """Clean up resources"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None