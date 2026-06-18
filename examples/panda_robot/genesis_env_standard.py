#
# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Teng Xue <teng.xue@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import numpy as np
import genesis as gs
import time
import matplotlib.pyplot as plt
from panda_robot.config import absjoin, PANDA_XML, SHELF_URDF, BASE_URDF, BASE_URDF

class GenesisEnv:
    def __init__(self, n_envs=16):
        self.n_envs = n_envs
        self._init_genesis()
        self._create_scene()
        self._setup_robot()
        self.TARGET_POS = self.cube.get_pos().cpu().numpy()
    
    def _init_genesis(self):
        try:
            gs.init(backend=gs.gpu)
        except RuntimeError:
            pass  # if already initialized, pass
        
    def _create_scene(self):
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, -1, 1.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                max_FPS=60,
            ),
            sim_options=gs.options.SimOptions(dt=0.01),
            show_viewer=True,
        )
        self.scene.add_entity(gs.morphs.Plane())
        self.cube = self.scene.add_entity(
            gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.65, 0.0, 0.02))
        )
        # self.cube.set_color((0.8, 0.2, 0.2, 1.0))
    
    def _setup_robot(self):
        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file=PANDA_XML)
        )
        self.scene.build(n_envs=self.n_envs, env_spacing=(1.0, 1.0))
        
        self.motors_dof = np.arange(7)
        self.fingers_dof = np.arange(7, 9)
        
        self.franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
        self.franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
        self.franka.set_dofs_force_range(
            np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            np.array([87, 87, 87, 87, 12, 12, 12, 100, 100])
        )
        self.INITIAL_JOINTS = np.zeros(7)
        self.reset_robot()
    
    def reset_robot(self):
        try:
            self.scene.reset()
        except Exception:
            self.franka.set_dofs_target(self.INITIAL_JOINTS, self.motors_dof)
            for _ in range(50):
                self.scene.step()
    
    def get_end_effector_pos(self):
        return self.franka.get_link('grasping_target_hand').get_pos().cpu().numpy()
    
    def visualize_joint_angles(self, joint_angles):
        """input joint angles"""
        self.reset_robot()
        self.franka.set_qpos(joint_angles, self.motors_dof)
        self.scene.step()
        final_ee_pos = self.get_end_effector_pos()
        # print("Final EE Pos:", final_ee_pos)
        cost = np.sum((final_ee_pos - self.TARGET_POS) ** 2)
        return cost

if __name__ == "__main__":
    visualizer = GenesisEnv()
    test_joint_angles = np.array([0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.5])
    cost = visualizer.visualize_joint_angles(test_joint_angles)
    print("Cost:", cost)
