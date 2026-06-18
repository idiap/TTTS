#
# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Teng Xue <teng.xue@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import os
import time
import numpy as np
import genesis as gs
import torch

from datetime import datetime
from copy import deepcopy
from scipy.spatial.transform import Rotation as Rot


from panda_robot.config import absjoin, PANDA_XML, SHELF_URDF, BASE_URDF, \
    BASE_URDF


class GenesisEnv():
    def __init__(self,
                 n_envs=1, env_spacing=(3.0, 3.0), 
                 use_gui=True, gpu=True,
                 p2p_motion=None):
        if gpu:
            if gs._initialized:
                pass
            else:
                gs.init(backend=gs.gpu, logging_level="warning")


            self.device = 'cuda'
        else:
            if gs._initialized:
                pass
            else:
                gs.init(backend=gs.cpu, logging_level="warning")

            self.device = 'cpu'
        self.scene = gs.Scene(show_viewer=use_gui,
            sim_options=gs.options.SimOptions(
                dt=1.0/240,
                gravity=(0, 0, -9.81),
                ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(-5., 0.0, 5.5),
                camera_lookat=(0.0, 0.0, 0.0),
                camera_fov=40,),
                )
        
        # Useful for recording videos
        self.cam = self.scene.add_camera(
                res    = (1280, 960),
                pos    = (-5.0, 3.0, 5.0),
                lookat = (0.7, 0.0, 1.2),
                fov    = 40,
                GUI    = False
            )
        
        self.n_envs=n_envs
        self.env_spacing=env_spacing

        self.p2p = p2p_motion
        self.n_joints = 7

        env_infos = self.env_infos_gen()
        self.load_world(env_infos, render_utility=True)

    def env_infos_gen(self):
        """
        This function generates the environment information 
        useful for defining the simulation environment.
        """
        shelf_info = {
            "name": "shelf",
            "urdf": SHELF_URDF,
            "base_pose": [0.7, 0.0, 0.015, 0, 0, 90],
        }

        base_info = {
            "name": "base",
            "urdf": BASE_URDF,
            "base_pose": [0.0, 0.0, 0.35, 0, 0, 90]
        }

        robot_info = {
            "name": "panda",
            "xml": PANDA_XML,
            # "base_pose": [0.0, 0.0, 0.8, 0, 0, 0],
            "base_pose": [0.0, 0.0, 0, 0, 0, 0],
            # "conf": [0.0, -np.pi/4, 0.0, -np.pi/2, 
            #          0, np.pi/2, np.pi/4, 0.06, 0.06],
            "conf": [1.76, -4.5e-01, -1.77, -1.35,  4.9e-01,
                    2.36, -2.8e-01, 0.06, 0.06],
            "jnt_names": ["joint1", "joint2", "joint3",
                        "joint4", "joint5", "joint6",
                        "joint7", "finger_joint1", "finger_joint2", ],
            "kp": np.array([4500, 4500, 3500, 3500, 2000, 2000, 
                                2000, 100, 100]),
            "kv": np.array([450, 450, 350, 350, 200, 200, 200, 
                            10, 10]),
            "force_range": {
                "lower": np.array([-87, -87, -87, -87, -12, -12, 
                                    -12, -100, -100]),
                "upper": np.array([87,  87,  87, 87,  12,  12, 
                                    12,  100,  100]),
            },
        }

        

        utilities = {
            "target_x1": [0.70, -0.15, 1.5],
            "target_x2": [0.60, 0.15, 1.1]
        }

        env_infos = {
            # "shelf": shelf_info,
            # "base": base_info,
            "robot": robot_info,
            "utilities": utilities
        }

        return env_infos
        
    def load_robot(self, robot_info):
        if "urdf" in robot_info.keys():
            robot_id = self.scene.add_entity(
                    gs.morphs.URDF(file=robot_info["urdf"],
                                pos=robot_info["base_pose"][:3],
                                euler=robot_info["base_pose"][3:],
                                fixed=True),
                                )
        elif "xml" in robot_info.keys():
            robot_id = self.scene.add_entity(
                    gs.morphs.MJCF(file=robot_info["xml"],
                                pos=robot_info["base_pose"][:3],
                                euler=robot_info["base_pose"][3:],
                                ),
                                )
            
        robot_info["id"] = robot_id


    def load_fixture(self, fixture_info):
        fixture_id = self.scene.add_entity(
                gs.morphs.URDF(file=fixture_info["urdf"],
                            pos=fixture_info["base_pose"][:3],
                            euler=fixture_info["base_pose"][3:],
                            fixed=True,
                            ),
                            )
        fixture_info["id"] = fixture_id

    
    def load_world(self, env_infos, render_utility=False):
        # load a plane in PyBullet
        self.load_plane()

        # shelf_info = env_infos["shelf"]
        # self.load_fixture(shelf_info)
        # self.shelf = shelf_info["id"]

        # base_info = env_infos["base"]
        # self.load_fixture(base_info)
        # self.base = base_info["id"]

        robot_info = env_infos["robot"]
        self.load_robot(robot_info)
        self.robot = robot_info["id"]

        self.cube = self.scene.add_entity(
            gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.65, 0.0, 0.02))
        )

        if render_utility:
            utilities = env_infos["utilities"]
            for name, position in utilities.items():
                self.scene.add_entity(gs.morphs.Sphere(radius=0.02, 
                                                       pos=position, 
                                                       fixed=True,
                                                       collision=False))
                
        self.scene.build(n_envs=self.n_envs, 
                         env_spacing=self.env_spacing)
        self.set_robot_control_confs(robot_info)

    def get_end_effector_pos(self):
        return self.robot.get_link('grasping_target_hand').get_pos().cpu().numpy()

    def set_robot_control_confs(self, robot_info):
        # generate robots information
        robot = robot_info["id"]
        n_joints = len(robot_info["conf"])
        dofs_idx = [robot.get_joint(name).dof_idx_local for name in robot_info["jnt_names"][:n_joints]]
        
        # set positional gains
        robot.set_dofs_kp(robot_info["kp"], dofs_idx)

        # set velocity gains
        robot.set_dofs_kv(robot_info["kv"], dofs_idx)

        # set force range for safety
        robot.set_dofs_force_range(lower=robot_info["force_range"]["lower"],
                                    upper=robot_info["force_range"]["upper"], 
                                    dofs_idx_local = dofs_idx)
        self.motors_dof = np.arange(7)
        self.fingers_dof = np.arange(7, 9)

    def set_robot_configuration(self, robot, conf, robot_info):
        n_joints = conf.shape[-1]
        dofs_idx = [robot.get_joint(name).dof_idx_local for name in robot_info["jnt_names"][:n_joints]]
        robot.control_dofs_position(conf, dofs_idx)
        self.scene.step()

    # def step(self, robot, action):
    #     robot.set_qpos(action)
    #     self.scene.step()

    def step(self, action):
        self.robot.set_qpos(action, self.motors_dof)
        self.scene.step()

    def forward_configuration(self, env_infos, trajectory, record=False):
        robot = env_infos["robot"]["id"]; robot_info = env_infos["robot"]
        shelf = env_infos["shelf"]["id"]

        # add trajectory execution with step function
        collision = torch.zeros((trajectory.shape[0], trajectory.shape[1])).to(self.device)
        for t in range(trajectory.shape[1]):
            self.set_robot_configuration(robot, trajectory[:, t, :], robot_info)
            contact_info = robot.get_contacts(shelf)

            if contact_info['valid_mask'].shape[1] != 0:
                collision_flag = np.clip(np.sum(contact_info['valid_mask'], axis=1),
                                            a_min=0, a_max=1)
                # no collisin points for all batched environments
                # print("contact information", collision_flag)
                collision[:, t] = torch.tensor(collision_flag, dtype=torch.float32).to(self.device)

            if record:
                self.cam.render()
        
        return collision

    def reset(self, env_infos):
        robot_info = env_infos["robot"]
        robot = env_infos["robot"]["id"]
        eef = robot.get_link("hand")
        conf = torch.tile(torch.tensor(env_infos["robot"]["conf"]), (self.n_envs, 1))
        for i in range(10):
            self.step(robot, conf)
            # print(eef.get_pos())

    """Utility functions for robot trajectory generation"""    
    def get_traj(self, x, theta_0):
        ''' 
        Get the trajectory for a given decision variable x and initial configuration theta_0
        '''
        batch_size = x.shape[0]
        theta_1 = x[:,:self.n_joints] #final configuration
        w = x[:,self.n_joints:] # weights of the basis function for motion between theta_0 to theta_1
        if theta_0.shape[0]!=batch_size:
            theta_0 = theta_0.view(1,-1).repeat(batch_size,1)

        theta_t = self.p2p.gen_traj_p2p(theta_0,theta_1, w)#batchxtimexjoint_angle

        return theta_t
    

    """Utility functions for visualization"""

    def load_plane(self):
        self.scene.add_entity(gs.morphs.Plane())