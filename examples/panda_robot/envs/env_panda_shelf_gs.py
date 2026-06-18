#
# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Teng Xue <teng.xue@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

"""
This script defines the environment for panda 
picking an object from a shelf.
"""

import os
import time
import numpy as np
import genesis as gs
import torch

from datetime import datetime
from copy import deepcopy
from scipy.spatial.transform import Rotation as Rot


class PandaShelf():
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

    # def load_shelf(self, shelf_info):
    #     """
    #     This function loads a shelf with Box primitive.
    #     """
    #     pos = shelf_info["base_pose"][:3]

    #     # load the bottom layer
    #     shelf_bottom = self.scene.add_entity(
    #         gs.morphs.Box(pos=pos, 
    #                       size=(0.9, 0.4, 0.03),
    #                       collision=False,
    #                       fixed=True), )
    #     shelf_left = self.scene.add_entity(
    #         gs.morphs.Box(pos=(pos[0]-0.46, pos[1], pos[2]-0.01), 
    #                       size=(0.02, 0.4, 1.80),
    #                       collision=True,
    #                       fixed=True), )
    #     shelf_right = self.scene.add_entity(
    #         gs.morphs.Box(pos=(pos[0]+0.46, pos[1], pos[2]-0.01),
    #                       size=(0.02, 0.4, 1.80),
    #                       collision=True,
    #                       fixed=True), )
    #     shelf_top = self.scene.add_entity(
    #         gs.morphs.Box(pos=(pos[0], pos[1], pos[2]+0.9),
    #                       size=(0.4, 0.9, 0.03),
    #                       collision=False,
    #                       fixed=True), )
    #     shelf_up1 = self.scene.add_entity(
    #         gs.morphs.Box(pos=(pos[0], pos[1], pos[2]+0.45),
    #                         size=(0.4, 0.02, 0.9),
    #                         collision=True,
    #                         fixed=True), )
    def load_world(self, env_infos, render_utility=False):
        # load a plane in PyBullet
        self.load_plane()

        shelf_info = env_infos["shelf"]
        self.load_fixture(shelf_info)

        base_info = env_infos["base"]
        self.load_fixture(base_info)

        robot_info = env_infos["robot"]
        self.load_robot(robot_info)

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

    def set_robot_configuration(self, robot, conf, robot_info):
        n_joints = conf.shape[-1]
        dofs_idx = [robot.get_joint(name).dof_idx_local for name in robot_info["jnt_names"][:n_joints]]
        # robot.control_dofs_position(conf, dofs_idx)
        robot.set_qpos(conf, dofs_idx)
        self.scene.step()

    def step(self, robot, action):
        robot.set_qpos(action)
        self.scene.step()

    def set_final_qpos(self, env_infos, trajectory, record=False):
        robot = env_infos["robot"]["id"]; robot_info = env_infos["robot"]
        shelf = env_infos["shelf"]["id"]

        # add trajectory execution with step function
        # collision = torch.zeros((trajectory.shape[0], trajectory.shape[1])).to(self.device)
        # for t in range(trajectory.shape[1]):
        self.set_robot_configuration(robot, trajectory[:, -1, :], robot_info)
            # contact_info = robot.get_contacts(shelf)

            # if contact_info['valid_mask'].shape[1] != 0:
            #     collision_flag = np.clip(np.sum(contact_info['valid_mask'], axis=1),
            #                                 a_min=0, a_max=1)
            #     # no collisin points for all batched environments
            #     # print("contact information", collision_flag)
            #     collision[:, t] = torch.tensor(collision_flag, dtype=torch.float32).to(self.device)

            # if record:
            #     self.cam.render()
        
        # return collision

    def final_configuration(self, env_infos, q, record=False):
        robot = env_infos["robot"]["id"]; robot_info = env_infos["robot"]
        shelf = env_infos["shelf"]["id"]

        self.set_robot_configuration(robot, q, robot_info)
        contact_info = robot.get_contacts(shelf)

        if contact_info['valid_mask'].shape[1] != 0:
            collision_flag = np.clip(np.sum(contact_info['valid_mask'], axis=1),
                                        a_min=0, a_max=1)
            # no collisin points for all batched environments
            # print("contact information", collision_flag)
            collision = torch.tensor(collision_flag, dtype=torch.float32).to(self.device)
        else:
            collision = torch.zeros((q.shape[0], 1)).to(self.device)

        return collision

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
    def get_traj_p2(self, x, theta_0):
        ''' 
        Get the trajectory for a given decision variable x and initial configuration theta_0
        '''
        batch_size = x.shape[0]
        # theta_1 = x[:,:self.n_joints] #final configuration
        w = x[:, :] # weights of the basis function for motion between theta_0 to theta_1
        if theta_0.shape[0]!=batch_size:
            theta_0 = theta_0.view(1,-1).repeat(batch_size,1)

        theta_t = self.p2p.gen_traj(theta_0, w)#batchxtimexjoint_angle

        return theta_t
    
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
    
    """Utility functions for robot trajectory generation"""    
    def get_traj_given_q(self, x, theta_0, theta_1):
        ''' 
        Get the trajectory for a given decision variable x and initial configuration theta_0
        '''
        batch_size = x.shape[0]
        # theta_1 = x[:,:self.n_joints] #final configuration
        theta_1 = theta_1.view(1,-1).repeat(batch_size,1)
        w = x[:,:] # weights of the basis function for motion between theta_0 to theta_1
        if theta_0.shape[0]!=batch_size:
            theta_0 = theta_0.view(1,-1).repeat(batch_size,1)

        theta_t = self.p2p.gen_traj_p2p(theta_0,theta_1, w)#batchxtimexjoint_angle

        return theta_t
    
    def dist_traj(self,x_t):
        '''
            metric for length of a trajectory (to seek minimum length trajectory)
            x_t: batch x time x coordinates
        '''
        d_shortest = torch.linalg.norm(x_t[:,-1,:]-x_t[:,0,:], dim=-1)
        d_traj = torch.sum(torch.linalg.norm(x_t[:,1:,:]-x_t[:,:-1,:],dim=-1),dim=-1)
        d_straight = torch.abs(d_traj-d_shortest)/(d_shortest+1e-6)
        return d_straight


    """Utility functions for visualization"""

    def load_plane(self):
        self.scene.add_entity(gs.morphs.Plane())