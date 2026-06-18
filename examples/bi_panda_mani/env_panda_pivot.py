#
# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Teng Xue <teng.xue@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

"""
This script defines the environment for two panda arm
collaboratively pivoting a big box.
"""

import os
import time
import numpy as np
import genesis as gs
import torch

from datetime import datetime
from copy import deepcopy
from scipy.spatial.transform import Rotation as Rot


class PandaPivot():
    def __init__(self, n_envs=1, env_spacing=(3.0, 3.0), 
                 use_gui=True, gpu=True, n_joints=6,
                 p2p_motion=None):
        if gpu:
            gs.init(backend=gs.gpu, logging_level="warning")
            self.device = 'cuda'
        else:
            gs.init(backend=gs.cpu, logging_level="warning")
            self.device = 'cpu'
        self.scene = gs.Scene(show_viewer=use_gui,
            sim_options=gs.options.SimOptions(
                dt=1.0/100,
                gravity=(0, 0, -9.81),
                ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(1., 0.5, 0.8),
                camera_lookat=(0.4, -0.4, 0.0),
                camera_fov=90,),
            rigid_options=gs.options.RigidOptions(
                dt = 1.0/100,
                gravity=(0, 0, -9.81),
                enable_collision=True,
                enable_self_collision=True,
                # use_contact_island=True,
                )
                )
        
        # Useful for recording videos
        self.cam = self.scene.add_camera(
                res    = (1280, 960),
                pos    = (1., 0.5, 2.0),
                lookat = (0.4, -0.4, 0.0),
                fov    = 90,
                GUI    = False
            )
        
        self.n_envs=n_envs
        self.env_spacing=env_spacing

        self.p2p = p2p_motion
        self.n_joints = n_joints

    def load_robot(self, robot_info):
        if "xml" in robot_info.keys():
            robot_id = self.scene.add_entity(
                    gs.morphs.MJCF(file=robot_info["xml"],
                                pos=robot_info["base_pose"][:3],
                                euler=robot_info["base_pose"][3:],
                                collision=True,
                                ),
                    material=gs.materials.Rigid(gravity_compensation=1.0),
                                )
            
        robot_info["id"] = robot_id

        return robot_id
    
    def load_fixture(self, fixture_info):
        """Load fixtures like table, etc."""
        if "xml" in fixture_info.keys():
            fixture_id = self.scene.add_entity(
                    gs.morphs.MJCF(file=fixture_info["xml"],
                                pos=fixture_info["base_pose"][:3],
                                euler=fixture_info["base_pose"][3:],
                                )
                                )
        elif "urdf" in fixture_info.keys():
            fixture_id = self.scene.add_entity(
                    gs.morphs.URDF(file=fixture_info["urdf"],
                                pos=fixture_info["base_pose"][:3],
                                euler=fixture_info["base_pose"][3:],
                                fixed=True,
                                ),
                                )
        fixture_info["id"] = fixture_id
        
        return fixture_id

    def load_box(self, box_info):
        box_id = self.scene.add_entity(
                gs.morphs.URDF(file=box_info["urdf"],
                            pos=box_info["base_pose"][:3],
                            euler=box_info["base_pose"][3:],
                            collision=True,
                            fixed=False,
                            ),
                            # visualize_contact=True
                            )
        box_info["id"] = box_id
        
        return box_id

    def load_world(self, env_infos):
        """
        This function loads a table with two robot arm on the table
        based on the setup in the real world.

        It loads a large box on the table for pivoting.
        """
        # load a plane
        self.load_plane()

        # load bi-franka
        self.robot = self.load_robot(env_infos["robot"])
        self.l_eef = self.robot.get_link("panda0_grasping_target_hand")
        self.r_eef = self.robot.get_link("panda1_grasping_target_hand")
        self.robot_info = env_infos["robot"]
        # load a table
        # table = self.load_fixture(env_infos["table"])

        # load a movable large box on the table
        self.box = self.load_box(env_infos["box"])

        self.scene.build(n_envs=self.n_envs, 
                         env_spacing=self.env_spacing)
        
        self.set_robot_control_confs()

        # self.plane.set_friction(0.8)
        # self.box.set_friction(0.8)

    def set_robot_control_confs(self, ):
        # generate robots information
        n_joints = len(self.robot_info["conf"])
        dofs_idx = [self.robot.get_joint(name).dof_idx_local for name in self.robot_info["jnt_names"][:n_joints]]
        
        # set positional gains
        self.robot.set_dofs_kp(self.robot_info["kp"], dofs_idx)

        # set velocity gains
        self.robot.set_dofs_kv(self.robot_info["kv"], dofs_idx)

        # set force range for safety
        self.robot.set_dofs_force_range(lower=self.robot_info["force_range"]["lower"],
                                    upper=self.robot_info["force_range"]["upper"], 
                                    dofs_idx_local = dofs_idx)
    
    def set_robot_configuration(self, conf):
        n_joints = conf.shape[-1]
        dofs_idx = [self.robot.get_joint(name).dof_idx_local for name in self.robot_info["jnt_names"][:n_joints]]
        self.robot.control_dofs_position(conf, dofs_idx)
        self.scene.step()

    def step(self, action):
        n_joints = action.shape[-1]
        dofs_idx = [self.robot.get_joint(name).dof_idx_local for name in self.robot_info["jnt_names"][:n_joints]]
        self.robot.control_dofs_position(action, dofs_idx)
        # self.robot.set_qpos(action)
        for i in range(5):
            self.scene.step()

    def set_robot_box_config(self, action, box_pose, dt=0.1, record=False):
        n_joints = action.shape[-1]
        dofs_idx = [self.robot.get_joint(name).dof_idx_local for name in self.robot_info["jnt_names"][:n_joints]]
        for i in range(action.shape[0]):
            self.robot.set_qpos(action[i, :][None, :], dofs_idx)
            # self.step(action[i, :][None, :])
            self.box.set_pos(box_pose[i, :3][None, :])
            self.box.set_quat(gs.xyz_to_quat(box_pose[i, 3:][None, :]))
            self.scene.step()
            time.sleep(dt)
            if record:
                self.cam.render()


    def get_box_state(self):
        pos = self.box.get_pos()
        xyz = gs.quat_to_xyz(self.box.get_quat())
        return torch.cat((pos, xyz), dim=-1), torch.cat((pos[:, :2], xyz[:, 2:]), dim=-1)

    def forward_configuration_full(self, env_infos, trajectory, record=False):
        self.reset(env_infos)
        # add trajectory execution with step function
        box_contacts = torch.zeros((trajectory.shape[0], trajectory.shape[1], 2)).to(self.device)
        box_orns = torch.zeros((trajectory.shape[0], trajectory.shape[1], 3)).to(self.device)
        box_poss = torch.zeros((trajectory.shape[0], trajectory.shape[1], 3)).to(self.device)

        for t in range(trajectory.shape[1]):
            self.step(trajectory[:, t, :])
            # contact_info = self.robot.get_contacts(self.box)

            # if contact_info['valid_mask'].shape[1] != 0:
            #     collision_flag = np.clip(np.sum(contact_info['valid_mask'], axis=1),
            #                                 a_min=0, a_max=1)
            #     # no collisin points for all batched environments
            #     # print("contact information", collision_flag)
            #     box_contacts[:, t] = torch.tensor(collision_flag, dtype=torch.float32).to(self.device)
            #     # print("time cost2 for contact detecting: ", time.time() - init_t)
            
            # get link contact forces to identify if the links are in contact
            # in shape of (n_envs, n_links=22, 3)
            n_links = len(self.robot._links)
            left_link_idx = [n for n in range(n_links) if n % 2 != 0]
            right_link_idx = [n for n in range(n_links) if n % 2 == 0]

            link_contact_force = self.robot.get_links_net_contact_force()
            link_contact_force_norm = torch.norm(link_contact_force, dim=-1)
            box_contacts[:, t, 0] = (link_contact_force_norm[:, left_link_idx] > 0.1).sum(dim=-1)
            box_contacts[:, t, 1] = (link_contact_force_norm[:, right_link_idx] > 0.1).sum(dim=-1)

            # in degrees
            box_orn = gs.quat_to_xyz(self.box.get_quat())
            box_pos = self.box.get_pos()

            box_orns[:, t] = torch.tensor(box_orn, dtype=torch.float32).to(self.device)
            box_poss[:, t] = torch.tensor(box_pos, dtype=torch.float32).to(self.device)

            if record:
                self.cam.render()

        return box_contacts, box_orns, box_poss




    def forward_configuration(self, env_infos, trajectory, record=False):
        self.reset(env_infos)
        # add trajectory execution with step function
        box_contacts = torch.zeros((trajectory.shape[0], trajectory.shape[1])).to(self.device)
        box_orns = torch.zeros((trajectory.shape[0], trajectory.shape[1])).to(self.device)

        for t in range(trajectory.shape[1]):
            self.step(trajectory[:, t, :])
            # contact_info = self.robot.get_contacts(self.box)

            # if contact_info['valid_mask'].shape[1] != 0:
            #     collision_flag = np.clip(np.sum(contact_info['valid_mask'], axis=1),
            #                                 a_min=0, a_max=1)
            #     # no collisin points for all batched environments
            #     # print("contact information", collision_flag)
            #     box_contacts[:, t] = torch.tensor(collision_flag, dtype=torch.float32).to(self.device)
            #     # print("time cost2 for contact detecting: ", time.time() - init_t)
            
            # in degrees
            box_orn = gs.quat_to_xyz(self.box.get_quat())[:, 2]
            box_orns[:, t] = torch.tensor(box_orn, dtype=torch.float32).to(self.device)

            if record:
                self.cam.render()

        return box_contacts, box_orns

    def forward_execution_2(self, env_infos, trajectory, cost_func, sols, record=False):

        """
        real-world execution
        """

        self.reset(env_infos)
        # add trajectory execution with step function
        box_orns = torch.zeros((trajectory.shape[0], trajectory.shape[1], 3)).to(self.device)
        box_poss = torch.zeros((trajectory.shape[0], trajectory.shape[1], 3)).to(self.device)

        box_contacts = torch.zeros((trajectory.shape[0], trajectory.shape[1], 2)).to(self.device)

        real_world_traj = torch.zeros((trajectory.shape[0], trajectory.shape[1], 14)).to(self.device)
        box_traj = torch.zeros((trajectory.shape[0], trajectory.shape[1], 6)).to(self.device)
        for t in range(trajectory.shape[1]):
            self.step(trajectory[:, t, :])

            n_links = len(self.robot._links)
            left_link_idx = [n for n in range(n_links) if n % 2 != 0]
            right_link_idx = [n for n in range(n_links) if n % 2 == 0]

            link_contact_force = self.robot.get_links_net_contact_force()
            link_contact_force_norm = torch.norm(link_contact_force, dim=-1)
            box_contacts[:, t, 0] = (link_contact_force_norm[:, left_link_idx] > 0.1).sum(dim=-1)
            box_contacts[:, t, 1] = (link_contact_force_norm[:, right_link_idx] > 0.1).sum(dim=-1)



            # in degrees
            box_orn = gs.quat_to_xyz(self.box.get_quat())
            box_orns[:, t] = torch.tensor(box_orn, dtype=torch.float32).to(self.device)
            box_pos = self.box.get_pos()
            box_poss[:, t] = torch.tensor(box_pos, dtype=torch.float32).to(self.device)
            box_traj[:, t, :3] = torch.tensor(box_pos, dtype=torch.float32).to(self.device)
            box_traj[:, t, 3:] = torch.tensor(box_orn, dtype=torch.float32).to(self.device)
            cur_q = self.robot.get_qpos()
            real_world_traj[:, t, :] = torch.tensor(cur_q, dtype=torch.float32).to(self.device)

            if record:
                self.cam.render()

        # env_infos["box"]["base_pose"][:3] = self.box.get_pos()
        # env_infos["box"]["base_pose"][3:] = gs.quat_to_xyz(self.box.get_quat())

        #find the box closest to the target
        # target_angle = env_infos["goal"]["target_orn"]
        # target_pos = env_infos["goal"]["target_pos"]

        # idx = torch.argmin(torch.abs(box_orn[:, -1] - target_angle), dim=-1)
        costs = cost_func(box_poss, box_orns, trajectory, box_contacts)
        idx = torch.argmin(costs, dim=-1)

        # best_theta_t = trajectory[idx, :, :]
        best_theta_t = real_world_traj[idx, :, :]

        # env_infos["robot"]["conf"] = list(self.robot.get_qpos()[idx, :])
        env_infos["box"]["base_pose"][:3] = list(self.box.get_pos()[idx, :].cpu().numpy())
        env_infos["box"]["base_pose"][3:] = list(gs.quat_to_xyz(self.box.get_quat())[idx, :].cpu().numpy())
        best_sol = sols[idx, :]
        return env_infos, box_traj[idx, :, :], best_sol, 

    def sim_forward_execution(self, env_infos, trajectory, cost_func, record=False):

        """
        real-world execution
        """

        self.reset(env_infos)
        # add trajectory execution with step function
        box_orns = torch.zeros((trajectory.shape[0], trajectory.shape[1], 3)).to(self.device)
        box_poss = torch.zeros((trajectory.shape[0], trajectory.shape[1], 3)).to(self.device)

        box_contacts = torch.zeros((trajectory.shape[0], trajectory.shape[1], 2)).to(self.device)

        real_world_traj = torch.zeros((trajectory.shape[0], trajectory.shape[1], 14)).to(self.device)
        box_traj = torch.zeros((trajectory.shape[0], trajectory.shape[1], 6)).to(self.device)
        for t in range(trajectory.shape[1]):
            self.step(trajectory[:, t, :])

            n_links = len(self.robot._links)
            left_link_idx = [n for n in range(n_links) if n % 2 != 0]
            right_link_idx = [n for n in range(n_links) if n % 2 == 0]

            link_contact_force = self.robot.get_links_net_contact_force()
            link_contact_force_norm = torch.norm(link_contact_force, dim=-1)
            box_contacts[:, t, 0] = (link_contact_force_norm[:, left_link_idx] > 0.1).sum(dim=-1)
            box_contacts[:, t, 1] = (link_contact_force_norm[:, right_link_idx] > 0.1).sum(dim=-1)



            # in degrees
            box_orn = gs.quat_to_xyz(self.box.get_quat())
            box_orns[:, t] = torch.tensor(box_orn, dtype=torch.float32).to(self.device)
            box_pos = self.box.get_pos()
            box_poss[:, t] = torch.tensor(box_pos, dtype=torch.float32).to(self.device)
            box_traj[:, t, :3] = torch.tensor(box_pos, dtype=torch.float32).to(self.device)
            box_traj[:, t, 3:] = torch.tensor(box_orn, dtype=torch.float32).to(self.device)
            cur_q = self.robot.get_qpos()
            real_world_traj[:, t, :] = torch.tensor(cur_q, dtype=torch.float32).to(self.device)

            if record:
                self.cam.render()

        # env_infos["box"]["base_pose"][:3] = self.box.get_pos()
        # env_infos["box"]["base_pose"][3:] = gs.quat_to_xyz(self.box.get_quat())

        #find the box closest to the target
        # target_angle = env_infos["goal"]["target_orn"]
        # target_pos = env_infos["goal"]["target_pos"]

        # idx = torch.argmin(torch.abs(box_orn[:, -1] - target_angle), dim=-1)
        costs = cost_func(box_poss, box_orns, trajectory, box_contacts)
        idx = torch.argmin(costs, dim=-1)

        # best_theta_t = trajectory[idx, :, :]
        best_theta_t = real_world_traj[idx, :, :]

        # # env_infos["robot"]["conf"] = list(self.robot.get_qpos()[idx, :])
        # env_infos["box"]["base_pose"][:3] = list(self.box.get_pos()[idx, :].cpu().numpy())
        # env_infos["box"]["base_pose"][3:] = list(gs.quat_to_xyz(self.box.get_quat())[idx, :].cpu().numpy())

        return env_infos, box_traj[idx, :, :], best_theta_t

    
    def forward_execution(self, env_infos, trajectory, cost_func, record=False):

        """
        real-world execution
        """

        self.reset(env_infos)
        # add trajectory execution with step function
        box_orns = torch.zeros((trajectory.shape[0], trajectory.shape[1], 3)).to(self.device)
        box_poss = torch.zeros((trajectory.shape[0], trajectory.shape[1], 3)).to(self.device)

        box_contacts = torch.zeros((trajectory.shape[0], trajectory.shape[1], 2)).to(self.device)

        real_world_traj = torch.zeros((trajectory.shape[0], trajectory.shape[1], 14)).to(self.device)
        box_traj = torch.zeros((trajectory.shape[0], trajectory.shape[1], 6)).to(self.device)
        for t in range(trajectory.shape[1]):
            self.step(trajectory[:, t, :])

            n_links = len(self.robot._links)
            left_link_idx = [n for n in range(n_links) if n % 2 != 0]
            right_link_idx = [n for n in range(n_links) if n % 2 == 0]

            link_contact_force = self.robot.get_links_net_contact_force()
            link_contact_force_norm = torch.norm(link_contact_force, dim=-1)
            box_contacts[:, t, 0] = (link_contact_force_norm[:, left_link_idx] > 0.1).sum(dim=-1)
            box_contacts[:, t, 1] = (link_contact_force_norm[:, right_link_idx] > 0.1).sum(dim=-1)



            # in degrees
            box_orn = gs.quat_to_xyz(self.box.get_quat())
            box_orns[:, t] = torch.tensor(box_orn, dtype=torch.float32).to(self.device)
            box_pos = self.box.get_pos()
            box_poss[:, t] = torch.tensor(box_pos, dtype=torch.float32).to(self.device)
            box_traj[:, t, :3] = torch.tensor(box_pos, dtype=torch.float32).to(self.device)
            box_traj[:, t, 3:] = torch.tensor(box_orn, dtype=torch.float32).to(self.device)
            cur_q = self.robot.get_qpos()
            real_world_traj[:, t, :] = torch.tensor(cur_q, dtype=torch.float32).to(self.device)

            if record:
                self.cam.render()

        # env_infos["box"]["base_pose"][:3] = self.box.get_pos()
        # env_infos["box"]["base_pose"][3:] = gs.quat_to_xyz(self.box.get_quat())

        #find the box closest to the target
        # target_angle = env_infos["goal"]["target_orn"]
        # target_pos = env_infos["goal"]["target_pos"]

        # idx = torch.argmin(torch.abs(box_orn[:, -1] - target_angle), dim=-1)
        costs = cost_func(box_poss, box_orns, trajectory, box_contacts)
        idx = torch.argmin(costs, dim=-1)

        # best_theta_t = trajectory[idx, :, :]
        best_theta_t = real_world_traj[idx, :, :]

        # env_infos["robot"]["conf"] = list(self.robot.get_qpos()[idx, :])
        env_infos["box"]["base_pose"][:3] = list(self.box.get_pos()[idx, :].cpu().numpy())
        env_infos["box"]["base_pose"][3:] = list(gs.quat_to_xyz(self.box.get_quat())[idx, :].cpu().numpy())

        return env_infos, box_traj[idx, :, :], best_theta_t

    def reset(self, env_infos):
        # reset the box pose
        # reset the robot pose
        conf = torch.tile(torch.tensor(env_infos["robot"]["conf"]), (self.n_envs, 1))
        for i in range(2):
            self.box.set_pos(torch.tile(torch.tensor(env_infos["box"]["base_pose"][:3]), (self.n_envs, 1)))
            self.box.set_quat(torch.tile(gs.xyz_to_quat(torch.tensor(env_infos["box"]["base_pose"][3:])), 
                                        (self.n_envs, 1)))
            self.robot.set_qpos(conf)
            self.scene.step()

    """Utility functions for robot trajectory generation"""    
    def get_traj_p2(self, x, theta_0):
        ''' 
        Get the trajectory for a given decision variable x and initial configuration theta_0
        '''
        batch_size = x.shape[0]
        # theta_1 = x[:,:self.n_joints] #final configuration
        w = x[:, :] # weights of the basis function for motion between theta_0 to theta_1
        if theta_0.shape[0]!= batch_size:
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
    

    """Utility functions for visualization"""
    def load_plane(self):
        self.plane = self.scene.add_entity(gs.morphs.Plane())

    