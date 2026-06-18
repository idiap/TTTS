#
# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Teng Xue <teng.xue@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

'''
    Copyright (c) 2022 Idiap Research Institute, http://www.idiap.ch/
    Written by Suhan Shetty <suhan.shetty@idiap.ch>,
   
    This file is part of TTGO.

    TTGO is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License version 3 as
    published by the Free Software Foundation.

    TTGO is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with TTGO. If not, see <http://www.gnu.org/licenses/>.
'''



import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
import seaborn as sns

import math
np.math = math


class PlanarManipulator:
    def __init__(self, n_joints=2, link_lengths=[], max_theta=torch.pi/1.1, n_kp=3, device="cpu"):
        ''' 
            n_joints: number of joints in the planar manipulator
            max_theta: max joint angle (same for al joints)
            link_lengths: a list containing length of each link
            n_kp: number of key-points on each link (for collision check)
        '''
        self.device = device

        self.n_joints = n_joints
        if link_lengths is None:
            self.link_lengths = torch.tensor([1./n_joints]*n_joints).to(self.device)
        else:
            self.link_lengths = link_lengths.to(device) 
        assert n_joints== link_lengths.shape[0], 'The length of the list containing link_lengths should match n_joints'
        
        self.max_config = torch.tensor([max_theta]*n_joints).to(self.device)
        self.min_config = -1*self.max_config
        self.theta_max = self.max_config
        self.theta_min = self.min_config

        self.n_kp = n_kp
        assert self.n_kp>=2, 'number of key points should be at least two'
        self.key_points = torch.empty(self.n_joints,self.n_kp).to(device)
        for i in range(n_joints):
            self.key_points[i] = torch.arange(0,self.n_kp)/(self.n_kp-1)

    # forward kinematics
    def forward_kin(self, q):
        ''' Given a batch of joint angles find the position of all the key-points and the end-effector  '''
        batch_size = q.shape[0]
        q = torch.clip(q,self.min_config, self.max_config)
        q_cumsum = torch.zeros(batch_size,self.n_joints).to(self.device)
        for joint in range(self.n_joints):
            q_cumsum[:,joint] = torch.sum(q[:,:joint+1],dim=1)

        cq = torch.cos(q_cumsum).view(batch_size,-1,1)
        sq = torch.sin(q_cumsum).view(batch_size,-1,1)
        cq_sq = torch.cat((cq,sq),dim=2)

        joint_loc = torch.zeros((batch_size, self.n_joints+1, 2)).to(self.device)
        key_loc = torch.empty((batch_size, self.n_joints,self.n_kp,2)).to(self.device)
        for i in range(self.n_joints):
            joint_loc[:,i+1,:] = joint_loc[:,i,:]+self.link_lengths[i]*cq_sq[:,i,:]
            key_loc[:,i,:,:] = joint_loc[:,i,:][:,None,:] + (joint_loc[:,i+1,:]-joint_loc[:,i,:])[:,None,:]*self.key_points[i].reshape(1,-1,1)
    
        end_loc = joint_loc[:,-1,:]
        # find the orientation of end-effector in range (0,2*pi)
        theta_orient = torch.fmod(q_cumsum[:,-1],2*torch.pi)
        theta_orient[theta_orient<0] = 2*torch.pi+theta_orient[theta_orient<0]
        
        # output shape: batch_size x (n_joints) x n_kp x 2, batch_size x 2
        return key_loc, joint_loc, end_loc, theta_orient 
    
    def plot_chain_batch_old(self, joint_loc_list, link_lengths, x_obst=[], r_obst=[], x_target=[], rect_patch=[], 
        batch=False, skip_frame=10, title=None, save_as=None, figsize=8,
        color_intensity=0.9, motion=False, alpha=0.5, contrast=0.4, idx_highlight=[], lw=7, task='ik'):

        fig, axs = plt.subplots(2, 2, figsize=(20, 20))

        # Set black background for all subplots
        for ax in axs.flat:
            ax.set_facecolor('k')
            ax.set_aspect('equal')

        # Plot each task's trajectories
        for idx, ax in enumerate(axs.flat):
            joint_loc = joint_loc_list[idx]
            # Color gradient based on time
            t_steps = np.linspace(0, 1, joint_loc.shape[0])
            colors = [(1-t, t, 0) for t in t_steps]  # Gradient from red to green
            


            for i,x in enumerate(x_obst):
                circ = plt.Circle(x,r_obst[i],color='orange',alpha=0.8)
                ax.add_patch(circ)


            else:

                T = joint_loc.shape[0]

                # ax.legend(["target","obstacle"])
                idx = np.arange(0,int(T), skip_frame)
                for count,i in enumerate(idx):
                    # color_ = np.where(motion, 1-k_[count], contrast)
                    x = joint_loc[i,:,0]
                    y = joint_loc[i,:,1]
                    ax.plot(x, y, 'o-',zorder=0.9,marker='o',color=colors[i],lw=lw,mfc='w',
                                solid_capstyle='round', alpha=alpha )
                    ax.plot(joint_loc[i,:-1,0],joint_loc[i,:-1,1],'oy', markersize=3,alpha=alpha)
                    ax.plot(joint_loc[i,-1,0],joint_loc[i,-1,1],'oy', markersize=3,alpha=alpha)


                # for count,i in enumerate(idx_highlight):
                #     color_ = [0.1]*3
                #     x = joint_loc[i,:,0]
                #     y = joint_loc[i,:,1]
                #     plt.plot(x, y, 'o-',zorder=0.9,marker='o',color='k',lw=lw,mfc='w',
                #                 solid_capstyle='round', alpha=0.5)
                for i, x_ in enumerate(x_target):
                    ax.scatter(x_[0],x_[1], color='white', marker='x', s=150, zorder=20, linewidths=3)
                

                ax.plot(0,0,color='y',marker='o', markersize=15)
                ax.grid(False)

        if not title is None:
            plt.title(title)
        if not save_as is None:
            fig.savefig('./images/'+save_as+".jpeg",bbox_inches='tight', pad_inches=0.01, dpi=300)

        return plt
    


    def plot_chain_batch(self, joint_loc_list, link_lengths, x_obst=[], r_obst=[], x_target=[], rect_patch=[], 
            batch=False, skip_frame=10, title=None, save_as=None, figsize=8,
            color_intensity=0.9, motion=False, alpha=0.5, contrast=0.4, idx_highlight=[], lw=7, task='ik', animation=False):

        fig, axs = plt.subplots(2, 2, figsize=(figsize, figsize))

        # Ensure all subplots have the same size and layout
        for ax in axs.flat:
            ax.set_facecolor('k')
            ax.set_aspect('equal')
            ax.set_xlim(-1.1 * np.sum(link_lengths), 1.1 * np.sum(link_lengths))
            ax.set_ylim(-1.1 * np.sum(link_lengths), 1.1 * np.sum(link_lengths))

        if animation:
            # Initialize lines for all joint_loc instances in joint_loc_list
            lines_list = []
            colors = plt.cm.rainbow(np.linspace(0, 1, len(joint_loc_list)))  # Generate distinct colors for each joint_loc

            for joint_loc, color in zip(joint_loc_list, colors):
                lines = []
                for _ in range(joint_loc.shape[1] - 1):
                    line, = ax.plot([], [], zorder=0.9, marker='o', color=color, lw=lw, mfc='w',
                                    solid_capstyle='round', alpha=alpha)
                    lines.append(line)
                lines_list.append(lines)

            for i, x in enumerate(x_obst):
                circ = plt.Circle(x, r_obst[i], color='orange', alpha=0.8)
                ax.add_patch(circ)

            def init():
                """Initialize the animation."""
                for lines in lines_list:
                    for line in lines:
                        line.set_data([], [])
                return [line for lines in lines_list for line in lines]

            def update(frame):
                """Update the animation for a given frame."""
                for lines, joint_loc in zip(lines_list, joint_loc_list):
                    for j in range(joint_loc.shape[1] - 1):
                        x = [joint_loc[frame, j, 0], joint_loc[frame, j + 1, 0]]
                        y = [joint_loc[frame, j, 1], joint_loc[frame, j + 1, 1]]
                        lines[j].set_data(x, y)
                return [line for lines in lines_list for line in lines]

            # Create the animation
            anim = FuncAnimation(fig, update, frames=range(0, joint_loc_list[0].shape[0], skip_frame),
                                init_func=init, blit=True, repeat=False)



            if save_as:
                writer = FFMpegWriter(fps=10, metadata={'artist': 'Matplotlib'}, bitrate=1800)
                anim.save(f'./images/{save_as}.mp4', writer=writer, dpi=300)

            return anim

        else:
            # Static image
            for idx, ax in enumerate(axs.flat):
                joint_loc = joint_loc_list[idx]
                t_steps = np.linspace(0, 1, joint_loc.shape[0])
                colors = [(1 - t, t, 0) for t in t_steps]  # Gradient from red to green

                for i, x in enumerate(x_obst):
                    circ = plt.Circle(x, r_obst[i], color='orange', alpha=0.8)
                    ax.add_patch(circ)

                T = joint_loc.shape[0]
                idx = np.arange(0, int(T), skip_frame)
                for count, i in enumerate(idx):
                    x = joint_loc[i, :, 0]
                    y = joint_loc[i, :, 1]
                    ax.plot(x, y, 'o-', zorder=0.9, marker='o', color=colors[i], lw=lw, mfc='w',
                            solid_capstyle='round', alpha=alpha)
                    ax.plot(joint_loc[i, :-1, 0], joint_loc[i, :-1, 1], 'oy', markersize=3, alpha=alpha)
                    ax.plot(joint_loc[i, -1, 0], joint_loc[i, -1, 1], 'oy', markersize=3, alpha=alpha)

                for i, x_ in enumerate(x_target):
                    ax.scatter(x_[0], x_[1], color='white', marker='x', s=150, zorder=20, linewidths=3)

                ax.plot(0, 0, color='y', marker='o', markersize=15)
                ax.grid(False)

            if not title is None:
                fig.suptitle(title)
            if save_as:
                fig.savefig(f'./images/{save_as}.jpeg', bbox_inches='tight', pad_inches=0.01, dpi=300)

            return plt

    def plot_chain_animation(self, joint_loc_list, link_lengths, x_obst=[], r_obst=[], x_target=[], rect_patch=[], 
            batch=False, skip_frame=10, title=None, save_as=None, figsize=8,
            color_intensity=0.9, motion=False, alpha=0.5, contrast=0.4, idx_highlight=[], lw=7, task='ik', animation=False):

        fig, ax = plt.subplots(figsize=(figsize, figsize))
        xmax_ = 1.1 * np.sum(link_lengths)
        ax.set_xlim(-xmax_, xmax_)
        ax.set_ylim(-xmax_, xmax_)
        ax.set_aspect('equal')
        ax.set_facecolor('k')

        for i, x in enumerate(x_obst):
            circ = plt.Circle(x, r_obst[i], color='orange', alpha=0.8)
            ax.add_patch(circ)

        for i, x_ in enumerate(x_target):
            ax.scatter(x_[0], x_[1], color='white', marker='x', s=150, zorder=20, linewidths=3)

        plt.plot(0,0,color='y',marker='o', markersize=15)
        t_steps = np.linspace(0, 1, joint_loc_list.shape[1])

        # colors = list(colors.TABLEAU_COLORS.values())
        colors = ['g', 'r', 'brown', 'gray', 'y', 'k', 'w', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

        if animation:
            # Initialize line and marker objects for each joint_loc in joint_loc_list
            lines_list = []
            for loc_idx in range(len(joint_loc_list)):
                joint_loc = joint_loc_list[loc_idx]
                lines = []
                for _ in range(joint_loc.shape[1] - 1):
                    color = colors[loc_idx % len(colors)]  # Assign color based on the loc_idx
                    line, = ax.plot([], [], zorder=0.9, marker='o', color=color, lw=lw, mfc='w',
                                    solid_capstyle='round', alpha=alpha)
                    lines.append(line)
                lines_list.append(lines)

            def init():
                """Initialize the animation."""
                for lines in lines_list:
                    for line in lines:
                        line.set_data([], [])
                return [line for lines in lines_list for line in lines]

            def update(frame):
                """Update the animation for a given frame."""
                for loc_idx, joint_loc in enumerate(joint_loc_list):
                    lines = lines_list[loc_idx]
                    for j in range(joint_loc.shape[1] - 1):
                        x = [joint_loc[frame, j, 0], joint_loc[frame, j + 1, 0]]
                        y = [joint_loc[frame, j, 1], joint_loc[frame, j + 1, 1]]
                        lines[j].set_data(x, y)
                return [line for lines in lines_list for line in lines]

            # Create the animation
            anim = FuncAnimation(fig, update, frames=range(0, joint_loc_list[0].shape[0], skip_frame),
                                init_func=init, blit=True, repeat=False)

            if save_as:
                writer = FFMpegWriter(fps=10, metadata={'artist': 'Matplotlib'}, bitrate=1800)
                anim.save(f'{save_as}.mp4', writer=writer, dpi=300)

            return anim
    


    # Modified plot_chain function to generate animation or static image
    def plot_chain_single_old(self, joint_loc_list, link_lengths, x_obst=[], r_obst=[], x_target=[], rect_patch=[], 
            batch=False, skip_frame=10, title=None, save_as=None, figsize=8,
            color_intensity=0.9, motion=False, alpha=0.5, contrast=0.4, idx_highlight=[], lw=7, task='ik', animation=False):

        fig, ax = plt.subplots(figsize=(figsize, figsize))
        xmax_ = 1.1 * np.sum(link_lengths)
        ax.set_xlim(-xmax_, xmax_)
        ax.set_ylim(-xmax_, xmax_)
        ax.set_aspect('equal')
        ax.set_facecolor('k')

        joint_loc = joint_loc_list[0]

        for i, x in enumerate(x_obst):
            circ = plt.Circle(x, r_obst[i], color='orange', alpha=0.8)
            ax.add_patch(circ)

        for i, x_ in enumerate(x_target):
            ax.scatter(x_[0], x_[1], color='white', marker='x', s=150, zorder=20, linewidths=3)

        plt.plot(0,0,color='y',marker='o', markersize=15)
        t_steps = np.linspace(0, 1, joint_loc.shape[0])
        colors = [(1-t, t, 0) for t in t_steps]  # Gradient from red to green
        # colors = list(colors.TABLEAU_COLORS.values())
        # colors = ['g', 'r', 'brown', 'gray', 'y', 'k', 'w', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']


        # Static image
        for i in range(0, joint_loc.shape[0], skip_frame):
            for j in range(joint_loc.shape[1] - 1):
                x = [joint_loc[i, j, 0], joint_loc[i, j + 1, 0]]
                y = [joint_loc[i, j, 1], joint_loc[i, j + 1, 1]]
                # ax.plot(x, y, 'o-', lw=lw, color='gray', mfc='w', solid_capstyle='round', alpha=alpha)
                plt.plot(x, y, 'o-',zorder=0.9,marker='o',color=colors[i],lw=lw,mfc='w',

                        solid_capstyle='round', alpha=alpha )

                plt.plot(joint_loc[i,:-1,0],joint_loc[i,:-1,1],'oy', markersize=3,alpha=alpha)

                plt.plot(joint_loc[i,-1,0],joint_loc[i,-1,1],'oy', markersize=3,alpha=alpha)

        if save_as:
            fig.savefig(f'./images/{save_as}.png', bbox_inches='tight', pad_inches=0.01, dpi=300)

        return plt
    
    def plot_chain_single(self, joint_loc_list, link_lengths, x_obst=[], r_obst=[], x_target=[], rect_patch=[], 
        batch=False, skip_frame=10, title=None, save_as=None, figsize=8,
        color_intensity=0.9, motion=False, alpha=0.5, contrast=0.4, idx_highlight=[], lw=7, task='ik', animation=False):

        fig, ax = plt.subplots(figsize=(figsize, figsize))
        xmax_ = 1.1 * np.sum(link_lengths)
        ax.set_xlim(-xmax_, xmax_)
        ax.set_ylim(-xmax_, xmax_)
        ax.set_aspect('equal')
        ax.set_facecolor('white')  # Set white background
        ax.axis('off')


        joint_loc = joint_loc_list[0]

        # Plot circular obstacles
        for i, x in enumerate(x_obst):
            circ = plt.Circle(x, r_obst[i], color='darkorange', alpha=0.8)  # Use darker orange for contrast
            ax.add_patch(circ)

        # Plot target positions
        for i, x_ in enumerate(x_target):
            ax.scatter(x_[0], x_[1], color='red', marker='x', s=150, zorder=20, linewidths=3)  # Red crosses for targets

        # Plot base joint at origin
        plt.plot(0, 0, color='black', marker='o', markersize=15)

        # Generate color gradient from red to green
        t_steps = np.linspace(0, 1, joint_loc.shape[0])
        colors = [(1 - t * color_intensity, t * color_intensity, 0) for t in t_steps]  # RGB tuples

        # Static image plot
        for i in range(0, joint_loc.shape[0], skip_frame):
            for j in range(joint_loc.shape[1] - 1):
                x = [joint_loc[i, j, 0], joint_loc[i, j + 1, 0]]
                y = [joint_loc[i, j, 1], joint_loc[i, j + 1, 1]]
                plt.plot(x, y, 'o-', zorder=0.9, marker='o', color=colors[i], lw=lw, mfc=None,
                        solid_capstyle='round', alpha=alpha)  # Keep original line width
                

            plt.plot(joint_loc[i, :-1, 0], joint_loc[i, :-1, 1], 'o', color='black', markersize=7, alpha=alpha)
            plt.plot(joint_loc[i, -1, 0], joint_loc[i, -1, 1], 'o', color='black', markersize=7, alpha=alpha)

            # Plot joints
            # plt.plot(joint_loc[i, :-1, 0], joint_loc[i, :-1, 1], 'o', 
            #         markerfacecolor='none', markeredgecolor='black', markersize=7, alpha=alpha, markeredgewidth=2)

            # plt.plot(joint_loc[i, -1, 0], joint_loc[i, -1, 1], 'o', 
            #         markerfacecolor='none', markeredgecolor='black', markersize=7, alpha=alpha, markeredgewidth=2)

        # Save figure if filename provided
        if save_as:
            fig.savefig(f'./images/{save_as}.png', bbox_inches='tight', pad_inches=0.01, dpi=300)

        return plt


    
    # Modified plot_chain function to generate animation or static image
    def plot_chain_ik(self, joint_loc, link_lengths, x_obst=[], r_obst=[], x_target=[], rect_patch=[], 
            batch=False, skip_frame=10, title=None, save_as=None, figsize=8,
            color_intensity=0.9, motion=False, alpha=0.5, contrast=0.4, idx_highlight=[], lw=7, task='ik', animation=False):

        fig, ax = plt.subplots(figsize=(figsize, figsize))
        xmax_ = 1.1 * np.sum(link_lengths)
        ax.set_xlim(-xmax_, xmax_)
        ax.set_ylim(-xmax_, xmax_)
        ax.set_aspect('equal')
        ax.set_facecolor('k')

        for i, x in enumerate(x_obst):
            circ = plt.Circle(x, r_obst[i], color='orange', alpha=0.8)
            ax.add_patch(circ)

        for i, x_ in enumerate(x_target):
            ax.scatter(x_[0], x_[1], color='white', marker='x', s=150, zorder=20, linewidths=3)

        plt.plot(0,0,color='y',marker='o', markersize=15)
        t_steps = np.linspace(0, 1, joint_loc.shape[0])

        colors = [(1-t, t, 0) for t in t_steps]  # Gradient from red to green

        
        # Static image
        for i in range(0, joint_loc.shape[0], skip_frame):
            for j in range(joint_loc.shape[1] - 1):
                x = [joint_loc[i, j, 0], joint_loc[i, j + 1, 0]]
                y = [joint_loc[i, j, 1], joint_loc[i, j + 1, 1]]
                # ax.plot(x, y, 'o-', lw=lw, color='gray', mfc='w', solid_capstyle='round', alpha=alpha)
                plt.plot(x, y, 'o-',zorder=0.9,marker='o',color=colors[i],lw=lw,mfc='w',

                        solid_capstyle='round', alpha=alpha )

                plt.plot(joint_loc[i,:-1,0],joint_loc[i,:-1,1],'oy', markersize=3,alpha=alpha)

                plt.plot(joint_loc[i,-1,0],joint_loc[i,-1,1],'oy', markersize=3,alpha=alpha)

        if save_as:
            fig.savefig(f'./images/{save_as}.png', bbox_inches='tight', pad_inches=0.01, dpi=300)

        return plt



    # Modified plot_chain function to generate animation or static image
    def plot_chain(self, joint_loc, link_lengths, x_obst=[], r_obst=[], x_target=[], rect_patch=[], 
            batch=False, skip_frame=10, title=None, save_as=None, figsize=8,
            color_intensity=0.9, motion=False, alpha=0.5, contrast=0.4, idx_highlight=[], lw=7, task='ik', animation=False):

        fig, ax = plt.subplots(figsize=(figsize, figsize))
        xmax_ = 1.1 * np.sum(link_lengths)
        ax.set_xlim(-xmax_, xmax_)
        ax.set_ylim(-xmax_, xmax_)
        ax.set_aspect('equal')
        ax.set_facecolor('k')

        for i, x in enumerate(x_obst):
            circ = plt.Circle(x, r_obst[i], color='orange', alpha=0.8)
            ax.add_patch(circ)

        for i, x_ in enumerate(x_target):
            ax.scatter(x_[0], x_[1], color='white', marker='x', s=150, zorder=20, linewidths=3)

        plt.plot(0,0,color='y',marker='o', markersize=15)
        t_steps = np.linspace(0, 1, joint_loc.shape[0])

        colors = [(1-t, t, 0) for t in t_steps]  # Gradient from red to green

        if animation:
            # Initialize line and marker objects
            lines = []
            for _ in range(joint_loc.shape[1] - 1):
                line, = ax.plot([], [], zorder=0.9,marker='o',color=colors[_],lw=lw,mfc='w',

                            solid_capstyle='round', alpha=alpha)
                lines.append(line)

            def init():
                """Initialize the animation."""
                for line in lines:
                    line.set_data([], [])
                return lines

            def update(frame):
                """Update the animation for a given frame."""
                for j in range(joint_loc.shape[1] - 1):
                    x = [joint_loc[frame, j, 0], joint_loc[frame, j + 1, 0]]
                    y = [joint_loc[frame, j, 1], joint_loc[frame, j + 1, 1]]
                    lines[j].set_data(x, y)
                return lines

            # Create the animation
            anim = FuncAnimation(fig, update, frames=range(0, joint_loc.shape[0], skip_frame),
                                init_func=init, blit=True, repeat=False)

            if save_as:
                writer = FFMpegWriter(fps=10, metadata={'artist': 'Matplotlib'}, bitrate=1800)
                anim.save(f'{save_as}.mp4', writer=writer, dpi=300)

            return anim

        else:
            # Static image
            for i in range(0, joint_loc.shape[0], skip_frame):
                for j in range(joint_loc.shape[1] - 1):
                    x = [joint_loc[i, j, 0], joint_loc[i, j + 1, 0]]
                    y = [joint_loc[i, j, 1], joint_loc[i, j + 1, 1]]
                    # ax.plot(x, y, 'o-', lw=lw, color='gray', mfc='w', solid_capstyle='round', alpha=alpha)
                    plt.plot(x, y, 'o-',zorder=0.9,marker='o',color=colors[i],lw=lw,mfc='w',

                            solid_capstyle='round', alpha=alpha )

                    plt.plot(joint_loc[i,:-1,0],joint_loc[i,:-1,1],'oy', markersize=3,alpha=alpha)

                    plt.plot(joint_loc[i,-1,0],joint_loc[i,-1,1],'oy', markersize=3,alpha=alpha)

            if save_as:
                fig.savefig(f'./images/{save_as}.png', bbox_inches='tight', pad_inches=0.01, dpi=300)

            return plt


    def plot_chain_old(self, joint_loc, link_lengths, x_obst=[], r_obst=[], x_target=[], rect_patch=[], 
        batch=False, skip_frame=10, title=None, save_as=None, figsize=8,
        color_intensity=0.9, motion=False, alpha=0.5, contrast=0.4, idx_highlight=[], lw=7, task='ik'):

        fig = plt.figure(edgecolor=[0.1,0.1,0.1], figsize=(figsize,figsize))
        

        # fig.set_size_inches(figsize, figsize)


        # fig.patch.set_facecolor('white')
        # fig.patch.set_alpha(0.9)
        xmax_ = 1.1*np.sum(link_lengths)
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                            xlim=(-xmax_, xmax_), ylim=(-xmax_, xmax_))
        ax.set_facecolor('k')

        for i,x in enumerate(x_obst):
            circ = plt.Circle(x,r_obst[i],color='orange',alpha=0.8)
            ax.add_patch(circ)
        # for i, x_ in enumerate(rect_patch):
        #     rect = plt.Rectangle(rect_patch[i][0:2],rect_patch[i][2],rect_patch[i][3], color='c',alpha=0.5)
        #     ax.add_patch(rect)
        color_ = ['g','r']


        t_steps = np.linspace(0, 1, joint_loc.shape[0])
        colors = [(1-t, t, 0) for t in t_steps]  # Gradient from red to green

        

        if batch is False:
            x = joint_loc[:,0]
            y = joint_loc[:,1]
            k_ = 1*color_intensity
            color_ = [k_,k_,k_]
            plt.plot(x, y, 'o-',zorder=0, marker='o',color=color_,lw=lw,mfc='w',
                        solid_capstyle='round')
        else:

            T = joint_loc.shape[0]
            print("joint_loc shape", joint_loc.shape)


            # ax.legend(["target","obstacle"])
            idx = np.arange(0,int(T), skip_frame)
            for count,i in enumerate(idx):
                # color_ = np.where(motion, 1-k_[count], contrast)
                x = joint_loc[i,:,0]
                y = joint_loc[i,:,1]
                plt.plot(x, y, 'o-',zorder=0.9,marker='o',color=colors[i],lw=lw,mfc='w',
                            solid_capstyle='round', alpha=alpha )
                plt.plot(joint_loc[i,:-1,0],joint_loc[i,:-1,1],'oy', markersize=3,alpha=alpha)
                plt.plot(joint_loc[i,-1,0],joint_loc[i,-1,1],'oy', markersize=3,alpha=alpha)


            # for count,i in enumerate(idx_highlight):
            #     color_ = [0.1]*3
            #     x = joint_loc[i,:,0]
            #     y = joint_loc[i,:,1]
            #     plt.plot(x, y, 'o-',zorder=0.9,marker='o',color='k',lw=lw,mfc='w',
            #                 solid_capstyle='round', alpha=0.5)
            for i, x_ in enumerate(x_target):
                ax.scatter(x_[0],x_[1], color='white', marker='x', s=150, zorder=20, linewidths=3)
            

        plt.plot(0,0,color='y',marker='o', markersize=15)
        ax.grid(False)

        if not title is None:
            plt.title(title)
        if not save_as is None:
            fig.savefig('./images/'+save_as+".jpeg",bbox_inches='tight', pad_inches=0.01, dpi=300)

        return plt






class PlanarManipulatorCost:
    ''' 
    Cost functions for various operations with planar manipulator
    Assumes obstacles to be spheres in a plane
    '''
    def __init__(self, robot,p2p_motion=None, x_obst=[], r_obst=[],margin=0.02,
     w_goal=0., w_obst=0.7, w_orient=0., w_ee=0., w_control=0.3,
     b_goal=0.2, b_obst=0.2,b_orient=0.5, b_ee=1., b_control=1., device='cpu'):
        self.device=device
        self.robot=robot # an object of class PlanarManipulator
        self.x_obst=[x.to(device) for x in x_obst] # centers of the obstacles/spheres
        self.r_obst=r_obst # radius of the spheres/spheres
        self.margin=margin # safety margin from the surface (considers the width of the links)
        # Define the nominal cost and weights for each individal cost
        self.b_goal=b_goal; self.b_obst=b_obst; self.b_orient=b_orient; self.b_ee=b_ee; self.b_control=b_control;
        self.w_goal=w_goal; self.w_obst=w_obst; self.w_orient=w_orient; self.w_ee=w_ee; self.w_control=w_control;
        self.p2p = p2p_motion
        self.n_joints = robot.n_joints

    def dist_goal(self, x_goal, ee_loc):
        '''
        metric for error in end-effector pose from the desired
        x_goal: desired position of the end-effector, batch x 2
        ee_loc: actual position/location of the end-effector, batch x 2
        '''
        d_goal = torch.linalg.norm(ee_loc-x_goal, dim=-1)
        return d_goal

    def dist_orient(self, theta_ee_goal, theta_ee_actual):
        '''
            Orientation error
        '''
        d_err = torch.abs(theta_ee_goal-theta_ee_actual)
        d_orient = torch.min(d_err,2*torch.pi-d_err)
        return d_orient

    def dist_traj(self,x_t):
        '''
            metric for length of a trajectory (to seek minimum length trajectory)
            x_t: batch x time x coordinates
        '''
        d_shortest = torch.linalg.norm(x_t[:,-1,:]-x_t[:,0,:], dim=-1)
        d_traj = torch.sum(torch.linalg.norm(x_t[:,1:,:]-x_t[:,:-1,:],dim=-1),dim=-1)
        d_straight = torch.abs(d_traj-d_shortest)/(d_shortest+1e-6)
        return d_straight

    def dist_control(self,theta_t):

        ''' 
        Quantify joint angle sweep 
        theta_t: batch x time x joint, d_theta: batch x 1 x joint
        '''
        theta_shortest = 0.5*torch.linalg.norm(self.robot.theta_max - self.robot.theta_min)
        theta_total = torch.sum(torch.linalg.norm(theta_t[:,1:,:]-theta_t[:,:-1,:],dim=2), dim=1)
        # theta_shortest = torch.linalg.norm(theta_t[:,-1,:]-theta_t[:,0,:], dim=1)
        d_control = torch.abs(theta_total-theta_shortest)/(1e-6+theta_shortest)
        return d_control


    def dist_obst(self, kp_loc):
        ''' 
            metric for obstacle avoidance 
            kp_loc: batch x time x joint x key-point x coordinate
        '''
        batch_size=kp_loc.shape[0]
        d_collisions = torch.zeros(batch_size).to(self.device)
        for i in range(len(self.x_obst)):
            dist2centre = torch.linalg.norm(kp_loc-self.x_obst[i].view(1,1,1,1,-1), dim=-1).view(batch_size,-1)
            dist_in = (dist2centre/(self.r_obst[i]+self.margin))
            dist_in = (1-dist_in)*(dist_in<1)
            d_collisions += torch.sum(dist_in,dim=-1)
        return d_collisions
        
    def cost_ik_2(self,x):
        ''' 
        Cost for inverse kinematics. 
        task-param: ee position and orientation
        decision variables: joint angles
        '''
        x = x.to(self.device)
        x_goal = x[:,:2] # desired position of the end-effector
        theta_ee_goal = x[:,2] # desired orientation of ee
        theta = x[:,3:] # joint angles
        kp_loc, joint_loc, ee_loc, theta_ee = self.robot.forward_kin(theta) # get position of key-points and the end-effector
        
        d_goal = self.dist_goal(x_goal, ee_loc)
        d_obst = self.dist_obst(kp_loc[:,None,:,:,:])
        d_orient = self.dist_orient(theta_ee_goal, theta_ee)

        c_total = self.w_goal*d_goal/self.b_goal+self.w_obst*d_obst/self.b_obst+self.w_orient*d_orient/self.b_orient
        c_return = torch.cat((c_total.view(-1,1), d_goal.view(-1,1), d_obst.view(-1,1),d_orient.view(-1,1)),dim=1)
        return c_return

    # def cost_ik(self,x):
    #     ''' 
    #     Cost for inverse kinematics. 
    #     task-param: ee position 
    #     decision variables: joint angles
    #     '''
    #     x = x.to(self.device)
    #     x_goal = x[:,:2] # desired position of the end-effector
    #     theta = x[:,2:] # joint angles
    #     kp_loc, joint_loc, ee_loc, theta_ee = self.robot.forward_kin(theta) # get position of key-points and the end-effector
        
    #     d_goal = self.dist_goal(x_goal, ee_loc)
    #     d_obst = self.dist_obst(kp_loc[:,None,:,:,:])

    #     c_total = self.w_goal*d_goal/self.b_goal+self.w_obst*d_obst/self.b_obst
    #     c_return = torch.cat((c_total.view(-1,1), d_goal.view(-1,1), d_obst.view(-1,1)),dim=1)
    #     return c_return

    def cost_ik(self,x_goal, x):
        ''' 
        Cost for inverse kinematics. 
        task-param: ee position 
        decision variables: joint angles
        '''
        x = x.to(self.device)
        theta = x # joint angles
        kp_loc, joint_loc, ee_loc, theta_ee = self.robot.forward_kin(theta) # get position of key-points and the end-effector
        
        d_goal = self.dist_goal(x_goal, ee_loc)
        d_obst = self.dist_obst(kp_loc[:,None,:,:,:])

        c_total = self.w_goal*d_goal/self.b_goal+self.w_obst*d_obst/self.b_obst
        c_return = torch.cat((c_total.view(-1,1), d_goal.view(-1,1), d_obst.view(-1,1)),dim=1)
        return c_return



    def cost_goal_x(self, x_goal, x, theta_0):
        ''' x: sequence of joint angles, x_goal: desired position of the end-effector '''
        # x_goal = x[:,:2] # desired position of the end-effector
        # theta = x[:,2:] # joint angles
        # theta_t = self.p2p.gen_traj(theta_0,w)#batchxtimexjoint_angle
        x_t_bounded = self.p2p.smooth_traj(x, theta_0)
        theta = x_t_bounded[:,-1,:]
        kp_loc, joint_loc, ee_loc, theta_ee = self.robot.forward_kin(theta) # get position of key-points and the end-effector
        d_goal = self.dist_goal(x_goal, ee_loc)
        return d_goal    
    

    

    def cost_obst_x(self,x,theta_0):
        ''' 
        Obstacle avoidance cost for motion planning from a fixed joint configuration (theta_0) 
        to a final configuration (theta_1)
        task-param: position of end-effector 
        '''
        batch_size = x.shape[0]
        # theta_1 = x[:,:self.n_joints] #final configuration
        # w = x[:,self.n_joints:] # weights of the basis function for motion between theta_0 to theta_1
        # if theta_0.shape[0]!=batch_size:
        #     theta_0 = theta_0.view(1,-1).repeat(batch_size,1)

        # theta_t = self.p2p.gen_traj(theta_0,w)#batchxtimexjoint_angle
        theta_t = self.p2p.smooth_traj(x, theta_0)
        T01 = theta_t.shape[1]
        kp_loc_t_01, joint_loc_t_01, ee_loc_t_01, theta_ee_t_01 = self.robot.forward_kin(theta_t.reshape(-1,self.n_joints))
        #kp_loc_t: (batch x time) x joint x kp x coordinates
        #ee_loc_t: (batch x time) x coordinates

        kp_loc_t_01 = kp_loc_t_01.view(batch_size,T01,*kp_loc_t_01.shape[1:])
        d_obst = self.dist_obst(kp_loc_t_01)
        
        return d_obst

    def cost_control_x(self,x,theta_0):
        ''' 
        Control cost for motion planning from a fixed joint configuration (theta_0) 
        to a final configuration (theta_1)
        task-param: position of end-effector 
        '''
        batch_size = x.shape[0]
        # theta_1 = x[:,:self.n_joints] #final configuration
        # w = x[:,self.n_joints:] # weights of the basis function for motion between theta_0 to theta_1
        if theta_0.shape[0]!=batch_size:
            theta_0 = theta_0.view(1,-1).repeat(batch_size,1)
        theta_0 = theta_0.reshape(batch_size,1,self.p2p.n).repeat(1,self.p2p.T,1)
        # theta_t = self.p2p.gen_traj(theta_0,w)#batchxtimexjoint_angle
        # z_t = x - x[:,0,:][:,None,:] # so that z(0) = 0
        # x_t = theta_0 + z_t
        # theta_t = self.p2p.bound_traj(x_t)
        theta_t = self.p2p.smooth_traj(x, theta_0)
        d_control = self.dist_traj(theta_t)

        return d_control
    
    def get_ee(self,theta):
        kp_loc, joint_loc, ee_loc, theta_ee = self.robot.forward_kin(theta)
        return ee_loc
    
    def get_traj(self, x, theta_0):
        ''' 
        Get the trajectory for a given decision variable x and initial configuration theta_0
        '''
        batch_size = x.shape[0]
        theta_1 = x[:,:self.n_joints] #final configuration
        w = x[:,self.n_joints:] # weights of the basis function for motion between theta_0 to theta_1
        if theta_0.shape[0]!=batch_size:
            theta_0 = theta_0.view(1,-1).repeat(batch_size,1)

        theta_t = self.p2p.gen_traj_p2p(theta_0,theta_1,w)#batchxtimexjoint_angle

        return theta_t

    
    def dist_ik(self, x_goal, theta):
        ''' 
        Cost for inverse kinematics. 
        task-param: ee position 
        decision variables: joint angles
        '''
        kp_loc, joint_loc, ee_loc, theta_ee = self.robot.forward_kin(theta)
        d_goal = torch.linalg.norm(ee_loc-x_goal, dim=-1)
        return d_goal
        


    def cost_goal_1(self, x_goal, w, theta_0):
        ''' compute the IK cost without obstacles '''
        # x_goal = x[:,:2] # desired position of the end-effector
        # theta = x[:,2:] # joint angles
        theta_t = self.p2p.gen_traj(theta_0,w)#batch x time x joint_angle
        theta = theta_t[:,-1,:]
        kp_loc, joint_loc, ee_loc, theta_ee = self.robot.forward_kin(theta) # get position of key-points and the end-effector
        d_goal = self.dist_goal(x_goal, ee_loc)
        return d_goal    
    
    def cost_goal(self,x_goal, x):
        ''' compute the IK cost without obstacles '''
        tarqet_q = x[:, :self.n_joints]

        w = x[:, self.n_joints:] # joint angles
        kp_loc, joint_loc, ee_loc, theta_ee = self.robot.forward_kin(tarqet_q) # get position of key-points and the end-effector
        d_goal = self.dist_goal(x_goal, ee_loc)
        return d_goal   
    
    def cost_obst(self,x,theta_0):
        ''' 
        Obstacle avoidance cost for motion planning from a fixed joint configuration (theta_0) 
        to a final configuration (theta_1)
        task-param: position of end-effector 
        '''
        batch_size = x.shape[0]
        theta_1 = x[:,:self.n_joints] #final configuration
        w = x[:,self.n_joints:] # weights of the basis function for motion between theta_0 to theta_1
        if theta_0.shape[0]!=batch_size:
            theta_0 = theta_0.view(1,-1).repeat(batch_size,1)

        theta_t = self.p2p.gen_traj_p2p(theta_0,theta_1,w)#batchxtimexjoint_angle
        T01 = theta_t.shape[1]
        kp_loc_t_01, joint_loc_t_01, ee_loc_t_01, theta_ee_t_01 = self.robot.forward_kin(theta_t.view(-1,self.n_joints))
        #kp_loc_t: (batch x time) x joint x kp x coordinates
        #ee_loc_t: (batch x time) x coordinates

        kp_loc_t_01 = kp_loc_t_01.view(batch_size,T01,*kp_loc_t_01.shape[1:])
        d_obst = self.dist_obst(kp_loc_t_01)
        
        return d_obst
    
    def cost_control(self,x,theta_0):
        ''' 
        Control cost for motion planning from a fixed joint configuration (theta_0) 
        to a final configuration (theta_1)
        task-param: position of end-effector 
        '''
        batch_size = x.shape[0]
        theta_1 = x[:,:self.n_joints] #final configuration
        w = x[:,self.n_joints:] # weights of the basis function for motion between theta_0 to theta_1
        if theta_0.shape[0]!=batch_size:
            theta_0 = theta_0.view(1,-1).repeat(batch_size,1)

        
        theta_t = self.p2p.gen_traj_p2p(theta_0,theta_1,w)#batchxtimexjoint_angle

        d_control = self.dist_traj(theta_t)

        return d_control

    def cost_obst_1(self,w,theta_0):
        ''' 
        Obstacle avoidance cost for motion planning from a fixed joint configuration (theta_0) 
        to a final configuration (theta_1)
        task-param: position of end-effector 
        '''
        batch_size = w.shape[0]
        # theta_1 = x[:,:self.n_joints] #final configuration
        # w = x[:,self.n_joints:] # weights of the basis function for motion between theta_0 to theta_1
        if theta_0.shape[0]!=batch_size:
            theta_0 = theta_0.view(1,-1).repeat(batch_size,1)

        theta_t = self.p2p.gen_traj(theta_0,w)#batchxtimexjoint_angle
        T01 = theta_t.shape[1]
        kp_loc_t_01, joint_loc_t_01, ee_loc_t_01, theta_ee_t_01 = self.robot.forward_kin(theta_t.view(-1,self.n_joints))
        #kp_loc_t: (batch x time) x joint x kp x coordinates
        #ee_loc_t: (batch x time) x coordinates

        kp_loc_t_01 = kp_loc_t_01.view(batch_size,T01,*kp_loc_t_01.shape[1:])
        d_obst = self.dist_obst(kp_loc_t_01)
        
        return d_obst

    def cost_control_1(self,w,theta_0):
        ''' 
        Control cost for motion planning from a fixed joint configuration (theta_0) 
        to a final configuration (theta_1)
        task-param: position of end-effector 
        '''
        batch_size = w.shape[0]
        # theta_1 = x[:,:self.n_joints] #final configuration
        # w = x[:,self.n_joints:] # weights of the basis function for motion between theta_0 to theta_1
        if theta_0.shape[0]!=batch_size:
            theta_0 = theta_0.view(1,-1).repeat(batch_size,1)
        
        theta_t = self.p2p.gen_traj(theta_0,w)#batchxtimexjoint_angle

        d_control = self.dist_traj(theta_t)

        return d_control

    def cost_j2p(self,x,theta_0):
        ''' 
        Cost for motion planning from a fixed joint configuration (theta_0) 
        to a given target point for end-effector (x_goal) 
        task-param: position of end-effector 
        '''
        x = x.to(self.device)
        batch_size = x.shape[0]
        x_goal = x[:,:2] # desired position of ee
        theta_1 = x[:,2:2+self.n_joints] #final configuration
        w = x[:,2+self.n_joints:] # weights of the basis function for motion between theta_0 to theta_1
       
        theta_t = self.p2p.gen_traj_p2p(theta_0.view(1,-1).repeat(batch_size,1),theta_1,w)#batchxtimexjoint_angle
        T01 = theta_t.shape[1]
        kp_loc_t_01, joint_loc_t_01, ee_loc_t_01, theta_ee_t_01 = self.robot.forward_kin(theta_t.view(-1,self.n_joints))
        #kp_loc_t: (batch x time) x joint x kp x coordinates
        #ee_loc_t: (batch x time) x coordinates

        kp_loc_t_01 = kp_loc_t_01.view(batch_size,T01,*kp_loc_t_01.shape[1:])
        ee_loc_t_01 = ee_loc_t_01.view(batch_size,T01,-1)
     
        x_actual_1 = ee_loc_t_01[:,-1,:]

        d_goal = self.dist_goal(x_goal,x_actual_1)
        d_obst = self.dist_obst(kp_loc_t_01)
        d_ee = self.dist_traj(ee_loc_t_01)
        d_control = self.dist_traj(theta_t)

        c_total = self.w_goal*d_goal/self.b_goal+self.w_obst*d_obst/self.b_obst +self.w_ee*d_ee/self.b_ee+self.w_control*d_control/self.b_control # total cost 

        c_return = torch.cat((c_total.view(-1,1),d_goal.view(-1,1),
            d_obst.view(-1,1), d_ee.view(-1,1),d_control.view(-1,1)),dim=-1) # for analysis
        
        return c_return 


    def cost_j2p_2(self,x,theta_0):
        ''' 
        Cost for motion planning from a fixed joint configuration (theta_0) 
        to a given target point for end-effector (x_goal) 
        task-param: position of end-effector and the orientation
        '''

        x = x.to(self.device)
        batch_size = x.shape[0]
        x_goal = x[:,:2] # desired position of ee
        theta_ee_goal = x[:,2]
        theta_1 = x[:,3:3+self.n_joints] #final configuration
        w = x[:,3+self.n_joints:] # weights of the basis function for motion between theta_0 to theta_1
       
        theta_t = self.p2p.gen_traj_p2p(theta_0.view(1,-1).repeat(batch_size,1),theta_1,w)#batchxtimexjoint_angle
        T01 = theta_t.shape[1]
        kp_loc_t_01, joint_loc_t_01, ee_loc_t_01, theta_ee_t_01 = self.robot.forward_kin(theta_t.view(-1,self.n_joints))
        #kp_loc_t: (batch x time) x joint x kp x coordinates
        #ee_loc_t: (batch x time) x coordinates

        kp_loc_t_01 = kp_loc_t_01.view(batch_size,T01,*kp_loc_t_01.shape[1:])
        ee_loc_t_01 = ee_loc_t_01.view(batch_size,T01,-1)
     
        theta_ee_1 = theta_ee_t_01.view(batch_size,T01)[:,-1]

        x_actual_1 = e_loc_t_01[:,-1,:]

        d_goal = self.dist_goal(x_goal,x_actual_1)
        d_obst = self.dist_obst(kp_loc_t)
        d_orient = self.dist_orient(theta_ee_goal, theta_ee_1)
        d_ee = self.dist_traj(ee_loc_t)
        d_control = self.dist_traj(theta_t)

        c_total = self.w_goal*d_goal/self.b_goal+self.w_obst*d_obst/self.b_obst+self.w_orient*d_orient/self.b_orient +self.w_ee*d_ee/self.b_ee+self.w_control*d_control/self.b_control # total cost 

        c_return = torch.cat((c_total.view(-1,1),d_goal.view(-1,1),
            d_obst.view(-1,1),d_orient.view(-1,1), d_ee.view(-1,1),d_control.view(-1,1)),dim=-1) # for analysis
        
        return c_return 

    def cost_j2p2j(self,x,theta_0, theta_2): 
        ''' 
            Cost for motion planning from a fixed joint configuration (theta_0) to
            a fixed final configuration (theta_2) via a target point for end-effector (x_goal) 
            task-param: position of end-effector
         '''
        x = x.to(self.device)
        batch_size = x.shape[0]
        x_goal = x[:,:2] # desired position of ee (via-point)

        theta_1 = x[:,2:2+self.n_joints] # intermediate configuration at the via-point
        w = x[:,2+self.n_joints:] # weights of the basis function
        w01 = w[:,:int(w.shape[-1]/2)] # weights for the first part of the motion: theta_0 to theta_1
        w12 = w[:,int(w.shape[-1]/2):] # for motion between theta_1 to theta_2

        theta_t_01 = self.p2p.gen_traj_p2p(theta_0.view(1,-1).repeat(batch_size,1),theta_1,w01)#batchxtimexjoint_angle
        theta_t_12 = self.p2p.gen_traj_p2p(theta_1,theta_2.view(1,-1).repeat(batch_size,1),w12)#batchxtimexjoint_angle
         
        T01 = theta_t_01.shape[1]
        T12 = theta_t_12.shape[1]

        kp_loc_t_01, joint_loc_t_01, ee_loc_t_01, theta_ee_t_01 = self.robot.forward_kin(theta_t_01.view(-1,self.n_joints))
        kp_loc_t_12, joint_loc_t_12, ee_loc_t_12, theta_ee_t_12 = self.robot.forward_kin(theta_t_12.view(-1,self.n_joints))
        
        #kp_loc_t: (batch x time) x joint x kp x coordinates
        #ee_loc_t: (batch x time) x coordinates

        kp_loc_t_01 = kp_loc_t_01.view(batch_size,T01,*kp_loc_t_01.shape[1:])
        kp_loc_t_12 = kp_loc_t_12.view(batch_size,T12,*kp_loc_t_12.shape[1:])
        ee_loc_t_01 = ee_loc_t_01.view(batch_size,T01,-1)
        ee_loc_t_12 = ee_loc_t_12.view(batch_size,T12,-1)


        x_actual_1 = ee_loc_t_01[:,-1,:]
        
        d_goal = self.dist_goal(x_goal,x_actual_1) # for via point
        d_obst = 0.5*(self.dist_obst(kp_loc_t_01)+self.dist_obst(kp_loc_t_12))
        d_ee = 0.5*(self.dist_traj(ee_loc_t_01)+self.dist_traj(ee_loc_t_12))
        d_control = 0.5*(self.dist_traj(theta_t_01)+self.dist_traj(theta_t_12))


        c_total = self.w_goal*d_goal/self.b_goal+self.w_obst*d_obst/self.b_obst+self.w_ee*d_ee/self.b_ee+self.w_control*d_control/self.b_control # total cost 

        c_return = torch.cat((c_total.view(-1,1),d_goal.view(-1,1),
            d_obst.view(-1,1),d_control.view(-1,1)),dim=-1) # for analysis

        return c_return


    def cost_j2p2j_2(self,x,theta_0, theta_2): 
        ''' 
            Cost for motion planning from a fixed joint configuration (theta_0) to
            a fixed final configuration (theta_2) via a target point for end-effector (x_goal) 
            task-param: position and orientation of end-effector
         '''

        x = x.to(self.device)
        batch_size = x.shape[0]
        x_goal = x[:,:2] # desired position of ee (via-point)
        theta_ee_goal = x[:,2]

        theta_1 = x[:,3:3+self.n_joints] # intermediate configuration at the via-point
        w = x[:,3+self.n_joints:] # weights of the basis function
        w01 = w[:,:int(w.shape[-1]/2)] # weights for the first part of the motion: theta_0 to theta_1
        w12 = w[:,int(w.shape[-1]/2):] # for motion between theta_1 to theta_2

        theta_t_01 = self.p2p.gen_traj_p2p(theta_0.view(1,-1).repeat(batch_size,1),theta_1,w01)#batchxtimexjoint_angle
        theta_t_12 = self.p2p.gen_traj_p2p(theta_1,theta_2.view(1,-1).repeat(batch_size,1),w12)#batchxtimexjoint_angle
         
        T01 = theta_t_01.shape[1]
        T12 = theta_t_12.shape[1]

        kp_loc_t_01, joint_loc_t_01, ee_loc_t_01, theta_ee_t_01 = self.robot.forward_kin(theta_t_01.view(-1,self.n_joints))
        kp_loc_t_12, joint_loc_t_12, ee_loc_t_12, theta_ee_t_12 = self.robot.forward_kin(theta_t_12.view(-1,self.n_joints))
        
        #kp_loc_t: (batch x time) x joint x kp x coordinates
        #ee_loc_t: (batch x time) x coordinates

        kp_loc_t_01 = kp_loc_t_01.view(batch_size,T01,*kp_loc_t_01.shape[1:])
        kp_loc_t_12 = kp_loc_t_12.view(batch_size,T12,*kp_loc_t_12.shape[1:])
        ee_loc_t_01 = ee_loc_t_01.view(batch_size,T01,-1)
        ee_loc_t_12 = ee_loc_t_12.view(batch_size,T12,-1)


        theta_ee_1 = theta_ee_t_01.view(batch_size,T01)[:,-1]
        x_actual_1 = ee_loc_t_01[:,-1,:]
        
        d_goal = self.dist_goal(x_goal,x_actual_1) # for via point
        d_obst = 0.5*(self.dist_obst(kp_loc_t_01)+self.dist_obst(kp_loc_t_12))
        d_ee = 0.5*(self.dist_traj(ee_loc_t_01)+self.dist_traj(ee_loc_t_12))
        d_control = 0.5*(self.dist_traj(theta_t_01)+self.dist_traj(theta_t_12))
        d_orient = self.dist_orient(theta_ee_goal, theta_ee_1)


        c_total = self.w_goal*d_goal/self.b_goal+self.w_obst*d_obst/self.b_obst+self.w_orient*d_orient/self.b_orient +self.w_ee*d_ee/self.b_ee+self.w_control*d_control/self.b_control # total cost 

        c_return = torch.cat((c_total.view(-1,1),d_goal.view(-1,1),
            d_obst.view(-1,1),d_orient.view(-1,1), d_ee.view(-1,1),d_control.view(-1,1)),dim=-1) # for analysis

        return c_return

    def cost_j2p2p(self,x,theta_0): 
        ''' 
            Cost for motion planning from a fixed joint configuration (theta_0) to
            a fixed final configuration (theta_2) via a target point for end-effector (x_goal) 
            task-param: position of ee at  via points
        '''
        x = x.to(self.device)
        batch_size = x.shape[0]
        x_goal_1 = x[:,:2] # desired position of ee at via point
        x_goal_2 = x[:,2:4] # desired position of ee at the final point
        theta_1 = x[:,4:4+self.n_joints] # via configuration
        theta_2 = x[:,4+self.n_joints:4+2*self.n_joints] # via configuration
        w = x[:,4+2*self.n_joints:] # weights of the basis function
        w01 = w[:,:int(w.shape[-1]/2)] # weights for the first part of the motion: theta_0 to theta_1
        w12 = w[:,int(w.shape[-1]/2):] # for motion between theta_1 to theta_2

        theta_t_01 = self.p2p.gen_traj_p2p(theta_0.view(1,-1).repeat(batch_size,1),theta_1,w01)#batchxtimexjoint_angle
        theta_t_12 = self.p2p.gen_traj_p2p(theta_1,theta_2,w12)#batchxtimexjoint_angle
         
        T01 = theta_t_01.shape[1]
        T12 = theta_t_12.shape[1]

        kp_loc_t_01, joint_loc_t_01, ee_loc_t_01, theta_ee_t_01 = self.robot.forward_kin(theta_t_01.view(-1,self.n_joints))
        kp_loc_t_12, joint_loc_t_12, ee_loc_t_12, theta_ee_t_12 = self.robot.forward_kin(theta_t_12.view(-1,self.n_joints))
        #kp_loc_t: (batch x time) x joint x kp x coordinates
        #ee_loc_t: (batch x time) x coordinates
    
        kp_loc_t_01 = kp_loc_t_01.view(batch_size,T01,*kp_loc_t_01.shape[1:])
        kp_loc_t_12 = kp_loc_t_12.view(batch_size,T12,*kp_loc_t_12.shape[1:])
        ee_loc_t_01 = ee_loc_t_01.view(batch_size,T01,-1)
        ee_loc_t_12 = ee_loc_t_12.view(batch_size,T12,-1)

    
        x_actual_1 = ee_loc_t_01[:,-1,:]
        x_actual_2 = ee_loc_t_12[:,-1,:]


        d_goal = 0.5*(self.dist_goal(x_goal_1, x_actual_1)+self.dist_goal(x_goal_2,x_actual_2)) # for via point
        d_obst = 0.5*(self.dist_obst(kp_loc_t_01)+self.dist_obst(kp_loc_t_12))
        d_ee = 0.5*(self.dist_traj(ee_loc_t_01)+self.dist_traj(ee_loc_t_12))
        d_control = 0.5*(self.dist_traj(theta_t_01)+self.dist_traj(theta_t_12))

        c_total = self.w_goal*d_goal/self.b_goal+self.w_obst*d_obst/self.b_obst+self.w_ee*d_ee/self.b_ee+self.w_control*d_control/self.b_control # total cost 

        c_return = torch.cat((c_total.view(-1,1),d_goal.view(-1,1),
            d_obst.view(-1,1), d_ee.view(-1,1),d_control.view(-1,1)),dim=-1) # for analysis

        return c_return



    def cost_j2p2p_2(self,x,theta_0): 
        ''' 
            Cost for motion planning from a fixed joint configuration (theta_0) to
            a fixed final configuration (theta_2) via a target point for end-effector (x_goal) 
            task-param: position and orientation of ee at  via points
        '''

        x = x.to(self.device)
        batch_size = x.shape[0]
        x_goal_1 = x[:,:2] # desired position of ee at via point
        x_goal_2 = x[:,2:4] # desired position of ee at the final point
        theta_ee_goal_1 = x[:,4]
        theta_ee_goal_1 = x[:,5]
        theta_1 = x[:,5:5+self.n_joints] # via configuration
        theta_2 = x[:,5+self.n_joints:5+2*self.n_joints] # via configuration
        w = x[:,5+2*self.n_joints:] # weights of the basis function
        w01 = w[:,:int(w.shape[-1]/2)] # weights for the first part of the motion: theta_0 to theta_1
        w12 = w[:,int(w.shape[-1]/2):] # for motion between theta_1 to theta_2

        theta_t_01 = self.p2p.gen_traj_p2p(theta_0.view(1,-1).repeat(batch_size,1),theta_1,w01)#batchxtimexjoint_angle
        theta_t_12 = self.p2p.gen_traj_p2p(theta_1,theta_2,w12)#batchxtimexjoint_angle
         
        T01 = theta_t_01.shape[1]
        T12 = theta_t_12.shape[1]

        kp_loc_t_01, joint_loc_t_01, ee_loc_t_01, theta_ee_t_01 = self.robot.forward_kin(theta_t_01.view(-1,self.n_joints))
        kp_loc_t_12, joint_loc_t_12, ee_loc_t_12, theta_ee_t_12 = self.robot.forward_kin(theta_t_12.view(-1,self.n_joints))
        #kp_loc_t: (batch x time) x joint x kp x coordinates
        #ee_loc_t: (batch x time) x coordinates
    
        kp_loc_t_01 = kp_loc_t_01.view(batch_size,T01,*kp_loc_t_01.shape[1:])
        kp_loc_t_12 = kp_loc_t_12.view(batch_size,T12,*kp_loc_t_12.shape[1:])
        ee_loc_t_01 = ee_loc_t_01.view(batch_size,T01,-1)
        ee_loc_t_12 = ee_loc_t_12.view(batch_size,T12,-1)
 
        theta_ee_1 = theta_ee_t_01.view(batch_size,T01)[:,-1]
        theta_ee_2 = theta_ee_t_12.view(batch_size,T12)[:,-1]

    
        x_actual_1 = ee_loc_t_01[:,-1,:]
        x_actual_2 = ee_loc_t_12[:,-1,:]


        d_goal = 0.5*(self.dist_goal(x_goal_1, x_actual_1)+self.dist_goal(x_goal_2,x_actual_2)) # for via point
        d_obst = 0.5*(self.dist_obst(kp_loc_t_01)+self.dist_obst(kp_loc_t_12))
        d_ee = 0.5*(self.dist_traj(ee_loc_t_01)+self.dist_traj(ee_loc_t_12))
        d_control = 0.5*(self.dist_traj(theta_t_01)+self.dist_traj(theta_t_12))
        d_orient = 0.5*(self.dist_orient(theta_ee_goal_1, theta_ee_1),self.dist_orient(theta_ee_goal_2, theta_ee_2))


        c_total = self.w_goal*d_goal/self.b_goal+self.w_obst*d_obst/self.b_obst+self.w_orient*d_orient/self.b_orient +self.w_ee*d_ee/self.b_ee+self.w_control*d_control/self.b_control # total cost 

        c_return = torch.cat((c_total.view(-1,1),d_goal.view(-1,1),
            d_obst.view(-1,1),d_orient.view(-1,1), d_ee.view(-1,1),d_control.view(-1,1)),dim=-1) # for analysis

        return c_return

    def cost_j2p2p2j(self,x,theta_0, theta_3): 
        ''' 
            Cost for motion planning from a fixed joint configuration (theta_0) to the final configuration (theta_3)
            but via two intermediary points for the end-effector: x_goal_1 and x_goal_2 
            task-param: position of ee at the via points
         '''
        x = x.to(self.device)
        batch_size = x.shape[0]
        x_goal_1 = x[:,:2] # desired position of ee at via point
        x_goal_2 = x[:,2:4] # desired position of ee at the final point
        theta_1 = x[:,4:4+self.n_joints] # via configuration
        theta_2 = x[:,4+self.n_joints:4+2*self.n_joints] # via configuration
        theta_1 = x[:,4:4+self.n_joints] # via configuration
        theta_2 = x[:,4+self.n_joints:4+2*self.n_joints] # via configuration
        w = x[:,4+2*self.n_joints:] # weights of the basis function
        w01 = w[:,:int(w.shape[-1]/3)] # weights for the first part of the motion: theta_0 to theta_1
        w12 = w[:,int(w.shape[-1]/3):2*int(w.shape[-1]/3)] # for motion between theta_1 to theta_2
        w23 = w[:,2*int(w.shape[-1]/3):] # for motion between theta_2 to theta_0

        theta_t_01 = self.p2p.gen_traj_p2p(theta_0.view(1,-1).repeat(batch_size,1),theta_1,w01)#batchxtimexjoint_angle
        theta_t_12 = self.p2p.gen_traj_p2p(theta_1,theta_2,w12)#batchxtimexjoint_angle
        theta_t_23 = self.p2p.gen_traj_p2p(theta_2,theta_3.view(1,-1).repeat(batch_size,1),w23)#batchxtimexjoint_angle
         
        T01 = theta_t_01.shape[1]
        T12 = theta_t_12.shape[1]
        T23 = theta_t_23.shape[1]

        kp_loc_t_01, joint_loc_t_01, ee_loc_t_01, theta_ee_t_01 = self.robot.forward_kin(theta_t_01.view(-1,self.n_joints))
        kp_loc_t_12, joint_loc_t_12, ee_loc_t_12, theta_ee_t_12 = self.robot.forward_kin(theta_t_12.view(-1,self.n_joints))
        kp_loc_t_23, joint_loc_t_23, ee_loc_t_23, theta_ee_t_23 = self.robot.forward_kin(theta_t_23.view(-1,self.n_joints))
        #kp_loc_t: (batch x time) x joint x kp x coordinates
        #ee_loc_t: (batch x time) x coordinates

        kp_loc_t_01 = kp_loc_t_01.view(batch_size,T01,*kp_loc_t_01.shape[1:])
        kp_loc_t_12 = kp_loc_t_12.view(batch_size,T12,*kp_loc_t_12.shape[1:])
        kp_loc_t_23 = kp_loc_t_23.view(batch_size,T23,*kp_loc_t_23.shape[1:])
        ee_loc_t_01 = ee_loc_t_01.view(batch_size,T01,-1)
        ee_loc_t_12 = ee_loc_t_12.view(batch_size,T12,-1)
        ee_loc_t_23 = ee_loc_t_23.view(batch_size,T23,-1)
        
   
        x_actual_1 = ee_loc_t_01[:,-1,:]
        x_actual_2 = ee_loc_t_12[:,-1,:]

    
        d_goal = 0.5*(self.dist_goal(x_goal_1,x_actual_1)+self.dist_goal(x_goal_2,x_actual_2)) # for via point
        d_obst = (1./3.)*(self.dist_obst(kp_loc_t_01)+self.dist_obst(kp_loc_t_12)+self.dist_obst(kp_loc_t_23))
        d_ee = (1./3.)*(self.dist_traj(ee_loc_t_01)+self.dist_traj(ee_loc_t_12)+self.dist_traj(ee_loc_t_23))
        d_control = (1./3.)*(self.dist_traj(theta_t_01)+self.dist_traj(theta_t_12)+self.dist_traj(theta_t_23))

        c_total = self.w_goal*d_goal/self.b_goal+self.w_obst*d_obst/self.b_obst+self.w_ee*d_ee/self.b_ee+self.w_control*d_control/self.b_control # total cost 

        c_return = torch.cat((c_total.view(-1,1),d_goal.view(-1,1),
            d_obst.view(-1,1),d_control.view(-1,1)),dim=-1) # for analysis

        return c_return




    def cost_j2p2p2j_2(self,x,theta_0, theta_3): 
        ''' 
            Cost for motion planning from a fixed joint configuration (theta_0) to the final configuration (theta_3)
            but via two intermediary points for the end-effector: x_goal_1 and x_goal_2 
            task-param: position and orientation of ee at the via points
         '''
        x = x.to(self.device)
        batch_size = x.shape[0]
        x_goal_1 = x[:,:2] # desired position of ee at via point
        x_goal_2 = x[:,2:4] # desired position of ee at the final point
        theta_ee_goal_1 = x[:,4]
        theta_ee_goal_1 = x[:,5]
        theta_1 = x[:,5:5+self.n_joints] # via configuration
        theta_2 = x[:,5+self.n_joints:5+2*self.n_joints] # via configuration
        theta_1 = x[:,5:5+self.n_joints] # via configuration
        theta_2 = x[:,5+self.n_joints:5+2*self.n_joints] # via configuration
        w = x[:,5+2*self.n_joints:] # weights of the basis function
        w01 = w[:,:int(w.shape[-1]/3)] # weights for the first part of the motion: theta_0 to theta_1
        w12 = w[:,int(w.shape[-1]/3):2*int(w.shape[-1]/3)] # for motion between theta_1 to theta_2
        w23 = w[:,2*int(w.shape[-1]/3):] # for motion between theta_2 to theta_0

        theta_t_01 = self.p2p.gen_traj_p2p(theta_0.view(1,-1).repeat(batch_size,1),theta_1,w01)#batchxtimexjoint_angle
        theta_t_12 = self.p2p.gen_traj_p2p(theta_1,theta_2,w12)#batchxtimexjoint_angle
        theta_t_23 = self.p2p.gen_traj_p2p(theta_2,theta_3.view(1,-1).repeat(batch_size,1),w23)#batchxtimexjoint_angle
         
        T01 = theta_t_01.shape[1]
        T12 = theta_t_12.shape[1]
        T23 = theta_t_23.shape[1]

        kp_loc_t_01, joint_loc_t_01, ee_loc_t_01, theta_ee_t_01 = self.robot.forward_kin(theta_t_01.view(-1,self.n_joints))
        kp_loc_t_12, joint_loc_t_12, ee_loc_t_12, theta_ee_t_12 = self.robot.forward_kin(theta_t_12.view(-1,self.n_joints))
        kp_loc_t_23, joint_loc_t_23, ee_loc_t_23, theta_ee_t_23 = self.robot.forward_kin(theta_t_23.view(-1,self.n_joints))
        #kp_loc_t: (batch x time) x joint x kp x coordinates
        #ee_loc_t: (batch x time) x coordinates

        kp_loc_t_01 = kp_loc_t_01.view(batch_size,T01,*kp_loc_t_01.shape[1:])
        kp_loc_t_12 = kp_loc_t_12.view(batch_size,T12,*kp_loc_t_12.shape[1:])
        kp_loc_t_23 = kp_loc_t_23.view(batch_size,T23,*kp_loc_t_23.shape[1:])
        ee_loc_t_01 = ee_loc_t_01.view(batch_size,T01,-1)
        ee_loc_t_12 = ee_loc_t_12.view(batch_size,T12,-1)
        ee_loc_t_23 = ee_loc_t_23.view(batch_size,T23,-1)
        
        theta_ee_1 = theta_ee_t_01.view(batch_size,T01)[:,-1]
        theta_ee_2 = theta_ee_t_12.view(batch_size,T12)[:,-1]

        x_actual_1 = ee_loc_t_01[:,-1,:]
        x_actual_2 = ee_loc_t_12[:,-1,:]

    
        d_goal = 0.5*(self.dist_goal(x_goal_1,x_actual_1)+self.dist_goal(x_goal_2,x_actual_2)) # for via point
        d_obst = (1./3.)*(self.dist_obst(kp_loc_t_01)+self.dist_obst(kp_loc_t_12)+self.dist_obst(kp_loc_t_23))
        d_ee = (1./3.)*(self.dist_traj(ee_loc_t_01)+self.dist_traj(ee_loc_t_12)+self.dist_traj(ee_loc_t_23))
        d_control = (1./3.)*(self.dist_traj(theta_t_01)+self.dist_traj(theta_t_12)+self.dist_traj(theta_t_23))
        d_orient = 0.5*(self.dist_orient(theta_ee_goal_1, theta_ee_1),self.dist_orient(theta_ee_goal_2, theta_ee_2))



        c_total = self.w_goal*d_goal/self.b_goal+self.w_obst*d_obst/self.b_obst+self.w_orient*d_orient/self.b_orient +self.w_ee*d_ee/self.b_ee+self.w_control*d_control/self.b_control # total cost 

        c_return = torch.cat((c_total.view(-1,1),d_goal.view(-1,1),
            d_obst.view(-1,1),d_orient.view(-1,1), d_ee.view(-1,1),d_control.view(-1,1)),dim=-1) # for analysis

        return c_return


    def cost_j2j(self,x, theta_0, theta_f):
        ''' Given  (init_joint_angle, final_joint_angle, basis_weights) define the cost for reaching task'''
        batch_size = x.shape[0]
        x = x.to(self.device)
        theta_0 = theta_0.repeat(batch_size,1)
        theta_f = theta_f.repeat(batch_size,1)
        w = 1*x # weights
        theta_t = self.p2p.gen_traj_p2p(theta_0,theta_f,w) #joint angles: batch x time x joint
        T = theta_t.shape[1]

        key_loc_t,joint_loc_t, ee_loc_t, theta_ee_t = self.robot.forward_kin(theta_t.view(-1,self.n_joints)) # (batchxtime) x joint x key x positions

        key_loc_t = key_loc_t.view(batch_size,T,*key_loc_t.shape[1:])
        ee_loc_t = ee_loc_t.view(batch_size,T,-1)

        # Cost due to obstacle
        d_obst = self.dist_obst(key_loc_t)

        
        # Cost on end-effector traj (aim to keep it straight)
        d_control = self.dist_control(theta_t)
    
        c_total =  self.w_obst*d_obst/self.b_obst + self.w_control*d_control/self.b_control

        c_return = torch.cat((c_total.view(-1,1), d_obst.view(-1,1),d_control.view(-1,1)),dim=-1) # for analysis
 

        return c_return



class PointMassCost:
    ''' Point-to-Point Motion of a point mass (motion planning) '''
    def __init__(self, p2p_motion, x_obst=[], r_obst=[], margin=0.01,
     w_obst=1., w_straight=1.,b_obst=1., b_straight=1., device='cpu'):
        '''
            p2p_motion: object of class Point2PointMotion (for generating trajectory)
            x_obst: a list of center of spherical obstacles in the plane
            r_obst: a list containing radius of the spherical obstscles
            margin: safety distance from the edge of the obstacle
            w_obst: weight for obstacle avoidance
            w_straight: weight for keeping the trajectory as staight as possible
            b_* are nominal values for the corresponding distances (see the cost function below)
        '''
        self.device=device
        self.x_obst=x_obst # centers of the obstacles/spheres
        self.r_obst=r_obst # radius of the spheres/spheres
        self.margin=margin # safety margin from the surface (considers the width of the links)
        self.b_obst=b_obst;self.b_straight=b_straight 
        self.w_obst=w_obst; self.w_straight=w_straight 
        self.p2p = p2p_motion

    def dist_obst(self, x_t):
        ''' A matric for obstacle collision '''
        batch_size=x_t.shape[0]
        d_collisions = torch.zeros(batch_size).to(self.device)
        for i in range(len(self.x_obst)):
            dist2centre = torch.linalg.norm(x_t-self.x_obst[i].view(1,1,-1), dim=-1).view(batch_size,-1)
            dist_in = (dist2centre/(self.r_obst[i]+self.margin))
            dist_in = (1-dist_in)*(dist_in<1)
            d_collisions += torch.sum(dist_in,dim=-1)
        return d_collisions

    def dist_straight(self,x_t):
        ''' metric for how straight is a trajectory (to seek minimal length) '''
        d_shortest = torch.linalg.norm(x_t[:,-1,:]-x_t[:,0,:], dim=-1)
        d_traj = torch.sum(torch.linalg.norm(x_t[:,1:,:]-x_t[:,:-1,:],dim=-1),dim=-1)
        d_straight = torch.abs(d_traj-d_shortest)/(d_shortest+1e-6)
        return d_straight
        
    def cost_motion(self,x, x0):
        '''move from left end of the plane to the right end '''
        x=x.to(self.device)
        x0 = x0.view(1,2).repeat(x.shape[0],1) # initial point
        xf = torch.cat((0.95+0*x[:,0],x[:,0]),dim=-1)
        w = x[:,1:] # weights of the basis function

        x_t = self.p2p.gen_traj_p2p(x0,xf,w)
        d_obst = self.dist_obst(x_t)
        d_straight = self.dist_straight(x_t)

        c_total = (self.w_straight*d_straight/self.b_straight+self.w_obst*d_obst/self.b_obst)
        c_return = torch.cat((c_total.view(-1,1),d_obst.view(-1,1),d_straight.view(-1,1)),dim=1)
        return c_return


##########################################################################################
##########################################################################################
##########################################################################################


class Point2PointMotion:
    '''
    Generates point to point motion satisfying the boundary conditions while maintaining:
        - the velocity at the intial and final step zero,
        - the bounds on the trajectory (Ex: joint limits)

    The generated trajectory trajectory represents the phase of the movement t in (0,1).
    params: 
        - dt: time/phase step (assumin t in (0,1))
        - K: number of basis functions 
        - basis: {"rbf", "rbf2", "bs"} where "rbf2" is the inverse rbf, "bs" is bernstein polynomial
        - n: number of variables/states
    '''
    def __init__(self, n,  T=1, dt=0.01, K=3, basis="rbf", bounds=None, device="cpu"):
        self.device = device
        self.n = n # number of variables/coordinates
        self.T = int(T/dt) # number of time steps
        self.t = torch.linspace(0,1,self.T).to(device) # phase
        self.K = K # number of basis functions
        self.basis = basis
        if basis == "rbf":
            self.Phi = self.Phi_rbf().to(device)
        elif basis == "rbf2":
            self.Phi = self.Phi_rbf2().to(device)
        elif basis == "bs":
            self.Phi = self.Phi_Bs().to(device)
        elif basis == "bs2":
            self.Phi = self.Phi_Bs2().to(device)
        self.set_bound(bounds) # bounds is either None (no limit) or a list containing lower and upper bound

    def set_device(self,device):
        self.device=device

    def set_bound(self, bounds):
        if bounds is None:
            bounds=[]
            bounds.append(torch.tensor([-10**5]*self.n).to(self.device)) # lower bound
            bounds.append(-1*bounds[0])
        self.lower_bound = bounds[0].reshape(1,1,-1)  # lower limit on the trajectory
        self.upper_bound = bounds[1].reshape(1,1,-1) # upper limit on the trajectory


    def Phi_rbf(self): #RBF
        t = torch.linspace(0,1,self.T).to(self.device)
        r_rbf = 0.5/(self.K) # radius
        c_rbf = torch.linspace(0,1,self.K+2).to(self.device)[1:-1] # centers
        Phi = torch.empty((self.T,self.K)).to(self.device)
        for k in range(self.K):
            Phi[:,k]=torch.exp(-(t-c_rbf[k])**2/r_rbf**2)
        return Phi

    def Phi_rbf2(self): # Inverse RBF
        t = torch.linspace(0,1,self.T).to(self.device)
        r_rbf = 0.5/self.K
        c_rbf = torch.linspace(0,1,self.K+2).to(self.device)[1:-1]   
        Phi = torch.empty((self.T,self.K)).to(self.device)
        for k in range(self.K):
            Phi[:,k] = (1/(1+torch.exp((t-c_rbf[k])**2/r_rbf**2)))
        return Phi


    def Phi_Bs(self): # Bernstein Polynomial
        t = torch.linspace(0,1,self.T)
        Phi = torch.zeros((self.T,self.K))
        for k in range(self.K):
            b = np.math.factorial(self.K)/(np.math.factorial(self.K-k)*np.math.factorial(k))
            Phi[:,k]=(b*((1-t)**(self.K-k))*(t**k))
        Phi = Phi - Phi[0,:]+ 1e-9 
        return Phi
    
    def Phi_Bs2(self): # Bernstein Polynomial
        t = torch.linspace(0,1,self.T)
        Phi = torch.zeros((self.T,self.K))
        for k in range(self.K):
            b = np.math.factorial(self.K)/(np.math.factorial(self.K-k)*np.math.factorial(k))
            Phi[:,k]=(b*((1-t)**(self.K-k))*(t**k))
        Phi = Phi + 1e-9 
        return Phi

    def gen_traj(self,x_0, w):
        '''
            Given the initial state (batch x n ) and the weights (batch x K*n)
            generate trajectories with only initial condition satisfied
        '''
        batch_size = w.shape[0]
        x_0 = x_0[:,None,:].repeat(1,self.T,1) #batch x time x n, initial condition
        w = w.reshape(batch_size,self.K,self.n) #weights
        z_t = torch.einsum('jk,ikl->ijl',self.Phi,w) # batch x time x n
        z_t = z_t - z_t[:,0,:][:,None,:] # so that z(0) = 0
        x_t = x_0 + z_t
        x_t_bounded = self.bound_traj(x_t) # clip the trajectory to maintain the upper and lower limits
        return x_t_bounded #.reshape(batch_size,self.T,self.n)
    
    def gen_traj_push(self,x_0, w):
        '''
            Given the initial state (batch x n ) and the weights (batch x K*n)
            generate trajectories with only initial condition satisfied
        '''
        batch_size = w.shape[0]
        x_0 = x_0[:,None,:].repeat(1,self.T,1) #batch x time x n, initial condition
        w = w.reshape(batch_size,self.K,self.n) #weights
        z_t = torch.einsum('jk,ikl->ijl',self.Phi,w) # batch x time x n
        # z_t = z_t - z_t[:,0,:][:,None,:] # so that z(0) = 0
        x_t = x_0 + z_t
        x_t_bounded = self.bound_traj(x_t) # clip the trajectory to maintain the upper and lower limits
        return x_t_bounded #.reshape(batch_size,self.T,self.n)

    def gen_traj_p2p(self,x_0, x_f, w):
        ''' 
            generate trajectory with boundary conditions satisfied
            x_0: batch x n, initial state
            x_f: batc x n, final state
            w: batch x (K*n), weights of basis function, the 
        '''
        batch_size = w.shape[0]
        x_0 = x_0.reshape(batch_size,1,self.n).repeat(1,self.T,1) #batch x time x n
        x_f = x_f.reshape(batch_size,1,self.n).repeat(1,self.T,1) #batch x time x n
        w = w.reshape(batch_size,self.K,self.n)
        z_t = torch.einsum('jk,ikl->ijl',self.Phi,w) # batch x time x n
        z_0 = z_t[:,0,:][:,None,:]
        z_f = z_t[:,-1,:][:,None,:]
        x_t = x_0 + z_t - z_0 + torch.einsum('j,ijk->ijk',self.t,x_f-x_0+z_0-z_f) # x(t) = x(0)+ z(t)-z(0)+t*(x(1)-x(0)+z(0)-z(1))
        x_t_bounded = self.bound_traj(x_t)  # clip the trajectory to maintain the upper and lower limits
        return x_t_bounded # (batch_size,self.T,self.n)
    
    def smooth_traj(self, x, theta_0):
        """
        fine-tune the trajectory to be smooth, bounded, and start from theta_0
        """
        z_t = x - x[:,0,:][:,None,:] # so that z(0) = 0
        x_t = theta_0 + z_t
        x_t_bounded = self.bound_traj(x_t)
        return x_t_bounded

 
    def bound_traj(self,x):
        ''' 
            clip the given trajectories (batch x T x n)
            within the limits and smoothen it and maintain the boundary conditions
        '''
        delta = self.upper_bound-self.lower_bound
        lower_x = self.lower_bound + delta*0.01
        upper_x = self.upper_bound - delta*0.01
        x = torch.clip(x, lower_x, upper_x ) # clip it
        
        # running average for filtering (also ensures zero velocity at the boundaries)
        k = 4 # set (k>0)
        x = torch.cat((x[:,0,:][:,None,:].repeat(1,2*k,1), x, 
            x[:,-1,:][:,None,:].repeat(1,2*k,1)),dim=1)
        
        cum_l = x[:,k:-k,:].shape[1]
        cum_x = 8*x[:,k:k+cum_l,:]+3*(x[:,(k-1):(k-1+cum_l),:]+
            x[:,(k+1):(k+1+cum_l),:])+2*(x[:,(k-2):(k-2+cum_l),:]+
            x[:,(k+2):(k+2+cum_l),:])+1*(x[:,(k-3):(k-3+cum_l),:]+
            x[:,(k+3):(k+3+cum_l),:])
        cum_w = 2*(4+3+2+1)

        x_transformed = cum_x/cum_w

        return x_transformed
