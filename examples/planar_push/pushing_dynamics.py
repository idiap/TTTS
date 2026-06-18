#
# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Teng Xue <teng.xue@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

'''
    Copyright (c) 2022 Idiap Research Institute, http://www.idiap.ch/
    Written by Teng Xue <teng.xue@idiap.ch>

    This file is the dynamics part of the paper:
    Xue T, Girgin H, Lembono T S, and Calinon S. Demonstration-guided Optimal Control
    for Long-term Non-prehensile Planar Manipulation[C], submitted to ICRA 2023

    explicit dynamics version for planar pushing task, where the
    different contact modes (sticking, sliding up, sliding down, separation and
    face-switching mechanism are formulated in the forward dynamics function as
    conditions. This formulation removes the hard constraints required by the
    implicit version.)
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time, os
from scipy import integrate
import math
import pdb
from celluloid import Camera

torch.set_default_dtype(torch.float64)

# from config import *
"""
Configuration
"""
SLIDER_R = .12 / 2
PUSHER_R = .01/2
PUSHER_X = -SLIDER_R - PUSHER_R
PUSHER_Y = 0
SLIDER_INIT = [0, 0., 0.04]
SLIDER_INIT_THETA = 0
# PUSHER_INIT = [1.3*PUSHER_X, 0, 0]#W.R.T slider-centered frame
PUSHER_ORI = [torch.pi, 0, 0]
TABLE_POS = [SLIDER_INIT[0], 0, 0]
CONTACT_TOL = 1E-3*5
DT = 0.01 # stepping time
# T = 500
VEL_BOUND = [[0, 0.05], [-0.05, 0.05]]
ACC_BOUND = [[-1, 1], [-1, 1]]
### boundary
# X_BOUND = [[-0.25, 0.25], [-0.25, 0.25], [-np.pi, np.pi], [-0.25, PUSHER_X],[-SLIDER_R, SLIDER_R]] --> [x, y, theta, px, py]
# U_BOUND = [[0, 0.05], [-0.05, 0.05]] --> Velocity
# CONTACT_FACE = {-1, -2, 1, 2}
"""
Pusher-slider-system
"""
class pusher_slider_sys():
    def __init__(self, p2p, n_switches,  state_min, state_max, p_target, dt=0.01, T= 1, device="cpu"):
        self.p2p = p2p
        self.n_switches = n_switches # times of face switching
        self.Dx = 6 # 7 #[slider_x, slider_y, slider_theta, pusher_x, pusher_y, pusher_xdot, pusher_ydot]
        self.Du = 2
        self.slider_r = torch.tensor(SLIDER_R).to(device)
        self.pusher_r = torch.tensor(PUSHER_R).to(device)
        self.pusher_hheight = 0.12 #higher height to switch face
        self.contact_tol = torch.tensor(CONTACT_TOL).to(device)  # Tolerance for contact evaluation
        self.px = (-self.slider_r - self.pusher_r).to(device)
        self.u_ps = 0.3  # Friction coefficient between pusher and slider
        self.u_gs = 0.35  # Friction  coefficient between ground and slider
        val, _ = integrate.dblquad(lambda x, y: np.sqrt(x**2 + y**2), 0, self.slider_r, 0, self.slider_r)
        
        self.c = (val / pow(self.slider_r, 2))
        self.device = device
        self.p_target = p_target
        self.state_min = state_min
        self.state_max = state_max
        self.dt = dt
        self.T = T

    def R_func(self, x):
        R_Matrix = [[torch.cos(x), -torch.sin(x)], [torch.sin(x), torch.cos(x)]]
        return torch.array(R_Matrix)

    def C_func(self, x):
        C_Matrix = [[torch.cos(x), torch.sin(x)], [-torch.sin(x), torch.cos(x)]]
        return torch.array(C_Matrix)

    def gama_t_func(self, px, py):
        u_ps = self.u_ps
        c = self.c
        gama_t = (u_ps * c ** 2 - px * py + u_ps * px ** 2) / (c ** 2 + py ** 2 - u_ps * px * py)
        return gama_t.view(-1,1)

    def gama_b_func(self, px, py):
        u_ps = self.u_ps
        c = self.c
        gama_b = (-u_ps * c ** 2 - px * py - u_ps * px ** 2) / (c ** 2 + py ** 2 + u_ps * px * py)
        return gama_b.view(-1,1)

    def Q_func(self, px, py):
        c = self.c
        Q1_M = [[c ** 2 + px ** 2, px * py], [px * py, c ** 2 + py ** 2]]
        Q2_M = c ** 2 + px ** 2 + py ** 2
        Q_Matrix = Q1_M / Q2_M
        return np.array(Q_Matrix)

    def b1_func(self, px, py):
        bs = len(py)
        out_ = torch.empty(bs,1,2).to(self.device)
        c = self.c
        out_[:,0,0] = -py / (c ** 2 + px ** 2 + py ** 2)
        out_[:,0,1] = px / (c ** 2 + px ** 2 + py ** 2)
        return out_

    def b2_func(self, px, py, gama_t):
        bs = len(py)
        out_ = torch.zeros(bs,1,2).to(self.device)
        c = self.c
        out_[:,0,0]= (-py+ gama_t.squeeze(-1) * px) / (c ** 2 + px ** 2 + py ** 2)
        return out_

    def b3_func(self, px, py, gama_b):
        bs = len(py)
        out_ = torch.zeros(bs,1,2).to(self.device)
        c = self.c
        out_[:,0,0] = (-py + gama_b.squeeze(-1) * px) / (c ** 2 + px ** 2 + py ** 2)
        return out_
    
    def forward_simulate(self,state,action,dt):
        next_state = self.dynamics(state,action,dt)
        return next_state#torch.clip(next_state,self.state_min,self.state_max)
    
    def reward_state_action(self,state,action):
        return -1*self.cost_func(state,action)

    # def dynamics(self, xs, us,dt):
    #     # faceid: -1 (left), -2 (bottom), 1 (right), 2 (top)
    #     # us = torch.cat((us[:,1:],us[:,0].view(-1,1)),dim=-1)
    #     faceid = us[:,-1].view(-1,1)
    #     faceid = (faceid-2.0)*(faceid<2)+(faceid-1.0)**(faceid>1)
    #     s_xy = xs[:,:2]
    #     s_theta = xs[:,2].view(-1,1)
    #     p_x = xs[:,3]; p_y = xs[:,4]
    #     vn = us[:, 0][:, None]
    #     vt = us[:, 1][:, None]
    #     u = 1*us[:, :2].double()
    #     bs = xs.size(0)
    #     face_beta = (torch.abs(faceid-(-1))*torch.pi/2).to(self.device) # batch x 1
    #     R_mat = torch.empty(bs,2,2).to(self.device) #np.array(self.R_func(face_beta))
    #     R_mat[:,0,0] = torch.cos(face_beta).squeeze(dim=-1)
    #     R_mat[:,-1,-1] = torch.cos(face_beta).squeeze(dim=-1)
    #     R_mat[:,0,1] = -1*torch.sin(face_beta).squeeze(dim=-1)
    #     R_mat[:,1,0] = torch.sin(face_beta).squeeze(dim=-1)

    #     R_theta = torch.empty(bs,2,2).to(self.device) #np.array(self.R_func(face_beta))
    #     R_theta[:,0,0] = torch.cos(s_theta).squeeze(dim=-1)
    #     R_theta[:,-1,-1] = torch.cos(s_theta).squeeze(dim=-1)
    #     R_theta[:,0,1] = -1*torch.sin(s_theta).squeeze(dim=-1)
    #     R_theta[:,1,0] = torch.sin(s_theta).squeeze(dim=-1)

    #     Q_mat = torch.empty(bs,2,2).to(self.device) #np.array(self.R_func(face_beta))
    #     Q_mat[:,0,0] = self.c**2 + p_x**2
    #     Q_mat[:,-1,-1] = self.c**2 + p_y**2
    #     Q_mat[:,0,1] = p_x*p_y
    #     Q_mat[:,1,0] = p_x*p_y
    #     Q_mat = Q_mat/(self.c**2+p_x**2+p_y**2)[:, None, None]
    #     """
    #     Let's assume xs[3:5] and us are defined in left face frame
    #     """
    #     # u = R_mat.T.dot(x[5:7]) # Represent u in left face frame
    #     # u = x[5:7]
    #     # x[3: 5] = R_mat.T.dot(x[3: 5]) #convert to left face frame
    #     gama_t = self.gama_t_func(self.px, p_y).to(self.device) # bs x 1
    #     gama_b = self.gama_b_func(self.px, p_y).to(self.device) # bs x 1 

    #     D = torch.zeros(bs,1,2).to(self.device)
    #     P1 = torch.eye(2).to(self.device)[None,:,:].expand(bs,-1,-1) # bs x 2 x 2
    #     P2 = torch.cat([P1[:, 0:1],torch.cat([gama_t, torch.zeros_like(gama_t)], axis=-1)[:, None]],axis=1)  # bs x 2 x 2
    #     P3 = torch.cat([P1[:, 0:1], torch.cat([gama_b, torch.zeros_like(gama_b)], axis=-1)[:, None]],axis=1) # bs x 2 x 2

    #     c1 = torch.zeros(bs,1,2).to(self.device) # bs x 1 x 2
    #     c2 = torch.cat([-gama_t, torch.ones_like(gama_t)], axis =-1)[:, None, :].to(self.device) # bs x 1 x 2
    #     c3 = torch.cat([-gama_b, torch.ones_like(gama_b)], axis=-1)[:, None, :].to(self.device) # bs x 1 x 2
    #     b1 = self.b1_func(self.px,p_y).view(-1,2)[:, None] # bs x 1 x 2
    #     b2 = self.b2_func(self.px,p_y, gama_t).view(-1,2)[:, None, :] # bs x 1 x 2
    #     b3 = self.b3_func(self.px,p_y, gama_b).view(-1,2) [:, None, :] # bs x 1 x 2
    #     # dyn1 = torch.empty(bs,7,2)
    #     dyn1 = torch.cat([torch.einsum('ijk,ikl,ilm -> ijm', R_theta, Q_mat.double(), P1),
    #                       b1, D, c1], axis=1) # bs x 5 x 2
    #     dyn2 = torch.cat([torch.einsum('ijk,ikl,ilm -> ijm', R_theta,Q_mat.double(), P2.double()),
    #                       b2, D, c2], axis=1) # bs x 5 x 2
    #     dyn3 = torch.cat([torch.einsum('ijk,ikl,ilm -> ijm', R_theta,Q_mat.double(), P3.double()),
    #                       b3, D, c3], axis=1) # bs x 5 x 2
    #     dyn4 = torch.cat([torch.zeros(bs, 3,2).to(self.device), torch.tile(torch.eye(2)[None].to(self.device), (bs, 1,1))], axis=1)

    #     cond_cone_stick = 1 * torch.logical_and(vt >= gama_b * vn, vt <= gama_t * vn) # bs x 1
    #     cond_cone_up = 1 * (vt > gama_t * vn) # bs x 1
    #     cond_cone_down = 1 * (vt < gama_b * vn) # bs x 1

    #     cond_touch = (torch.abs(p_x - self.px) <= self.contact_tol)[:, None]

    #     #Assuming vn is always >0, so that the seperation mode is only determined by the distance between pusher and slider
    #     cond1 = 1 * torch.logical_and(cond_cone_stick, cond_touch)# bs x 1
    #     cond2 = 1 * torch.logical_and(cond_cone_up, cond_touch)# bs x 1
    #     cond3 = 1 * torch.logical_and(cond_cone_down, cond_touch)# bs x 1
    #     cond4 = 1 - cond1 - cond2 - cond3# bs x 1

        
    #     # f is the system state w.r.t. global frame, but assuming pushing the initial left face
    #     f = cond1 * torch.einsum('ijk,ikl -> ijl', dyn1, u[:,:, None])[..., 0] # bs x 5
    #     f += cond2 * torch.einsum('ijk,ikl -> ijl', dyn2.double(), u[:,:, None])[..., 0] # bs x 5
    #     f += cond3 * torch.einsum('ijk,ikl -> ijl', dyn3.double(), u[:,:, None])[..., 0] # bs x 5
    #     f += cond4 * torch.einsum('ijk,ikl -> ijl', dyn4.double(), u[:,:, None])[..., 0] # bs x 5

    #     # transform the state obtained by assuming left face to current face
    #     f[:, :2][:, :, None] = torch.einsum('ijk,ikl -> ijl', R_mat, f[:, :2][:, :, None])
    #     next_state = torch.cat(((xs[:, :5]+f*dt), us[:, 0][:, None]), dim=1).to(self.device).double()
    #     next_state[:,2] = ((next_state[:,2] + torch.pi) % (2 * torch.pi)) - torch.pi # normalize theta
    #     return next_state

    def dynamics(self, xs, us,dt):
        # faceid: -1 (left), -2 (bottom), 1 (right), 2 (top)
        # us = torch.cat((us[:,1:],us[:,0].view(-1,1)),dim=-1)
        faceid = us[:,0].view(-1,1)
        faceid = (faceid-2.0)*(faceid<2)+(faceid-1.0)**(faceid>1)
        s_xy = xs[:,:2]
        s_theta = xs[:,2].view(-1,1)
        p_x = xs[:,3]; p_y = xs[:,4]
        vn = us[:, 1][:, None]
        vt = us[:, 2][:, None]
        u = 1*us[:, 1:].double()
        bs = xs.size(0)
        face_beta = (torch.abs(faceid-(-1))*torch.pi/2).to(self.device) # batch x 1
        R_mat = torch.empty(bs,2,2).to(self.device) #np.array(self.R_func(face_beta))
        R_mat[:,0,0] = torch.cos(face_beta).squeeze(dim=-1)
        R_mat[:,-1,-1] = torch.cos(face_beta).squeeze(dim=-1)
        R_mat[:,0,1] = -1*torch.sin(face_beta).squeeze(dim=-1)
        R_mat[:,1,0] = torch.sin(face_beta).squeeze(dim=-1)

        R_theta = torch.empty(bs,2,2).to(self.device) #np.array(self.R_func(face_beta))
        R_theta[:,0,0] = torch.cos(s_theta).squeeze(dim=-1)
        R_theta[:,-1,-1] = torch.cos(s_theta).squeeze(dim=-1)
        R_theta[:,0,1] = -1*torch.sin(s_theta).squeeze(dim=-1)
        R_theta[:,1,0] = torch.sin(s_theta).squeeze(dim=-1)

        Q_mat = torch.empty(bs,2,2).to(self.device) #np.array(self.R_func(face_beta))
        Q_mat[:,0,0] = self.c**2 + p_x**2
        Q_mat[:,-1,-1] = self.c**2 + p_y**2
        Q_mat[:,0,1] = p_x*p_y
        Q_mat[:,1,0] = p_x*p_y
        Q_mat = Q_mat/(self.c**2+p_x**2+p_y**2)[:, None, None]
        """
        Let's assume xs[3:5] and us are defined in left face frame
        """
        # u = R_mat.T.dot(x[5:7]) # Represent u in left face frame
        # u = x[5:7]
        # x[3: 5] = R_mat.T.dot(x[3: 5]) #convert to left face frame
        gama_t = self.gama_t_func(self.px, p_y).to(self.device) # bs x 1
        gama_b = self.gama_b_func(self.px, p_y).to(self.device) # bs x 1 

        D = torch.zeros(bs,1,2).to(self.device)
        P1 = torch.eye(2).to(self.device)[None,:,:].expand(bs,-1,-1) # bs x 2 x 2
        P2 = torch.cat([P1[:, 0:1],torch.cat([gama_t, torch.zeros_like(gama_t)], axis=-1)[:, None]],axis=1)  # bs x 2 x 2
        P3 = torch.cat([P1[:, 0:1], torch.cat([gama_b, torch.zeros_like(gama_b)], axis=-1)[:, None]],axis=1) # bs x 2 x 2

        c1 = torch.zeros(bs,1,2).to(self.device) # bs x 1 x 2
        c2 = torch.cat([-gama_t, torch.ones_like(gama_t)], axis =-1)[:, None, :].to(self.device) # bs x 1 x 2
        c3 = torch.cat([-gama_b, torch.ones_like(gama_b)], axis=-1)[:, None, :].to(self.device) # bs x 1 x 2
        b1 = self.b1_func(self.px,p_y).view(-1,2)[:, None] # bs x 1 x 2
        b2 = self.b2_func(self.px,p_y, gama_t).view(-1,2)[:, None, :] # bs x 1 x 2
        b3 = self.b3_func(self.px,p_y, gama_b).view(-1,2) [:, None, :] # bs x 1 x 2
        # dyn1 = torch.empty(bs,7,2)
        dyn1 = torch.cat([torch.einsum('ijk,ikl,ilm -> ijm', R_theta, Q_mat.double(), P1),
                          b1, D, c1], axis=1) # bs x 5 x 2
        dyn2 = torch.cat([torch.einsum('ijk,ikl,ilm -> ijm', R_theta,Q_mat.double(), P2.double()),
                          b2, D, c2], axis=1) # bs x 5 x 2
        dyn3 = torch.cat([torch.einsum('ijk,ikl,ilm -> ijm', R_theta,Q_mat.double(), P3.double()),
                          b3, D, c3], axis=1) # bs x 5 x 2
        dyn4 = torch.cat([torch.zeros(bs, 3,2).to(self.device), torch.tile(torch.eye(2)[None].to(self.device), (bs, 1,1))], axis=1)

        cond_cone_stick = 1 * torch.logical_and(vt >= gama_b * vn, vt <= gama_t * vn) # bs x 1
        cond_cone_up = 1 * (vt > gama_t * vn) # bs x 1
        cond_cone_down = 1 * (vt < gama_b * vn) # bs x 1

        cond_touch = (torch.abs(p_x - self.px) <= self.contact_tol)[:, None]

        #Assuming vn is always >0, so that the seperation mode is only determined by the distance between pusher and slider
        cond1 = 1 * torch.logical_and(cond_cone_stick, cond_touch)# bs x 1
        cond2 = 1 * torch.logical_and(cond_cone_up, cond_touch)# bs x 1
        cond3 = 1 * torch.logical_and(cond_cone_down, cond_touch)# bs x 1
        cond4 = 1 - cond1 - cond2 - cond3# bs x 1

        
        # f is the system state w.r.t. global frame, but assuming pushing the initial left face
        f = cond1 * torch.einsum('ijk,ikl -> ijl', dyn1, u[:,:, None])[..., 0] # bs x 5
        f += cond2 * torch.einsum('ijk,ikl -> ijl', dyn2.double(), u[:,:, None])[..., 0] # bs x 5
        f += cond3 * torch.einsum('ijk,ikl -> ijl', dyn3.double(), u[:,:, None])[..., 0] # bs x 5
        f += cond4 * torch.einsum('ijk,ikl -> ijl', dyn4.double(), u[:,:, None])[..., 0] # bs x 5

        # transform the state obtained by assuming left face to current face
        f[:, :2][:, :, None] = torch.einsum('ijk,ikl -> ijl', R_mat, f[:, :2][:, :, None])
        next_state = torch.cat(((xs[:, :5]+f*dt), us[:, 0][:, None]), dim=1).to(self.device).double()
        next_state[:,2] = ((next_state[:,2] + torch.pi) % (2 * torch.pi)) - torch.pi # normalize theta
        return next_state


    def forward_rollout(self, x0, us):
        
        """
        Rollout the system dynamics
        """
        if x0.shape[0] ==1:
            x0 = x0.repeat(us.shape[0], 1)

        # steps = int(self.T/self.dt)


        u_0 = torch.zeros(us.shape[0], self.Du).to(self.device)
        face_id = us[:, :self.n_switches] # bs x 4

        

        w = us[:, self.n_switches:]
        c_u = self.p2p.gen_traj_push(u_0, w) # bs x T x 2

        i_u = face_id.repeat_interleave(c_u.shape[1] // self.n_switches, dim=1).unsqueeze(-1) # bs x T x 1
        if i_u.shape[1] < c_u.shape[1]:
            padding = c_u.shape[1] - i_u.shape[1]
            i_u = torch.cat((i_u, i_u[:, -1:].repeat(1, padding, 1)), dim=1)

        cost_u = torch.linalg.norm(c_u, dim=[1, 2]) #+ torch.linalg.norm(face_id[:, 1:] - face_id[:, :-1], dim=-1)*0.01

        action = torch.cat((i_u, c_u), dim=-1)

        state = x0

        traj = state[:,:].clone()[:,None,:] #bsx(T+1)x6
        for t in range(c_u.shape[1]):
            state = self.forward_simulate(state, action[:, t], self.dt)
            traj = torch.cat((traj, state[:,None,:]), dim=1)

        cost_x  = self.cost_func_x(state)

        cost = cost_x + cost_u*0.01

        return state, cost, traj, action
    

    def cost_func_x(self, xs):
        pos_error = torch.linalg.norm(xs[:,:2]-self.p_target[:,:2], dim=-1)
        ori_error = (xs[:,2]-self.p_target[:,2]).abs()
        return pos_error + 0.1*ori_error


    def cost_func(self, xs, us, tol=0.01, scale=0.3):
        pos_error = torch.linalg.norm(xs[:,:2]-self.p_target[:,:2], dim=-1)/(tol*scale)
        ori_error = (xs[:,2]-self.p_target[:,2]).abs()/(tol*torch.pi)

        cost_state = 0.5*pos_error + 0.5*ori_error
        cost_face_switch = (xs[:, -1] != us[:, 0])
        cost_vel = torch.linalg.norm(us[:, 1:], dim=-1)
        cost = cost_state+cost_face_switch
        return cost
    
    def cost_func_v2(self, xs, us, tol=0.03, scale=0.01):
        pos_error = torch.linalg.norm(xs[:,:2]-self.p_target[:,:2], dim=-1)#/tol

        ori_error = (xs[:,2]-self.p_target[:,2]).abs()/(1*torch.pi)
        # ori_error=ori_error*torch.exp(-(pos_error*scale/tol)**1/3)
        
        fs_error = (xs[:, -1] != us[:, -1])*1/3*0.4
        vel_error = torch.linalg.norm(us[:, :2], dim=-1)*0.01
        error = pos_error#+ pos_error + fs_error + vel_error

        # # refine orientation error
        # ext_error = torch.cat([(error[:,2]%(2*torch.pi))[:, None], (error[:,2]%(2*torch.pi)-2* torch.pi)[:, None], (error[:,2]%(2*torch.pi)+2*torch.pi)[:, None]], dim=1)
        # error[:, 2], _ = torch.max(ext_error, 1)
        return pos_error#**2
    


    # def plot_planarpush(self, x_t, u_t, x_target, step_skip=1, dt=0.01, xmax=1, x_obst=[],r_obst=[],batch=False,title=None, save_as=None,figsize=3, scale=0, slider_r=0.06, animation=False):
    #     # x_t: (x,y,theta, px, py)
    #     fig = plt.figure(edgecolor=[0.1,0.1,0.1])
    #     fig.set_size_inches(figsize, figsize)

    #     # fig.patch.set_facecolor('white')
    #     # fig.patch.set_alpha(0.9)
    #     ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
    #                         xlim=(-xmax, xmax), ylim=(-xmax, xmax))

    #     if not x_obst is None:
    #         for i,x in enumerate(x_obst):
    #             circ = plt.Circle(x,r_obst[i],color='grey',alpha=0.5)
    #             ax.add_patch(circ)
    #     # if not rect_patch is None:
    #     #     rect = plt.Rectangle(rect_patch[0:2],rect_patch[2],rect_patch[3], color='c',alpha=0.5)
    #     #     ax.add_patch(rect)
        
    #     # cx = x_t[:, 0, 0]
    #     # cy = x_t[:, 0, 1]
    #     # theta = x_t[:, 0, 2]
    #     # px = x_t[:, 0, 3]
    #     # py = x_t[:, 0, 4]
    #     # R = R_func(theta).transpose(2, 0, 1) # bsx2x2
    #     # rec = np.array([[-slider_r, -slider_r, slider_r, slider_r, -slider_r], [-slider_r, slider_r, slider_r, -slider_r, -slider_r]])
    #     # msh = np.einsum('kij,jl -> kil', R, rec) + np.repeat(np.stack([cx, cy], axis=1)[:, :, None], 5, axis=2) # bsx2x5
    #     # plt_msh = msh.transpose(1, 2, 0) #2x5xbs
    #     # plt.plot(plt_msh[0,:,:], plt_msh[1,:,:], 'darkgray')
    #     # plt.plot(plt_msh[0,0:2,:], plt_msh[1,0:2,:], 'black')

    #     # #pusher plot
    #     # pusher_xys_rel = np.einsum('kil, klj -> kij',R, x_t[:, 0, 3:5][:, :, None])  #bsx2x1
    #     # pusher_xys = pusher_xys_rel + x_t[:, 0, :2][:, :, None] #bsx1x2
    #     # ax.plot(pusher_xys[:, 0, 0], pusher_xys[:, 1, 0], 'og',markersize=4)

    #     # color_list = ['y', 'g', 'b', 'm', 'orange', 'r', 'k', 'c', 'bisque', 'blueviolet', 'brown', 'darkblue', ]
    #     import matplotlib.colors as colors
    #     color_list = list(colors._colors_full_map.values())
    #     print()
    #     cx = x_t[:, :, 0] #bs x T
    #     cy = x_t[:, :, 1]
    #     theta = x_t[:, :, 2]
    #     px = x_t[:, :, 3]
    #     py = x_t[:, :, 4]

    #     def R_func(x):
    #         R_Matrix = [[np.cos(x), -np.sin(x)], [np.sin(x), np.cos(x)]]
    #         return np.array(R_Matrix)

    #     R = R_func(theta).transpose(2, 3, 0,1) # bsxTx2x2
    #     #slider plot
    #     rec = np.array([[-slider_r, -slider_r, slider_r, slider_r, -slider_r], [-slider_r, slider_r, slider_r, -slider_r, -slider_r]])
    #     msh = np.einsum('ktij,jl -> ktil', R, rec) + np.repeat(np.stack([cx, cy], axis=2)[:, :, :, None], 5, axis=3) # bsxTx2x5
    #     plt_msh = msh.transpose(2, 3, 0, 1) #2x5xbsxT

    #     camera = Camera(fig)
    #     if animation:
    #         dt=dt*step_skip
    #         interval = (1/dt)#*10**-3 # in ms


    #     #pusher plot
    #     #face switching
    #     #u_t: bs x T x 3
    #     face_angle = (u_t[:, :, -1]*np.pi/2)# bsxT
    #     R_face = R_func(theta[:, :-1]+face_angle).transpose(2, 3, 0,1) # bsxTx2x2

    #     pusher_xys_rel = np.einsum('ktil, ktlj -> ktij',R_face, x_t[:, :-1, 3:5][:, :, :, None])  #bsxTx2x1
    #     pusher_xys = pusher_xys_rel + x_t[:, :-1, :2][:, :, :, None] #bsxTx2x1

    #     T = plt_msh.shape[3]
    #     if animation: # for loop takes long time
    #         for bt in range(len(plt_msh[0][0])):
    #             for i in range(0,T-1,step_skip):
    #                 # print("plot_step", i)
    #                 alpha_ = np.clip(0.1+i*1.0/T, 0,1) if (not animation) else 1.0
    #                 # plt.plot(plt_msh[0,:,bt,::step_skip], plt_msh[1,:,bt, ::step_skip], color=((random.random(),random.random(),random.random(), 1)))
    #                 plt.plot(plt_msh[0,:,bt,i], plt_msh[1,:,bt, i], color=color_list[bt], alpha=alpha_)
    #                 plt.plot(plt_msh[0,0:2,bt, i], plt_msh[1,0:2,bt, i], 'black', linewidth=3, alpha=alpha_)
    #                 ax.scatter(pusher_xys[bt, i, 0, 0], pusher_xys[bt, i, 1, 0], c=color_list[bt], edgecolors='r', marker='o', s=50, alpha=alpha_)
    #                 camera.snap()
    #     plt.grid("True")
        
    #     # else: #not animated, plot in batch form
    #     if animation:
    #         fig2 = plt.figure(edgecolor=[0.1,0.1,0.1])
    #         fig2.set_size_inches(figsize, figsize)
    #         ax2 = fig2.add_subplot(111, aspect='equal', autoscale_on=False,
    #                             xlim=(-xmax, xmax), ylim=(-xmax, xmax))
    #         for bt in range(len(plt_msh[0][0])):
    #             # plt.plot(plt_msh[0,:,bt,::step_skip], plt_msh[1,:,bt, ::step_skip], color=color_list[bt])
    #             # plt.plot(plt_msh[0,0:2,bt, ::step_skip], plt_msh[1,0:2,bt, ::step_skip], 'black', linewidth=3)
    #             # ax.scatter(pusher_xys[bt, ::step_skip, 0, 0], pusher_xys[bt, ::step_skip, 1, 0], c=color_list[bt], edgecolors='r', marker='o', s=50)   
    #             ax2.plot(plt_msh[0,:,bt,-1], plt_msh[1,:,bt, -1], color=color_list[bt])
    #             ax2.plot(plt_msh[0,0:2,bt, -1], plt_msh[1,0:2,bt, -1], 'black', linewidth=3)
    #             ax2.scatter(pusher_xys[bt, -1, 0, 0], pusher_xys[bt, -1, 1, 0], c=color_list[bt], edgecolors='r', marker='o', s=50)   

    #     else:
    #         for bt in range(len(plt_msh[0][0])):
    #             plt.plot(plt_msh[0,:,bt,::step_skip], plt_msh[1,:,bt, ::step_skip], color=color_list[bt])
    #             plt.plot(plt_msh[0,0:2,bt, ::step_skip], plt_msh[1,0:2,bt, ::step_skip], 'black', linewidth=3)
    #             ax.scatter(pusher_xys[bt, ::step_skip, 0, 0], pusher_xys[bt, ::step_skip, 1, 0], c=color_list[bt], edgecolors='r', marker='o', s=50)   
    #             # color_rgb = color_list[bt]
    #             # ax.plot(plt_msh[0,:,bt,-1], plt_msh[1,:,bt, -1], color=color_rgb)
    #             # ax.plot(plt_msh[0,0:2,bt, -1], plt_msh[1,0:2,bt, -1], 'black', linewidth=3)
    #             # ax.scatter(pusher_xys[bt, -1, 0, 0], pusher_xys[bt, -1, 1, 0], c=color_rgb, edgecolors='r', marker='o', s=50)   

    #     # ax.plot(x_t[:,0,0],x_t[:,0,1],'og', markersize=10)
    #     ax.plot(x_target[0],x_target[1],'or', markersize=10)
        
    #     quil = 0.1 #quiver length
    #     ax.quiver(x_target[0], x_target[1], quil*np.cos(x_target[2]), quil*np.sin(x_target[2]), color='r', scale=0.5, width=0.01)

                
    #     if animation:
    #         # animation = camera.animate(interval=interval, repeat=False)
    #         animation = camera.animate()
    #         animation.save('planar_push.mp4')


    #     # ax.legend(["target","init","obstacle"])
    #     plt.grid("True")

    #     if not title is None:
    #         plt.title(title)
    #     if not save_as is None:
    #         fig.savefig(save_as+".jpeg",bbox_inches='tight', pad_inches=0.01, dpi=300)

    #     return plt


    def plot_planarpush(self, x_t, u_t, x_target, step_skip=1, dt=0.01, xmax=1, x_obst=[],r_obst=[],batch=False,title=None, 
                        save_as=None,figsize=3, scale=0, slider_r=0.06, animation=False, file_name=None):
        # x_t: (x,y,theta, px, py)
        fig = plt.figure(edgecolor=[0.1,0.1,0.1])
        fig.set_size_inches(figsize, figsize)

        # fig.patch.set_facecolor('white')
        # fig.patch.set_alpha(0.9)
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                            xlim=(-0.2, xmax), ylim=(-0.2, xmax))

        if not x_obst is None:
            for i,x in enumerate(x_obst):
                circ = plt.Circle(x,r_obst[i],color='grey',alpha=0.5)
                ax.add_patch(circ)
        # if not rect_patch is None:
        #     rect = plt.Rectangle(rect_patch[0:2],rect_patch[2],rect_patch[3], color='c',alpha=0.5)
        #     ax.add_patch(rect)
        
        # cx = x_t[:, 0, 0]
        # cy = x_t[:, 0, 1]
        # theta = x_t[:, 0, 2]
        # px = x_t[:, 0, 3]
        # py = x_t[:, 0, 4]
        # R = R_func(theta).transpose(2, 0, 1) # bsx2x2
        # rec = np.array([[-slider_r, -slider_r, slider_r, slider_r, -slider_r], [-slider_r, slider_r, slider_r, -slider_r, -slider_r]])
        # msh = np.einsum('kij,jl -> kil', R, rec) + np.repeat(np.stack([cx, cy], axis=1)[:, :, None], 5, axis=2) # bsx2x5
        # plt_msh = msh.transpose(1, 2, 0) #2x5xbs
        # plt.plot(plt_msh[0,:,:], plt_msh[1,:,:], 'darkgray')
        # plt.plot(plt_msh[0,0:2,:], plt_msh[1,0:2,:], 'black')

        # #pusher plot
        # pusher_xys_rel = np.einsum('kil, klj -> kij',R, x_t[:, 0, 3:5][:, :, None])  #bsx2x1
        # pusher_xys = pusher_xys_rel + x_t[:, 0, :2][:, :, None] #bsx1x2
        # ax.plot(pusher_xys[:, 0, 0], pusher_xys[:, 1, 0], 'og',markersize=4)

        color_list = ['y', 'g', 'b', 'm', 'orange', 'r', 'k', 'c', 'bisque', 'blueviolet', 'brown', 'darkblue', ]
        import matplotlib.colors as colors
        # color_list = list(colors._colors_full_map.values())
        print()

        def R_func(x):
            R_Matrix = [[np.cos(x), -np.sin(x)], [np.sin(x), np.cos(x)]]
            return np.array(R_Matrix)

        cx = x_t[:, :, 0] #bs x T
        cy = x_t[:, :, 1]
        theta = x_t[:, :, 2]
        px = x_t[:, :, 3]
        py = x_t[:, :, 4]
        R = R_func(theta).transpose(2, 3, 0,1) # bsxTx2x2
        #slider plot
        rec = np.array([[-slider_r, -slider_r, slider_r, slider_r, -slider_r], [-slider_r, slider_r, slider_r, -slider_r, -slider_r]])
        msh = np.einsum('ktij,jl -> ktil', R, rec) + np.repeat(np.stack([cx, cy], axis=2)[:, :, :, None], 5, axis=3) # bsxTx2x5
        plt_msh = msh.transpose(2, 3, 0, 1) #2x5xbsxT
        h2 = plt.plot(plt_msh[0,:,0, 0], plt_msh[1,:,0, 0], 'green', linewidth=5)
        camera = Camera(fig)
        if animation:
            dt=dt*step_skip
            interval = (1/dt)#*10**-3 # in ms


        #pusher plot
        #face switching
        #u_t: bs x T x 3
        face_angle = (u_t[:, :, 0]*np.pi/2)# bsxT
        R_face = R_func(theta[:, :-1]+face_angle).transpose(2, 3, 0,1) # bsxTx2x2

        pusher_xys_rel = np.einsum('ktil, ktlj -> ktij',R_face, x_t[:, :-1, 3:5][:, :, :, None])  #bsxTx2x1
        pusher_xys = pusher_xys_rel + x_t[:, :-1, :2][:, :, :, None] #bsxTx2x1

        T = plt_msh.shape[3]
        if animation: # for loop takes long time
            for bt in range(len(plt_msh[0][0])):
                for i in range(0,T-1,step_skip):
                    # print("plot_step", i)
                    alpha_ = np.clip(0.1+i*1.0/T, 0,1) if (not animation) else 1.0
                    # plt.plot(plt_msh[0,:,bt,::step_skip], plt_msh[1,:,bt, ::step_skip], color=((random.random(),random.random(),random.random(), 1)))
                    plt.plot(plt_msh[0,:,bt,i], plt_msh[1,:,bt, i], color=color_list[bt], alpha=alpha_)
                    plt.plot(plt_msh[0,0:2,bt, i], plt_msh[1,0:2,bt, i], 'black', linewidth=3, alpha=alpha_)
                    ax.scatter(pusher_xys[bt, i, 0, 0], pusher_xys[bt, i, 1, 0], c=color_list[bt], edgecolors='r', marker='o', s=50, alpha=alpha_)
                    camera.snap()
        # plt.grid("True")
        
        # else: #not animated, plot in batch form
        if animation:
            fig2 = plt.figure(edgecolor=[0.1,0.1,0.1])
            fig2.set_size_inches(figsize, figsize)
            ax2 = fig2.add_subplot(111, aspect='equal', autoscale_on=False,
                                xlim=(-xmax, xmax), ylim=(-xmax, xmax))
            for bt in range(len(plt_msh[0][0])):
                # plt.plot(plt_msh[0,:,bt,::step_skip], plt_msh[1,:,bt, ::step_skip], color=color_list[bt])
                # plt.plot(plt_msh[0,0:2,bt, ::step_skip], plt_msh[1,0:2,bt, ::step_skip], 'black', linewidth=3)
                # ax.scatter(pusher_xys[bt, ::step_skip, 0, 0], pusher_xys[bt, ::step_skip, 1, 0], c=color_list[bt], edgecolors='r', marker='o', s=50)   
                ax2.plot(plt_msh[0,:,bt,-1], plt_msh[1,:,bt, -1], color=color_list[bt])
                ax2.plot(plt_msh[0,0:2,bt, -1], plt_msh[1,0:2,bt, -1], 'black', linewidth=3)
                ax2.scatter(pusher_xys[bt, -1, 0, 0], pusher_xys[bt, -1, 1, 0], c=color_list[bt], edgecolors='r', marker='o', s=50)   

        else:
            for bt in range(len(plt_msh[0][0])):
                # t = np.linspace(0, 1, plt_msh.shape[3])
                # for i in range(0, plt_msh.shape[3] - 1, step_skip):

                #     plt.plot(plt_msh[0, :, bt, i], plt_msh[1, :,bt,i], color=plt.cm.get_cmap('plasma')(t[i]), linewidth=4)
                #     ax.scatter(pusher_xys[bt, i, 0, 0], pusher_xys[bt, i, 1, 0], c='r', edgecolors='r', marker='o', s=50)   
                plt.plot(plt_msh[0,:,bt,::step_skip], plt_msh[1,:,bt, ::step_skip], color=color_list[bt])
                plt.plot(plt_msh[0,0:2,bt, ::step_skip], plt_msh[1,0:2,bt, ::step_skip], 'black', linewidth=3)
                ax.scatter(pusher_xys[bt, ::step_skip, 0, 0], pusher_xys[bt, ::step_skip, 1, 0], c=color_list[bt], edgecolors='r', marker='o', s=50)   
                # color_rgb = color_list[bt]
                # ax.plot(plt_msh[0,:,bt,-1], plt_msh[1,:,bt, -1], color=color_rgb)
                # ax.plot(plt_msh[0,0:2,bt, -1], plt_msh[1,0:2,bt, -1], 'black', linewidth=3)
                # ax.scatter(pusher_xys[bt, -1, 0, 0], pusher_xys[bt, -1, 1, 0], c=color_rgb, edgecolors='r', marker='o', s=50)   

        # ax.plot(x_t[:,0,0],x_t[:,0,1],'og', markersize=10)
        # ax.plot(x_t[:, :, 0],x_t[:,:,1],'ok', alpha=0.6, markersize=10)
        t = np.linspace(0, 1, x_t.shape[1])
        cmap = 'turbo'  # 'plasma', 'coolwarm', 'inferno', 'magma', 'cividis'， ‘Greys’
        for i in range(x_t.shape[1] - 1):
            plt.plot(x_t[0, i:i+2, 0], x_t[0, i:i+2,1], color=plt.cm.get_cmap(cmap)(t[i]), linewidth=4)

        h1 = ax.plot(x_target[0],x_target[1],'or', markersize=10)
        
        quil = 0.07 #quiver length
        ax.quiver(x_target[0], x_target[1], quil*np.cos(x_target[2]), quil*np.sin(x_target[2]), color='r', scale=0.5, width=0.008)

        
                
        if animation:
            # animation = camera.animate(interval=interval, repeat=False)
            animation = camera.animate()
            animation.save(file_name+'.mp4')


        ax.legend([h1[0], h2[0]], ["Target","Init"], loc='upper right', fontsize=18)

        ax.set_xlabel('x (m)', fontsize=18)
        ax.set_ylabel('y (m)', fontsize=18)
        ax.tick_params(axis='both', labelsize=18)


        # plt.grid("True")

        if not title is None:
            plt.title(title)
        if not save_as is None:
            fig.savefig(save_as+".jpeg",bbox_inches='tight', pad_inches=0.01, dpi=300)

        return plt


    def plot_planarpush_v2(self, x_t, u_t, x_target, step_skip=1, dt=0.01, xmax=1, x_obst=[],r_obst=[],batch=False,title=None, 
                        save_as=None,figsize=3, scale=0, slider_r=0.06, animation=False, file_name=None):
        # x_t: (x,y,theta, px, py)
        fig = plt.figure(edgecolor=[0.1,0.1,0.1])
        fig.set_size_inches(figsize, figsize)

        # fig.patch.set_facecolor('white')
        # fig.patch.set_alpha(0.9)
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                            xlim=(-0.2, xmax), ylim=(-0.2, xmax))

        if not x_obst is None:
            for i,x in enumerate(x_obst):
                circ = plt.Circle(x,r_obst[i],color='grey',alpha=0.5)
                ax.add_patch(circ)
        # if not rect_patch is None:
        #     rect = plt.Rectangle(rect_patch[0:2],rect_patch[2],rect_patch[3], color='c',alpha=0.5)
        #     ax.add_patch(rect)
        
        # cx = x_t[:, 0, 0]
        # cy = x_t[:, 0, 1]
        # theta = x_t[:, 0, 2]
        # px = x_t[:, 0, 3]
        # py = x_t[:, 0, 4]
        # R = R_func(theta).transpose(2, 0, 1) # bsx2x2
        # rec = np.array([[-slider_r, -slider_r, slider_r, slider_r, -slider_r], [-slider_r, slider_r, slider_r, -slider_r, -slider_r]])
        # msh = np.einsum('kij,jl -> kil', R, rec) + np.repeat(np.stack([cx, cy], axis=1)[:, :, None], 5, axis=2) # bsx2x5
        # plt_msh = msh.transpose(1, 2, 0) #2x5xbs
        # plt.plot(plt_msh[0,:,:], plt_msh[1,:,:], 'darkgray')
        # plt.plot(plt_msh[0,0:2,:], plt_msh[1,0:2,:], 'black')

        # #pusher plot
        # pusher_xys_rel = np.einsum('kil, klj -> kij',R, x_t[:, 0, 3:5][:, :, None])  #bsx2x1
        # pusher_xys = pusher_xys_rel + x_t[:, 0, :2][:, :, None] #bsx1x2
        # ax.plot(pusher_xys[:, 0, 0], pusher_xys[:, 1, 0], 'og',markersize=4)

        color_list = ['y', 'g', 'b', 'm', 'orange', 'r', 'k', 'c', 'bisque', 'blueviolet', 'brown', 'darkblue', ]
        import matplotlib.colors as colors
        # color_list = list(colors._colors_full_map.values())
        print()

        def R_func(x):
            R_Matrix = [[np.cos(x), -np.sin(x)], [np.sin(x), np.cos(x)]]
            return np.array(R_Matrix)

        cx = x_t[:, :, 0] #bs x T
        cy = x_t[:, :, 1]
        theta = x_t[:, :, 2]
        px = x_t[:, :, 3]
        py = x_t[:, :, 4]
        R = R_func(theta).transpose(2, 3, 0,1) # bsxTx2x2
        target_R = R_func(x_target[2])[None, None, :, :] # bsxTx2x2
        #slider plot
        rec = np.array([[-slider_r, -slider_r, slider_r, slider_r, -slider_r], [-slider_r, slider_r, slider_r, -slider_r, -slider_r]])
        msh = np.einsum('ktij,jl -> ktil', R, rec) + np.repeat(np.stack([cx, cy], axis=2)[:, :, :, None], 5, axis=3) # bsxTx2x5
        plt_msh = msh.transpose(2, 3, 0, 1) #2x5xbsxT

        a = np.einsum('ktij,jl -> ktil', target_R, rec)
        target_msh = np.einsum('ktij,jl -> ktil', target_R, rec) + np.repeat(np.stack([x_target[0], x_target[1]])[None, None, :, None], 5, axis=3) # bsxTx2x5
        target_msh = target_msh.transpose(2, 3, 0, 1) #2x5xbsxT


        h2 = plt.plot(plt_msh[0,:,0, 0], plt_msh[1,:,0, 0], 'green', linewidth=5)
        camera = Camera(fig)
        if animation:
            dt=dt*step_skip
            interval = (1/dt)#*10**-3 # in ms


        #pusher plot
        #face switching
        #u_t: bs x T x 3
        face_angle = (u_t[:, :, 0]*np.pi/2)# bsxT
        R_face = R_func(theta[:, :-1]+face_angle).transpose(2, 3, 0,1) # bsxTx2x2

        pusher_xys_rel = np.einsum('ktil, ktlj -> ktij',R_face, x_t[:, :-1, 3:5][:, :, :, None])  #bsxTx2x1
        pusher_xys = pusher_xys_rel + x_t[:, :-1, :2][:, :, :, None] #bsxTx2x1

        T = plt_msh.shape[3]
        if animation: # for loop takes long time
            for bt in range(len(plt_msh[0][0])):
                for i in range(0,T-1,step_skip):
                    # print("plot_step", i)
                    alpha_ = np.clip(0.1+i*1.0/T, 0,1) if (not animation) else 1.0
                    # plt.plot(plt_msh[0,:,bt,::step_skip], plt_msh[1,:,bt, ::step_skip], color=((random.random(),random.random(),random.random(), 1)))
                    plt.plot(plt_msh[0,:,bt,i], plt_msh[1,:,bt, i], color=color_list[bt], alpha=alpha_)
                    plt.plot(plt_msh[0,0:2,bt, i], plt_msh[1,0:2,bt, i], 'black', linewidth=3, alpha=alpha_)
                    ax.scatter(pusher_xys[bt, i, 0, 0], pusher_xys[bt, i, 1, 0], c=color_list[bt], edgecolors='r', marker='o', s=50, alpha=alpha_)
                    
                    h1 = plt.plot(target_msh[0,:,bt, -1], target_msh[1,:,bt, -1], color='red', alpha=0.7, linewidth=5, label='Target')
                    plt.plot(target_msh[0,0:2,bt, -1], target_msh[1,0:2,bt, -1], 'black',  alpha=0.7, linewidth=5)
                    camera.snap()
        # plt.grid("True")
        
        # else: #not animated, plot in batch form
        if animation:
            fig2 = plt.figure(edgecolor=[0.1,0.1,0.1])
            fig2.set_size_inches(figsize, figsize)
            ax2 = fig2.add_subplot(111, aspect='equal', autoscale_on=False,
                                xlim=(-xmax, xmax), ylim=(-xmax, xmax))
            for bt in range(len(plt_msh[0][0])):
                # plt.plot(plt_msh[0,:,bt,::step_skip], plt_msh[1,:,bt, ::step_skip], color=color_list[bt])
                # plt.plot(plt_msh[0,0:2,bt, ::step_skip], plt_msh[1,0:2,bt, ::step_skip], 'black', linewidth=3)
                # ax.scatter(pusher_xys[bt, ::step_skip, 0, 0], pusher_xys[bt, ::step_skip, 1, 0], c=color_list[bt], edgecolors='r', marker='o', s=50)   
                ax2.plot(plt_msh[0,:,bt,-1], plt_msh[1,:,bt, -1], color=color_list[bt])
                ax2.plot(plt_msh[0,0:2,bt, -1], plt_msh[1,0:2,bt, -1], 'black', linewidth=3)
                ax2.scatter(pusher_xys[bt, -1, 0, 0], pusher_xys[bt, -1, 1, 0], c=color_list[bt], edgecolors='r', marker='o', s=50)   

        else:
            for bt in range(len(plt_msh[0][0])):

                # t = np.linspace(0, 1, plt_msh.shape[3])
                # for i in range(0, plt_msh.shape[3] - 1, step_skip):

                #     plt.plot(plt_msh[0, :, bt, i], plt_msh[1, :,bt,i], color=plt.cm.get_cmap('plasma')(t[i]), linewidth=4)
                #     ax.scatter(pusher_xys[bt, i, 0, 0], pusher_xys[bt, i, 1, 0], c='r', edgecolors='r', marker='o', s=50)   
                plt.plot(plt_msh[0,:,bt,::step_skip], plt_msh[1,:,bt, ::step_skip], color=color_list[bt])
                plt.plot(plt_msh[0,0:2,bt, ::step_skip], plt_msh[1,0:2,bt, ::step_skip], 'black', linewidth=3)
                ax.scatter(pusher_xys[bt, ::step_skip, 0, 0], pusher_xys[bt, ::step_skip, 1, 0], c=color_list[bt], edgecolors='r', marker='o', s=50)   
                # color_rgb = color_list[bt]

                # target drawing
                h1 = plt.plot(target_msh[0,:,bt, -1], target_msh[1,:,bt, -1], color='red', alpha=0.7, linewidth=5, label='Target')
                plt.plot(target_msh[0,0:2,bt, -1], target_msh[1,0:2,bt, -1], 'black',  alpha=0.7, linewidth=5)


                # ax.scatter(pusher_xys[bt, -1, 0, 0], pusher_xys[bt, -1, 1, 0], c=color_rgb, edgecolors='r', marker='o', s=50)   

        # ax.plot(x_t[:,0,0],x_t[:,0,1],'og', markersize=10)
        # ax.plot(x_t[:, :, 0],x_t[:,:,1],'ok', alpha=0.6, markersize=10)
        t = np.linspace(0, 1, x_t.shape[1])
        cmap = 'turbo'  # 'plasma', 'coolwarm', 'inferno', 'magma', 'cividis'， ‘Greys’
        for i in range(x_t.shape[1] - 1):
            plt.plot(x_t[0, i:i+2, 0], x_t[0, i:i+2,1], color=plt.cm.get_cmap(cmap)(t[i]), linewidth=4)

        ax.plot(x_target[0],x_target[1],'or', markersize=10)

        
        # quil = 0.07 #quiver length
        # ax.quiver(x_target[0], x_target[1], quil*np.cos(x_target[2]), quil*np.sin(x_target[2]), color='r', scale=0.5, width=0.008)
        

                
        if animation:
            # animation = camera.animate(interval=interval, repeat=False)
            animation = camera.animate()
            animation.save(file_name+'.mp4')


        ax.legend([h1[0], h2[0]], ["Target","Init"], loc='upper right', fontsize=18)

        ax.set_xlabel('x (m)', fontsize=18)
        ax.set_ylabel('y (m)', fontsize=18)
        ax.tick_params(axis='both', labelsize=18)


        # plt.grid("True")

        if not title is None:
            plt.title(title)
        if not save_as is None:
            fig.savefig(save_as+".jpeg",bbox_inches='tight', pad_inches=0.01, dpi=300)

        return plt

