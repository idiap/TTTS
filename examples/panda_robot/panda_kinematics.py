#
# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Teng Xue <teng.xue@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import torch
import math
torch.set_default_dtype(torch.float64)

class PandaKinematics:
    '''
    Modified Franka-Emika Panda forward kinematics:
      - 7 DH parameters
      - Base T0 = identity matrix
      - No additional 8th joint
      - dh_d[-1] = 0.107 corresponds to the link7 to flange distance (if -45° is needed, multiply afterward)
    '''

    def __init__(self, device="cpu", key_points_data=None):
        self.device = device
        self.n_joints = 7

        # Official Franka joint limits (similar to MuJoCo)
        self.theta_max_robot = torch.tensor([
            2.8973, 1.7628, 2.5, 
            -0.0698, 2.8973, 3.7525, 
            2.8973
        ], dtype=torch.float64).to(device)
        self.theta_min_robot = torch.tensor([
            -2.8973, -1.7628, -2.5,
            -3.0718, -2.8973, -0.0175,
            -2.8973
        ], dtype=torch.float64).to(device)

        # A bit of safety margin; adjust as needed
        self.theta_max = self.theta_max_robot - 0.2
        self.theta_min = self.theta_min_robot + 0.2

        self.max_config = self.theta_max.reshape(1,-1).to(device)
        self.min_config = self.theta_min.reshape(1,-1).to(device)

        # DH parameters: 7 sets (up to link7)
        # According to common references: a, alpha, d can differ slightly
        # The following is just an example; you can adjust based on your URDF
        self.dh_a = torch.tensor([
            0.0,     0.0,     0.0,
            0.0825, -0.0825,  0.0,
            0.088
        ], dtype=torch.float64).to(device)

        self.dh_d = torch.tensor([
            0.333,   0.0,     0.316,
            0.0,     0.384,   0.0,
            0.107+0.105   # Note: In MuJoCo, link7->hand also has 0.107; included here for simplicity
        ], dtype=torch.float64).to(device)

        # alpha_i * pi/2
        # Must be consistent with actual hardware or URDF coordinates
        self.dh_alpha = (torch.pi/2) * torch.tensor([
            0, -1,  1,
            1, -1,  1,
            1
        ], dtype=torch.float64).to(device)

        # =========== Keypoint data (can ignore or keep) ===========
        if key_points_data is None:
            # If collision detection is not relevant, set empty defaults
            self.key_points = torch.zeros(7,1,3, dtype=torch.float64).to(device)
            self.key_points_weight = torch.ones(7,1,1, dtype=torch.float64).to(device) / 8
            self.key_points_margin = 0.1*torch.ones(7,1,1, dtype=torch.float64).to(device)
        else:
            self.key_points = key_points_data[0].to(self.device).double()
            self.key_points_weight = key_points_data[1].to(device).double()
            self.key_points_margin = key_points_data[2].to(device).double()

        self.n_kp = self.key_points.shape[1]

        # =========== Build constant DH transform T_prod ===========
        ca = torch.cos(self.dh_alpha)
        sa = torch.sin(self.dh_alpha)

        # Talpha: rotation about x-axis by alpha_i
        Talpha = torch.eye(4,4, dtype=torch.float64).reshape(1,4,4).repeat(len(self.dh_alpha),1,1)
        Talpha[:,1,1] = ca
        Talpha[:,1,2] = -sa
        Talpha[:,2,1] = sa
        Talpha[:,2,2] = ca

        # Ta: translation along x-axis, a_i
        Ta = torch.eye(4,4, dtype=torch.float64).reshape(1,4,4).repeat(len(self.dh_a),1,1)
        Ta[:,0,3] = self.dh_a

        # Td: translation along z-axis, d_i
        Td = torch.eye(4,4, dtype=torch.float64).reshape(1,4,4).repeat(len(self.dh_d),1,1)
        Td[:,2,3] = self.dh_d

        # Final T_prod = Talpha * Ta * Td (the constant part for each joint)
        self.T_prod = torch.einsum('ijk,ikl,ilm->ijm', Talpha, Ta, Td).to(device)

        # =========== Base transform (T0 = identity) ===========
        # In MuJoCo, link0 has no extra pos or quat => aligned with the world frame
        T0 = torch.eye(4,4, dtype=torch.float64).reshape(1,1,4,4).to(device)
        self.T0 = T0.clone()

    def set_device(self, device):
        self.device = device

    # --------------------------------------
    #          Forward Kinematics
    # --------------------------------------
    def forward_kin(self, q):
        '''
          Given joint angles q (batch, 7),
          Returns:
            - key_positions: (batch, joints=7, key_points_per_joint, 3)
            - ee_position: (batch, 3)
            - ee_orientation: (batch, 3x3) rotation matrix
        '''
        self.T = self.computeTransformation(q)  # (batch, 7, 4, 4)
        self.key_positions = self.getKeyPosition() 
        self.ee_position = self.key_positions[:,-1,-1,:]  # Take the last key point of the last joint (usually end-effector)
        self.ee_orientation = self.getEndPose() 
        return  self.key_positions, self.ee_position, self.ee_orientation  

    def computeTransformation(self, q):
        '''
          Given (batch, 7) joint angles, return (batch, 7, 4, 4),
          where T[:, i] is the transform from the base to the (i+1)-th joint
        '''
        q = q.to(self.device, dtype=torch.float64)
        batch_size = q.shape[0]

        # 1) Build rotation matrices Rz(q_i) (batch,7,4,4)
        Tz = torch.eye(4,4, dtype=torch.float64).to(self.device).reshape(1,1,4,4)
        Tz = Tz.repeat(batch_size, self.n_joints, 1, 1)

        # Fill the rotational part
        cq = torch.cos(q)
        sq = torch.sin(q)
        Tz[:,:,0,0] = cq
        Tz[:,:,0,1] = -sq
        Tz[:,:,1,0] = sq
        Tz[:,:,1,1] = cq

        # 2) Multiply by T_prod (constant DH) => total transform for each joint
        #    If your DH formula is Rz(q) then translations, you'd do Tz @ T_prod instead
        T_joints = torch.einsum('jkl, ijlm -> ijkm', self.T_prod, Tz)  # (batch,7,4,4)

        # 3) Sequentially multiply from the base (4x4 identity) => base->each joint
        T_all = torch.eye(4,4, dtype=torch.float64).to(self.device).unsqueeze(0).unsqueeze(1)
        T_all = T_all.repeat(batch_size, self.n_joints+1, 1, 1)  # shape: (batch, 8, 4,4)
        T_all[:,0,:,:] = self.T0[:,0,:,:]  # 0th transform = base

        for i in range(self.n_joints):
            T_all[:, i+1, :, :] = torch.matmul(T_all[:, i, :, :], T_joints[:, i, :, :])

        # Return the 7 transforms (batch,7,4,4)
        return T_all[:,1:,:,:]

    def getKeyPosition(self):
        '''
          Extract the 3D positions of each joint from self.T,
          then add local offsets from key_points_data (if any).
          Return shape: (batch, 7, M, 3)
        '''
        # Get translation for 7 joints
        x_joint = self.T[:, :, :3, 3]  # (batch,7,3)
        R_joint = self.T[:, :, :3, :3] # (batch,7,3,3)

        x_key = x_joint.view(x_joint.shape[0], x_joint.shape[1], 1, x_joint.shape[2]) \
               + torch.einsum('ijpr,jkr->ijkp', R_joint, self.key_points)
        return x_key

    def getEndPose(self):
        '''
          Return the end-effector rotation matrix, shape=(batch,3,3)
        '''
        # End-effector corresponds to self.T[:, -1]
        R = self.T[:, -1, :3, :3].clone()
        return R

    def getEndPoseEuler(self):
        '''
          Return the end-effector pose (batch,6): position + Euler angles, and rotation matrix R
        '''
        x = self.T[:, -1, :3, 3]
        R = self.T[:, -1, :3, :3]
        # Using ZYX convention (similar to the original):
        sy = torch.sqrt(R[:,0,0]*R[:,0,0] + R[:,1,0]*R[:,1,0])
        singular = sy < 1e-6

        # Two possible solutions
        t1 = torch.stack([
            torch.atan2(R[:,2,1], R[:,2,2]),
            torch.atan2(-R[:,2,0], sy),
            torch.atan2(R[:,1,0], R[:,0,0])
        ], dim=1)

        t2 = torch.stack([
            torch.atan2(-R[:,1,2], R[:,1,1]),
            torch.atan2(-R[:,2,0], sy),
            torch.zeros_like(sy)
        ], dim=1)

        ts = torch.where(singular.unsqueeze(1), t2, t1)
        euler_xyz = ts  # shape=(batch,3)
        pose = torch.cat([x, euler_xyz], dim=1) # (batch,6)

        return pose, R


#  # ---------------------------------------------
#     def hand_transform(self, T_link7):
#         """
#         MuJoCo:
#         <body name="hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">
#         This quaternion corresponds to a -45° rotation around the Z-axis,
#         not around the X-axis.
        
#         So we do:
#         1) Translate +0.107 in z
#         2) Rotate -45° around z
#         """
#         # Step 1: +0.107 along z
#         T_offset = torch.eye(4,4, dtype=torch.float64).to(self.device)
#         T_offset[2,3] = 0.107

#         # Step 2: -45° around Z (NOT around X)
#         ang = math.pi/4
#         Rz = torch.tensor([
#             [ math.cos(ang), -math.sin(ang), 0, 0],
#             [ math.sin(ang),  math.cos(ang), 0, 0],
#             [ 0,              0,             1, 0],
#             [ 0,              0,             0, 1],
#         ], dtype=torch.float64).to(self.device)

#         # Combine: link7 -> (translate) -> (rotate about Z)
#         T_hand = torch.matmul(T_link7, T_offset)
#         T_hand = torch.matmul(T_hand, Rz)
#         return T_hand


#     def hand_to_grasping_target_transform(self, T_hand):
#         '''
#           <body name="grasping_target_hand" pos="0 0 0.105" />
#           => an additional 0.105m along z from "hand" to "grasping_target_hand"
#         '''
#         T_offset = torch.eye(4,4, dtype=torch.float64).to(self.device)
#         T_offset[2,3] = 0.105
#         T_gt = torch.matmul(T_hand, T_offset)
#         return T_gt

#     # -------------- Final "grasping_target" chain --------------
#     def forward_kin_grasping_target(self, q):
#         '''
#           Returns the transform from base -> grasping_target_hand
#         '''
#         # 1) base -> link7
#         T_all = self.computeTransformation(q)   # shape=(batch,7,4,4)
#         T_link7 = T_all[:, -1]                  # (batch,4,4)

#         # 2) link7 -> hand
#         T_hand = self.hand_transform(T_link7)

#         # 3) hand -> grasping_target_hand
#         T_gt   = self.hand_to_grasping_target_transform(T_hand)
#         return T_gt  # (batch,4,4)

#     def get_grasping_target_pose(self, q):
#         '''
#           Return (pos, R) for "grasping_target_hand".
#         '''
#         T_gt = self.forward_kin_grasping_target(q)
#         pos = T_gt[:, :3, 3]
#         R = T_gt[:, :3, :3]
#         return pos, R