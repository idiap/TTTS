#
# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Teng Xue <teng.xue@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import numpy as np
import matplotlib.pyplot as plt
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BasisModel:
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
    def __init__(self, dim_x, dim_w, x_min, x_max, T, K, basis="rbf", bounds=None, device="cpu"):
        self.device = device
        self.dim_x = dim_x # dimension of the original function domain, not after reduction
        self.dim_w = dim_w # dimension of the weight matrix domain
        # # (e.g., if x_dim=8, and we use one dim of basis functions to apprx. 4 vars, then n=2)
        self.T = T # number of decision variables per dimension
        self.t = torch.linspace(0,1,self.T).to(device) # phase
        self.K = K # number of basis functions
        self.basis = basis
        self.x_min = x_min
        self.x_max = x_max
        if basis == "rbf":
            self.Phi = self.Phi_rbf().to(device)
        elif basis == "rbf2":
            self.Phi = self.Phi_rbf2().to(device)
        elif basis == "bs":
            self.Phi = self.Phi_Bs().to(device)
        self.set_bound(bounds) # bounds is either None (no limit) or a list containing lower and upper bound
        self.device=device

    def set_bound(self, bounds):
        if bounds is None:
            bounds=[]
            bounds.append(torch.tensor([-10**5]*self.dim_x).to(self.device)) # lower bound
            bounds.append(-1*bounds[0])
        self.lower_bound = bounds[0].reshape(1,1,-1)  # lower limit on the trajectory
        self.upper_bound = bounds[1].reshape(1,1,-1) # upper limit on the trajectory


    def Phi_rbf(self): #RBF
        t = torch.linspace(0,1,self.T).to(self.device)
        r_rbf = 1.5/(self.K) # radius
        c_rbf = torch.linspace(0,1,self.K+2).to(self.device)[1:-1] # centers
        Phi = torch.empty((self.T,self.K)).to(self.device)
        for k in range(self.K):
            Phi[:,k]=torch.exp(-(t-c_rbf[k])**2/r_rbf**2)
        return Phi

    
    def Phi_Bs(self):  # Corrected Bernstein Polynomial
        t = torch.linspace(0, 1, self.T)
        Phi = torch.zeros((self.T, self.K))
        for k in range(self.K):
            b = np.math.factorial(self.K-1) / (np.math.factorial(self.K-1-k)*np.math.factorial(k))
            Phi[:, k] = b * ((1-t)**(self.K-1-k)) * (t**k)
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

    def gen_traj_p2p(self, x_0, x_f, w):
        ''' 
            generate trajectory with boundary conditions satisfied
            x_0: batch x n, initial state
            x_f: batc x n, final state
            w: batch x (K*n), weights of basis function, the 
        '''
        batch_size = w.shape[0]
        x_0 = x_0.repeat(w.shape[0],1) # batch x dim_x
        x_f = x_f.repeat(w.shape[0],1)  # batch x dim_x

        x_0 = x_0.reshape(batch_size,1,self.dim_w).repeat(1,self.T,1) #batch x time x n
        x_f = x_f.reshape(batch_size,1,self.dim_w).repeat(1,self.T,1) #batch x time x n
        w = w.reshape(batch_size,self.K,self.dim_w) #weights
        z_t = torch.einsum('jk,ikl->ijl',self.Phi,w) # batch x time x n
        z_0 = z_t[:,0,:][:,None,:]
        z_f = z_t[:,-1,:][:,None,:]
        x_t = x_0 + z_t - z_0 + torch.einsum('j,ijk->ijk',self.t, x_f-x_0+z_0-z_f) # x(t) = x(0)+ z(t)-z(0)+t*(x(1)-x(0)+z(0)-z(1))
        # x_t_bounded = self.bound_traj(x_t)  # clip the trajectory to maintain the upper and lower limits
        # return x_t_bounded # (batch_size,self.T,self.n)

        return x_t
    
    def get_x_from_w(self, w):
        ''' 
            generate x given basis function and the weights
            x_0: batch x n, initial state
            x_f: batc x n, final state
            w: batch x (K*n), weights of basis function, the 
        '''
        batch_size = w.shape[0]
        w = w.reshape(batch_size, self.K, self.dim_w) #weights

        x_t = torch.einsum('jk,ikl->ijl',self.Phi, w).to(self.device) # batch x time x n
        # clip x_t given x_min and x_max
        x_t = torch.clip(x_t, self.x_min, self.x_max)
        return x_t

 
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
    
if __name__ == "__main__":

    # Define parameters for the trajectory generator
    dim_x = 2  # Dimension of the space
    dim_w = 2  # Dimension of the weight vector
    x_min = torch.tensor([-1.0, -1.0])  # Minimum boundary
    x_max = torch.tensor([1.0, 1.0])   # Maximum boundary
    T = 100  # Number of time steps in the trajectory
    K = 3  # Number of basis functions
    basis = "bs"  # Type of basis function

    # Initialize the BasisModel
    model = BasisModel(dim_x=dim_x, dim_w=dim_w, x_min=x_min, x_max=x_max, T=T, K=K, basis=basis)

    # Define grid dimensions
    num_rows, num_cols = 3, 3  # Grid size
    num_tasks = num_rows * num_cols  # Total number of tasks
    num_trajectories_per_task = 10  # Number of trajectories per task
    x_0 = torch.tensor([[0.0, 0.0]])  # Start point

    # Generate random end points for each task
    x_f_list = torch.rand((num_tasks, dim_x)) * 2 - 1  # Random end points in [-1, 1]

    # Generate random weights for all tasks and trajectories
    w = torch.randn(num_tasks * num_trajectories_per_task, K * dim_w) * 0.1  # Random weights

    # Generate trajectories for all tasks, each with multiple trajectories
    trajectories = []  # List to hold all trajectories
    for i in range(num_tasks):
        x_f = x_f_list[i].unsqueeze(0)  # Single end point
        task_trajectories = []
        for j in range(num_trajectories_per_task):
            idx = i * num_trajectories_per_task + j  # Index for weight
            traj = model.gen_traj_p2p(x_0, x_f, w[idx:idx+1]).squeeze(0).cpu().detach().numpy()
            task_trajectories.append(traj)
        trajectories.append((task_trajectories, x_f_list[i].numpy()))  # Store all trajectories for this task

    # Plot trajectories in a grid
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))

    # Set black background for all subplots
    for ax in axs.flat:
        ax.set_facecolor('k')


    # Plot each task's trajectories
    for idx, ax in enumerate(axs.flat):
        if idx < len(trajectories):
            task_trajectories, x_f = trajectories[idx]
            for traj in task_trajectories:
                # Color gradient based on time
                t_steps = np.linspace(0, 1, traj.shape[0])
                colors = [(1-t, t, 0) for t in t_steps]  # Gradient from red to green
                for k in range(len(traj) - 1):
                    ax.plot(traj[k:k+2, 0], traj[k:k+2, 1], color=colors[k], lw=1, alpha=0.6)
            # Mark start and end points
            ax.scatter(x_0[0, 0], x_0[0, 1], color='white', marker='o', s=100, label='Start', zorder=5)
            ax.scatter(x_f[0], x_f[1], color='white', marker='x', s=100, zorder=5)
            ax.set_xlim([-1.5, 1.5])
            ax.set_ylim([-1.5, 1.5])
            ax.set_xticks([])
            ax.set_yticks([])
            # ax.set_title(f"Task {idx + 1}", color='white')

    plt.tight_layout()
    plt.show()


    
