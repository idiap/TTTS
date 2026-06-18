#
# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Teng Xue <teng.xue@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import torch
import sys, os
from termcolor import colored
import numpy as np
import time

cur_path = os.path.dirname(__file__)
sys.path.append(cur_path)

par_path = os.path.abspath(os.path.join(cur_path, os.pardir))
sys.path.append(par_path)

from tt_utils import guided_cross_approximate, ucb_select, cross_approximate, deterministic_top_k_sol, get_conditioned_tt
from tt_utils import get_value_from_cores, stochastic_top_k_sol, fcn_batch_limited
import tntorch as tnt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



import logging
logging.basicConfig(level=logging.ERROR)

class TTTS:
    def __init__(self, func, domain, max_tt = 50, value_k= 5, visit_k= 5, num_mcts_sample=10**3,
                 num_mcts_batch =20, kickrank=10, cross_max_iter=5, 
                 max_batch=10**3, val_size=10**3, save_path = "manipulator_3D",
                 warm_start = True, verbose=True, test_mode=False, task_var=None):
        """
        Initialize the TTTS class.

        Parameters:
        - func: The target function to optimize.
        - domain: The discrete domain for the function.
        - K: Number of points to sample in each iteration.
        """
        self.func = func
        self.domain = [x.to(device) for x in domain]
        self.aug_domain = [x.to(device) for x in domain]
        self.dim = len(domain)
        self.dim_state = torch.tensor([len(x) for x in domain]).to(device)
        self.aug_dim_state = torch.tensor([len(x) for x in domain]).to(device)
        self.visit_k = visit_k
        self.value_k = value_k
        self.updated_flag = True
        self.verbose = verbose
        self.warm_start = warm_start

        self.num_mcts_sim = num_mcts_sample
        self.num_mcts_batch = num_mcts_batch # batch size for parallel mcts

        self.rmax_visit_tt = 50 # max tt for visit model to truncate

        self.domain_tensors = tnt.meshgrid(domain) #domain tt for cross approximation

        self.free_block_dim = 2 # for block indices visiting
        self.ts_ranks_tt = max(self.value_k, self.free_block_dim) #the rank of delta_visit_tt is self.free_block_dim
        self.apprx_max_tt = max_tt # the maximum tt for initial cross approximation
        self.apprx_kickrank = kickrank # the kickrank for initial cross approximation
        self.apprx_max_iter = cross_max_iter # the maximum iteration for initial cross approximation

        self.max_batch = max_batch
        self.val_size = val_size

        self.reset()

        self.task_var = task_var # task variable for conditional sampling
        self.save_path = save_path

        # init tt model
        if test_mode:
            self.save_tt = False
        else:
            self.save_tt = True

        # TT model initialization
        if self.warm_start:
            self.init_tt(self.save_path, max_tt=self.apprx_max_tt, max_iter=self.apprx_max_iter)
        else:
            self.func_tt = 0*(tnt.rand(self.dim_state,ranks_tt=1).tt()).to(device)
        
        # augment_tt for mpc
        self.aug_func_tt = self.func_tt.clone()

    def reset(self):
        self.visit_x = torch.tensor([]).to(device)
        self.visit_indices = torch.tensor([]).to(device)
        self.top_k_values = torch.tensor([]).to(device)
        self.top_k_indices = torch.tensor([]).to(device)
        self.top_k_x = torch.tensor([]).to(device)

    def get_tt_cond_x(self, x):
        """
        Get the tensor train model for the function conditioned on state x.

        Parameters:
        - x: The input tensor.

        Returns:
        - The tensor train model.
        """
        tt_model = get_conditioned_tt(self.aug_func_tt.cores.copy(), x[:1], self.aug_domain, device=device)

        return tt_model

    def refined_func_test(self, x):
        """
        refined function for low-rank cross approximation, by setting already visited non-top-k values to zero.
        """
        values = self.func(x)
        # if x in self.top_k_x, return 1, elif x in self.vist_x, return 0
        # else return the original value

        x_expanded = x.unsqueeze(1)  # Shape (batch_size, 1, dim)
        top_k_expanded = self.top_k_x.unsqueeze(0)  # Shape (1, K, dim)
        matches = torch.all(x_expanded == top_k_expanded, dim=2)  # Shape (batch_size, K)
        top_k_flag = torch.any(matches, dim=1).int()



        visit_nums = get_value_from_cores(self.visit_tt.cores, x, self.domain, self.dim_state, device=device).int()
        # conver the visit_flag to binary
        visit_flag = (visit_nums > 0).int()

        # x_expanded = x.unsqueeze(1)  # Shape (batch_size, 1, dim)
        # visit_expanded = self.visit_x.unsqueeze(0)  # Shape (1, K, dim)
        # matches = torch.all(x_expanded == visit_expanded, dim=2)  # Shape (batch_size, K)
        # visit_flag = torch.any(matches, dim=1).int()

        visit_not_top_k_flag = visit_flag * (1 - top_k_flag)

        if visit_not_top_k_flag.sum() >= 1:
            print("visit_not_top_k_flag.sum()", visit_not_top_k_flag.sum())

        return values * visit_not_top_k_flag

    def refined_func(self, x):
        """
        refined function for low-rank cross approximation, by setting already visited non-top-k values to zero.
        """
        values = self.func(x)
        # if x in self.top_k_x, return 1, elif x in self.vist_x, return 0
        # else return the original value

        x_expanded = x.unsqueeze(1)  # Shape (batch_size, 1, dim)
        top_k_expanded = self.top_k_x.unsqueeze(0)  # Shape (1, K, dim)
        matches = torch.all(x_expanded == top_k_expanded, dim=2)  # Shape (batch_size, K)
        top_k_flag = torch.any(matches, dim=1).int()



        visit_nums = get_value_from_cores(self.visit_tt.cores, x, self.domain, self.dim_state, device=device).int()
        # conver the visit_flag to binary
        visit_flag = (visit_nums > 0).int()

        

        # x_expanded = x.unsqueeze(1)  # Shape (batch_size, 1, dim)
        # visit_expanded = self.visit_x.unsqueeze(0)  # Shape (1, K, dim)
        # matches = torch.all(x_expanded == visit_expanded, dim=2)  # Shape (batch_size, K)
        # visit_flag = torch.any(matches, dim=1).int()

        visit_not_top_k_flag = visit_flag * (1 - top_k_flag)

        # if visit_not_top_k_flag.sum() >= 1:
        #     print("visit_not_top_k_flag.sum()", visit_not_top_k_flag.sum())

        return values * (1 - visit_not_top_k_flag)
    


    def cross_apprx_obj(self, func, iteration, Xs_star):
        """
        Get the optimal indices of objective function

        Parameters:
        - n_samples: Number of samples to generate.
        - iteration: The current iteration number.
        - Xs_star: The indices for cross apprx.

        Returns:
        - A tensor of shape (n_samples, dim) with the initial indices.
        """
        # tt_model = cross_approximate(func, self.domain,
        #                             max_batch=self.max_batch, val_size=self.val_size, verbose=True, device=device)
        if iteration == 0:
            if self.warm_start:
                tt_model = cross_approximate(func, self.domain,  rmax=self.apprx_max_tt, nswp=self.apprx_max_iter, kickrank=self.apprx_kickrank,
                                         max_batch=self.max_batch, val_size=self.val_size, verbose=self.verbose, device=device)
            else:
                tt_model = guided_cross_approximate(func, self.domain, nswp=2, ranks_tt=self.value_k, tensors=self.domain_tensors,
                            max_batch=self.max_batch, val_size=self.val_size, verbose=self.verbose, Xs_star=Xs_star, device=device)
        else:
            # tt_model = guided_cross_approximate(func, self.domain, nswp=1, ranks_tt=self.value_k, tensors=self.domain_tensors,
            #                 max_batch=self.max_batch, val_size=self.val_size, verbose=self.verbose, Xs_star=self.mcts_max_indices, device=device)
            tt_model = guided_cross_approximate(func, self.domain, nswp=1, ranks_tt=self.value_k, tensors=self.domain_tensors,
                            max_batch=self.max_batch, val_size=self.val_size, verbose=self.verbose, Xs_star=Xs_star, device=device)
        # tt_model.round_tt(1e-5)

        # Get the top-k indices
        samples, indices = deterministic_top_k_sol(tt_model.cores, self.domain, n_samples = self.value_k, device=device)

        # #stochastic top-k
        # samples, indices = stochastic_top_k_sol(tt_model.cores, self.domain, n_samples = self.value_k, alpha=0.5, device=device)
        
        return samples, indices

    def value_func(self, x, sigma=1.0, smooth=False):
        """
        Batch evaluation of the value function with optional smoothing.

        Parameters:
        - x: Tensor of shape (batch_size, dim), input to evaluate.
        - sigma: Smoothing parameter, larger values make the function smoother (used only if `smooth=True`).
        - smooth: If True, apply smoothing to make the function continuous.

        Returns:
        - A tensor of shape (batch_size,) with values in [0, 1] (smoothed) or binary (original).
        """
        x_expanded = x.unsqueeze(1)  # Shape (batch_size, 1, dim)
        top_k_expanded = self.top_k_x.unsqueeze(0)  # Shape (1, K, dim)

        if smooth:
            # Smooth version: Use Gaussian similarity for smoothing
            distances = torch.norm(x_expanded - top_k_expanded, dim=2)  # Shape (batch_size, K)
            similarities = torch.exp(-distances**2 / (2 * sigma**2))  # Gaussian smoothing
            return torch.sum(similarities, dim=1) / torch.sum(torch.exp(-distances**2 / (2 * sigma**2)), dim=1)
        else:
            # Original version: Binary comparison
            matches = torch.all(x_expanded == top_k_expanded, dim=2)  # Shape (batch_size, K)
            return torch.any(matches, dim=1).int()


    def visit_func(self, x, sigma=1.0, smooth=False):
        """
        Batch evaluation of the visit function with optional smoothing.

        Parameters:
        - x: Tensor of shape (batch_size, dim), input to evaluate.
        - sigma: Smoothing parameter, larger values make the function smoother (used only if `smooth=True`).
        - smooth: If True, apply smoothing to make the function continuous.

        Returns:
        - A tensor of shape (batch_size,) with values in [0, 1] (smoothed) or binary (original).
        """
        x_expanded = x.unsqueeze(1)  # Shape (batch_size, 1, dim)
        top_k_expanded = self.visit_x.unsqueeze(0)  # Shape (1, K, dim)

        if smooth:
            # Smooth version: Use Gaussian similarity for smoothing
            distances = torch.norm(x_expanded - top_k_expanded, dim=2)  # Shape (batch_size, K)
            similarities = torch.exp(-distances**2 / (2 * sigma**2))  # Gaussian smoothing
            return torch.sum(similarities, dim=1) / torch.sum(torch.exp(-distances**2 / (2 * sigma**2)), dim=1)
        else:
            # Original version: Binary comparison
            matches = torch.all(x_expanded == top_k_expanded, dim=2)  # Shape (batch_size, K)
            return torch.any(matches, dim=1).int()


    def cross_update_top_k(self, samples, indices):
        """
        Update the top-k values, x, and indices based on the given samples and indices.

        Parameters:
        - samples: The new samples to evaluate.
        - indices: Indices of the points corresponding to the samples.
        """
        # Evaluate the function on the new samples
        values = self.func(samples)  # Shape: (n_samples,)
        max_values, max_idx = torch.topk(values, self.value_k, largest=True)
        self.mcts_max_indices = indices[max_idx].long()

        # Concatenate current top-k values with new values
        all_values = torch.cat([self.top_k_values, values])
        all_samples = torch.cat([self.top_k_x, samples])
        all_indices = torch.cat([self.top_k_indices, indices])

        # Ensure unique indices and samples in the top_k list
        unique_indices, local_idx = torch.unique(all_indices, return_inverse=True, dim=0)

        scatter_target = torch.zeros(unique_indices.size(0)).to(device)

        # Use scatter_ to map the values to their corresponding unique rows
        scatter_target.scatter_(0, local_idx, all_values)

        # Retrieve the maximum value for each unique row
        unique_values = scatter_target

        # Get the top-k values from unique_values
        top_k_values, topk_idx = torch.topk(unique_values, self.value_k, largest=True)

        # Retrieve the indices of the selected rows
        selected_indices = unique_indices[topk_idx].long()

        if len(self.top_k_values)==0 or not torch.all(top_k_values == self.top_k_values):
            if self.verbose:
                print(colored("The top-k values are updated!", "red", "on_white"))
            self.updated_flag = True

            self.top_k_values = top_k_values
            self.top_k_x = self.get_x_from_indices(selected_indices)
            self.top_k_indices = selected_indices.long()
        else:
            self.updated_flag = False

        self.cross_top_k_values = top_k_values
        self.visit_indices = indices.long()
        self.visit_x = samples # because we are computing the delta visit_tt
    
    def get_parent_tt(self, tt_model):
        """
        Get the parent tt by summing the site
        """
        cores = tt_model.tt().cores
        last_dim = len(cores) - 1
        last_core = cores[last_dim].sum(1) # r_{i-1} x 1
        core_dot = torch.einsum(
                'ijk, kl -> ijl', cores[last_dim-1], last_core
            ) # r_{i-1} x n_{i-1} x 1
        new_cores = cores[:last_dim-1]
        new_cores.append(core_dot)
        return tnt.Tensor(new_cores).to(device)

    def update_visit_tt(self, delta_tt):
        dim = len(delta_tt.tt().cores)
        self.visit_tt[dim-1] = self.visit_tt[dim-1] + delta_tt

        for i in range(dim-1, 0, -1):
            parent_tt = self.get_parent_tt(delta_tt)
            self.visit_tt[i-1] = self.visit_tt[i-1] + parent_tt
            delta_tt = parent_tt
            self.visit_tt[i-1].round_tt(1e-5)

    def get_index_from_x(self, x):
        """
        Get the index of closest value of x in given domain

        Parameters:
        - x: The input tensor.

        Returns:
        - The index tensor.
        """
        return torch.stack([torch.argmin(torch.abs(self.domain[i].unsqueeze(0) - x[:, i].unsqueeze(1)), dim=1) for i in range(self.dim)], dim=1).to(device)
        
    def cmaes_update_top_k(self):
        top_k_res = self.parallel_cmaes_gpu(self.top_k_x, num_iterations=10, popsize=25, warm_start=True)
        values = self.func(top_k_res)
        indices = self.get_index_from_x(top_k_res)

        # Concatenate current top-k values with new values
        all_values = torch.cat([self.top_k_values, values])
        all_samples = torch.cat([self.top_k_x, top_k_res])
        all_indices = torch.cat([self.top_k_indices, indices])

        # # Ensure unique indices and samples in the top_k list
        # unique_indices, local_idx = torch.unique(all_indices, return_inverse=True, dim=0)

        # scatter_target = torch.zeros(unique_indices.size(0)).to(device)

        # # Use scatter_ to map the values to their corresponding unique rows
        # scatter_target.scatter_(0, local_idx, all_values)

        # # Retrieve the maximum value for each unique row
        # unique_values = scatter_target

        # Get the top-k values from unique_values
        top_k_values, topk_idx = torch.topk(all_values, self.value_k, largest=True)

        # Retrieve the indices of the selected rows
        selected_indices = all_indices[topk_idx].long()
        selected_samples = all_samples[topk_idx]

        self.top_k_values = top_k_values
        self.top_k_x = self.get_x_from_indices(selected_indices)
        self.top_k_indices = selected_indices.long()


        # sorted_indices = torch.argsort(top_k_values, descending=True)
        # self.top_k_values = top_k_values[sorted_indices]
        # self.top_k_x = top_k_res[sorted_indices]
        
        # self.top_k_indices = self.get_index_from_x(self.top_k_x)


    def mcts_update_top_k(self, leaf_list, edge_list, samples, indices):
        """
        Update the top-k values, x, and indices based on the given samples and indices.

        Parameters:
        - samples: The new samples to evaluate.
        - indices: Indices of the points corresponding to the samples.
        """
        # Update the visit_tt model
        for leaf_indices in leaf_list:
            site = leaf_indices.shape[1] if leaf_indices.dim() > 1 else 1
            if leaf_indices.dim() == 1:
                leaf_indices = leaf_indices.unsqueeze(1)
            self.visit_indices = leaf_indices.long()
            self.visit_x = self.get_x_from_indices(self.visit_indices)
            
            # print("visit_indices", self.visit_indices)
            delta_visit_tt = guided_cross_approximate(self.visit_func, self.domain[:site], ranks_tt= self.visit_indices.shape[0], #self.ts_ranks_tt,
                        max_batch=self.max_batch, val_size=self.val_size, verbose=self.verbose, Xs_star = self.visit_indices, device=device)
            self.update_visit_tt(delta_visit_tt)
            # self.visit_tt[site-1] = self.visit_tt[site-1] + delta_visit_tt
            
        if edge_list.size()[0] > 0:
            self.visit_indices = edge_list.long() #TODO: check the shape
            self.visit_x = self.get_x_from_indices(self.visit_indices)
            site = edge_list.shape[1]
            delta_visit_tt = guided_cross_approximate(self.visit_func, self.domain[:site], ranks_tt= self.visit_indices.shape[0], #self.ts_ranks_tt,
                        max_batch=self.max_batch, val_size=self.val_size, verbose=self.verbose, Xs_star = self.visit_indices, device=device)
            
            #backpropogate the visit_tt
            self.update_visit_tt(delta_visit_tt)
            # self.visit_tt[site] = self.visit_tt[site] + delta_visit_tt



        #update the top k values
        if self.task_var is not None:
            aug_samples =torch.concat([self.task_var.repeat(samples.shape[0], 1), samples], dim=-1).to(device)
        else:
            aug_samples = samples.to(device)
        st_time = time.time()
        values = fcn_batch_limited(self.func, self.max_batch, device=device)(aug_samples)  # Shape: (n_samples,)
        self.rollout_time = time.time() - st_time
        max_values, max_idx = torch.topk(values, self.value_k, largest=True)
        self.mcts_max_indices = indices[max_idx].long()

        # Concatenate current top-k values with new values
        all_values = torch.cat([self.top_k_values, values])
        all_samples = torch.cat([self.top_k_x, samples])
        all_indices = torch.cat([self.top_k_indices, indices])


        ##*******************************************************************************##
        # # Ensure unique indices and samples in the top_k list, but it typically takes time

        # unique_indices, local_idx = torch.unique(all_indices, return_inverse=True, dim=0)
        # scatter_target = torch.zeros(unique_indices.size(0)).to(device)

        # # Use scatter_ to map the values to their corresponding unique rows
        # scatter_target.scatter_(0, local_idx, all_values)
        # # Retrieve the maximum value for each unique row
        # unique_values = scatter_target

        # all_values = unique_values.clone()
        # all_indices = unique_indices.clone()
        ##*******************************************************************************##

        # Get the top-k values from unique_values
        top_k_values, topk_idx = torch.topk(all_values, self.value_k, largest=True)

        # Retrieve the indices of the selected rows
        selected_indices = all_indices[topk_idx].long()

        self.top_k_values = top_k_values
        self.top_k_x = self.get_x_from_indices(selected_indices)
        self.top_k_indices = selected_indices.long()

    def get_block_indices(self, ucb_indices, fixed_dim=None):
        """
        A simple way
        Generate block indices based on ucb values.

        Parameters:
        - ucb_values: UCB values for each index.
        - ucb_indices: selected indices to visit in MCTS

        Returns:
        - block indices prepared for visiting, with the objective of low-rankness
        """
        # weights = ucb_values[:, 0] / ucb_values.sum()

        # fixed_dim = torch.topk(weights, k=len(self.domain)-3)[1]
        comparsion = (ucb_indices == ucb_indices[0])
        same_count = comparsion.sum(dim=0)

        if fixed_dim is None:
            # #Trick: adding a small random noise to break ties
            random_noise = torch.rand_like(same_count.float()) * 1e-6 
            same_count = same_count + random_noise

            fixed_dim = torch.topk(same_count, len(self.domain)-self.free_block_dim)[1]
            fixed_dim = fixed_dim.sort()[0]
        
        fixed_x = ucb_indices[0, fixed_dim]

        # fill the indicies and samples for block visiting
        # num_visits = torch.prod(self.dim_state) // torch.prod(self.dim_state[fixed_dim])
        num_visits = self.dim_state[0]**self.free_block_dim

        block_indices = torch.zeros(num_visits, len(self.domain)).long().to(device)
        block_indices[:, fixed_dim] = fixed_x.repeat(num_visits, 1)
        free_dim = torch.tensor([i for i in range(len(self.domain)) if i not in fixed_dim])
        free_ids = torch.cartesian_prod(*[torch.arange(self.dim_state[dim]) for dim in free_dim]).to(device)
        block_indices[:, free_dim] = free_ids[:, None] if len(free_dim) == 1 else free_ids

        block_samples = torch.stack([self.domain[i][block_indices[:, i]] for i in range(len(self.domain))], dim=1)
        
        return block_indices, block_samples # num_visits x dim

    def get_block_indices_v2(self, ucb_values, ucb_indices):
        """
        Generate block indices based on ucb values.

        Parameters:
        - ucb_values: UCB values for each index.
        - ucb_indices: selected indices to visit in MCTS

        Returns:
        - block indices prepared for visiting, with the objective of low-rankness
        """
        weights = ucb_values / ucb_values.sum()

        # Compute the ucb value for each unique index
        unique_indices = torch.zeros(ucb_indices.shape[1])
        unique_weights = torch.zeros(ucb_indices.shape[1]) 
        for i in range(ucb_indices.shape[1]):
            unique_id = torch.unique(ucb_indices[:, i], return_inverse=True)[0]
            category_weights = torch.zeros(len(unique_id))
            for j in range(len(unique_id)):
                category_weights[j] = weights[ucb_indices[:, i] == unique_id[j]].sum()
                
            #pick the indices based on the weights
            top_weight, top_id = torch.topk(category_weights, 1)
            unique_indices[i] = unique_id[top_id]
            unique_weights[i] = top_weight

        # decide which dimension to be fixed, while others are free
        # top weight means the dim and value should be definitely visited
        fixed_dim = torch.topk(unique_weights, len(self.domain)-3)[1]
        fixed_x = unique_indices[fixed_dim]

        # fill the indicies and samples for block visiting
        num_visits = torch.prod(self.dim_state) // torch.prod(self.dim_state[fixed_dim])
        block_indices = torch.zeros(num_visits, len(self.domain))
        block_indices[:, fixed_dim] = fixed_x.repeat(num_visits)
        free_dim = torch.tensor([i for i in range(len(self.domain)) if i not in fixed_dim])
        # Generate block indices using torch.meshgrid


        # Generate block indices using torch.meshgrid
        meshgrid = torch.meshgrid([torch.arange(self.dim_state[i]) for i in range(len(self.dim_state))])
        block_indices[:, free_dim] = torch.stack(meshgrid, dim=-1).reshape(-1, len(free_dim))

        block_samples = torch.stack([self.domain[i][block_indices[:, i]] for i in range(len(self.domain))], dim=1)
        
        return block_indices, block_samples # num_visits x dim
    
    def init_tt(self, file_name, max_tt=50, max_iter=5):
        if os.path.exists(file_name+".pt"):
            self.func_tt = torch.load(file_name+".pt", weights_only=False)
            print("The initial tt model is loaded!")
                    
        else:
            #cross approximation for the initial tt model
            self.func_tt = cross_approximate(self.func, self.domain,  rmax=max_tt, nswp=max_iter, kickrank=self.apprx_kickrank,
                                            max_batch=self.max_batch, val_size=self.val_size, verbose=True, device=device)

            #save the initial tt model
            if self.save_tt:
                torch.save(self.func_tt, file_name+".pt")
                print("The initial tt model is saved!")


    def old_iterate(self, max_iters=100, mcts_iters=5, param_C=3, ranks_tt=None, convergence_threshold=1e-3):
        """
        Interleaved cross approximation over the refined_function and MCTS.
        """
        for iteration in range(max_iters):

            # print(f"Iteration {iteration + 1}, top-k values: {self.top_k_values[:5].tolist()}")
            
            # cross approximation for the refined_function
            if self.verbose:
                print(colored(f"Iteration {iteration}, Cross Approximation", "green", "on_white"))


            # mcts given the top-k samples and indices
            if self.verbose:
                print(colored(f"Iteration {iteration}, MCTS", "blue", "on_white"))
            self.mcts_iterate(mcts_iters=mcts_iters, param_C=param_C, ranks_tt=ranks_tt, convergence_threshold=convergence_threshold)

        return self.top_k_x # K x dim
    
    def get_x_from_indices(self, indices):
        return torch.stack([self.domain[i][indices[:, i]] for i in range(indices.shape[1])], dim=1)


    def get_solutions(self, x, n_samples, determinstic=True):
        if determinstic:
            sol, _ = deterministic_top_k_sol(self.aug_func_tt.cores, self.aug_domain, x, n_samples=n_samples, device=device)
        else:
            sol, _ = stochastic_top_k_sol(self.aug_func_tt.cores, self.aug_domain, x, n_samples=n_samples, device=device)
        return sol

    def iterate(self, max_iters=10, param_C=3, test_mode=False, init_flag=True):
        """
        Perform the iterative optimization process.

        Parameters:
        - max_iters: Maximum number of iterations.
        - convergence_threshold: Threshold for convergence based on top-k values.
        """
        if init_flag:
            if self.task_var is not None:
                self.func_tt = self.get_tt_cond_x(self.task_var)
                
                dim_cond_state = self.aug_dim_state[self.task_var.shape[-1]:]
                self.visit_tt = [0*(tnt.rand(dim_cond_state[:i+1],ranks_tt=1).tt()).to(device) for i in range(len(dim_cond_state))]
                self.domain = self.aug_domain[self.task_var.shape[-1]:]
                self.dim_state = self.aug_dim_state[self.task_var.shape[-1]:]
            else:
                self.func_tt = self.aug_func_tt
                self.visit_tt = [0*(tnt.rand(self.dim_state[:i+1],ranks_tt=1).tt()).to(device) for i in range(len(self.dim_state))]
                self.domain = self.aug_domain
                self.dim_state = self.aug_dim_state

        mcts_time = 0
        t0 = time.time()
        for iteration in range(max_iters):
            if self.verbose:
                print("\033[1m" +
                colored(f"MCTS Iteration {iteration+1}", "red", "on_white") + "\033[0m" + 
                f" top-k values: {self.top_k_values.tolist()}")

            leaf_list, edge_list, samples, indices = ucb_select(self, self.func_tt, self.visit_tt, self.domain, param_C=param_C, 
                                                             simulation_samples=self.num_mcts_sim,
                                                             n_samples=self.num_mcts_batch, device=device)
            # print("leaf_list", leaf_list)
            # aug_samples =torch.concat([self.task_var.repeat(samples.shape[0], 1), samples], dim=-1).to(device)

            self.mcts_update_top_k(leaf_list, edge_list, samples, indices)
            
            
            # self.cmaes_update_top_k()
            # mcts_time += time.time() - t0 #- self.rollout_time

            if test_mode and iteration==0:
                tt_res = self.top_k_x.clone()
                if self.task_var is not None:
                    aug_tt_res = torch.concat([self.task_var.repeat(tt_res.shape[0], 1), tt_res], dim=-1).to(device)
                else:
                    aug_tt_res = tt_res.to(device)
                obj_values = self.func(aug_tt_res)
                sorted_indices = torch.argsort(torch.tensor(obj_values), descending=True)
                sorted_tt_res = [tt_res[i] for i in sorted_indices]


            #fine-tune using cmaes
            # top_k_res = self.parallel_cmaes_gpu(self.top_k_x, num_iterations=5, popsize=25, warm_start=True)
            # self.top_k_values = self.func(top_k_res)
            print(f"mcts-iters: {iteration+1}, top-k values: {self.top_k_values.tolist()}")
        mcts_time = time.time() - t0
        if test_mode:
            return self.top_k_x, mcts_time, torch.stack(sorted_tt_res)
        else:
            return self.top_k_x


    def parallel_cmaes_gpu(self, initial_guesses, num_iterations=20, popsize=25, sigma=0.5,
                           warm_start=True, test_mode=False, integer_var=None):
        """
        Use GPU to simultaneously run multiple CMA-ES optimization instances.
        
        Args:
            obj_func: The objective function, supports batch input.
            initial_guesses: Multiple initial points, each corresponding to a CMA-ES instance.
            bounds: Optional optimization bounds.
            task_var: discrete task variables
        
        Returns:
            results: A list containing the optimal solution for each CMA-ES instance.
        """
        import cma

        # Initialize multiple CMA-ES optimizers
        initial_guesses = initial_guesses.detach().cpu().numpy()
        if warm_start:
            es_instances = [
                cma.CMAEvolutionStrategy(initial_guess, sigma, {'popsize': popsize, 'verbose': -1})
                for initial_guess in initial_guesses
            ]
        else:
            es_instances = [
                cma.CMAEvolutionStrategy(np.zeros_like(initial_guess), sigma, {'popsize': popsize, 'verbose': -1})
                for initial_guess in initial_guesses
            ]

        t0 = time.time()
        for iteration in range(num_iterations):
            solutions_list = [es.ask() for es in es_instances]  # Candidate solutions for each CMA-ES

            if self.task_var is not None:
                all_solutions = []
                for i, solutions in enumerate(solutions_list):
                    sol = torch.tensor(solutions).to(device)
                    if integer_var is not None:
                        mixed_sol = torch.cat([integer_var[i].repeat(sol.shape[0], 1), sol], dim=1)
                        all_solutions.append(torch.cat([self.task_var.repeat(mixed_sol.shape[0], 1), mixed_sol], dim=1))
                    else:
                        all_solutions.append(torch.cat([self.task_var.repeat(sol.shape[0], 1), sol], dim=1))
                all_solutions = torch.vstack(all_solutions)
            else:
                if integer_var is not None:
                    for i, solutions in enumerate(solutions_list):
                        sol = torch.tensor(solutions).to(device)
                        mixed_sol = torch.cat([integer_var.repeat(sol.shape[0], 1), sol], dim=1)
                        all_solutions.append(torch.cat([self.task_var.repeat(mixed_sol.shape[0], 1), mixed_sol], dim=1))
                    all_solutions = torch.vstack(all_solutions)
                else:
                    all_solutions = torch.tensor(
                            [sol for solutions in solutions_list for sol in solutions],
                            device=device
                        )  # Batch all candidate solutions and move to GPU

            losses = -fcn_batch_limited(self.func, self.max_batch, device=device)(all_solutions).cpu().numpy()  # Maximize the objective function

            split_losses = torch.split(torch.tensor(losses), popsize)

            for es, instance_losses, instance_solutions in zip(es_instances, split_losses, solutions_list):
                es.tell(instance_solutions, instance_losses.cpu().numpy())

            max_losses = [-min(instance_losses) for instance_losses in split_losses]
            print(f'#CMA-ES, Iteration: {iteration + 1}, Max values: {max(max_losses)}', end="\r" if iteration < num_iterations - 1 else "\n")

        results = torch.stack([torch.tensor(es.result.xbest).to(device=device) for es in es_instances])

        if self.task_var is not None:
            if integer_var is not None:
                aug_results = torch.cat([self.task_var.repeat(results.shape[0], 1), integer_var, results], dim=1)
            else:
                aug_results = torch.cat([self.task_var.repeat(results.shape[0], 1), results], dim=1)
            obj_values = self.func(aug_results)
        else:
            obj_values = self.func(results)
            
        cmaes_time = time.time() - t0 #- rollout_time
        # return results with an order of decresed objective value
        sorted_indices = torch.argsort(torch.tensor(obj_values), descending=True)
        results = torch.stack([results[i] for i in sorted_indices])

        if test_mode:
            return results, cmaes_time
        else:
            return results

    def cost(self, x):
        return -1 * self.func(x)

    def scipy_optimize_2(self, x, domain, task_dim, bound=True, method='SLSQP', tol=1e-3):
        ''' 
            Optimize from an initial guess x.
            To Do: Move it to pytorch based optimization instead of depending on scipy (slow)
            method: 'L-BFGS-B' or 'SLSQP'
            bound: if True the optimizaton (decision) variables  will be constrained to the domain provided
        '''
        from scipy.optimize import minimize
        from scipy.optimize import Bounds

        # For optimization/fine-tuning
        lb = []; ub = []
        for domain_i in domain:
            lb.append(domain_i[0].item())
            ub.append(domain_i[-1].item())
        self.scipy_bounds = Bounds(np.array(lb),np.array(ub))

        # pytorch-to-numpy interface
        @torch.enable_grad()
        def cost_fcn(x):
            return self.cost(torch.from_numpy(x).reshape(1,-1).to(device)).to("cpu").numpy()
        @torch.enable_grad()
        def jacobian_cost(x):
            jac= torch.autograd.functional.jacobian(self.cost,torch.from_numpy(x).reshape(1,-1).to(device)).reshape(-1)
            sites_task = torch.arange(task_dim).long().to(device)
            jac[sites_task] = 0
            return jac.cpu().numpy().reshape(-1)
        
        if bound ==True: # constrained optimization
            results = minimize(cost_fcn, x.reshape(-1), tol=tol, bounds=self.scipy_bounds)
        else: # unconstrained optimization
            results = minimize(cost_fcn, x.reshape(-1), method=method,jac=jacobian_cost, tol=tol)
        return torch.from_numpy(results.x).view(1,-1).to(device), results

    def scipy_optimize(self, x, domain, integer_var=None, bound=True, method='SLSQP', max_iter=5, tol=1e-3):
        ''' 
            Optimize from an initial guess x.
            To Do: Move it to pytorch based optimization instead of depending on scipy (slow)
            method: 'L-BFGS-B' or 'SLSQP'
            bound: if True the optimizaton (decision) variables  will be constrained to the domain provided
        '''
        from scipy.optimize import minimize
        from scipy.optimize import Bounds

        # For optimization/fine-tuning
        lb = []; ub = []
        for domain_i in domain:
            lb.append(domain_i[0].item())
            ub.append(domain_i[-1].item())
        self.scipy_bounds = Bounds(np.array(lb),np.array(ub))

        # pytorch-to-numpy interface
        @torch.enable_grad()
        def cost_fcn(x):
            if integer_var is not None:
                xs = np.concatenate([integer_var.cpu().numpy(), x], axis=0)
            else:
                xs = x
            xss = torch.cat([self.task_var, torch.from_numpy(xs).reshape(1,-1).to(device)], dim=1).to(device)
            return self.cost(xss).to("cpu").numpy()
        @torch.enable_grad()
        def jacobian_cost(x):
            if integer_var is not None:
                xs = np.concatenate([integer_var.cpu().numpy(), x], axis=0)
            else:
                xs = x
            xs_torch = torch.from_numpy(xs).reshape(1, -1).to(device).requires_grad_(True)
            xss = torch.cat([self.task_var, xs_torch], dim=1).to(device)
            cost_val = self.cost(xss)
            cost_val.backward()
            print("Grad:", xs_torch.grad)
            grad = xs_torch.grad
            return grad.cpu().numpy()
        
        if bound ==True: # constrained optimization
            results = minimize(cost_fcn, x.cpu().reshape(-1), method=method,jac=jacobian_cost, tol=tol, bounds=self.scipy_bounds, 
                                   options={
                                            'disp': True,
                                            'eps': 1e-2,    
                                            'maxiter': max_iter,
                                        })
        else: # unconstrained optimization
            results = minimize(cost_fcn, x.cpu().reshape(-1), method=method,jac=jacobian_cost, tol=tol, options={'disp': True})
        # print(results.message)
        if integer_var is not None:
            xs = torch.cat([integer_var[None, :], torch.from_numpy(results.x).view(1,-1).to(device)], dim=-1).to(device)
        else:
            xs =  torch.from_numpy(results.x).view(1,-1).to(device)
        return xs, results


