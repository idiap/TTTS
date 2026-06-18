#
# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Teng Xue <teng.xue@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import torch
import tntorch as tnt
import numpy as np
import scipy

from typing import Any, Callable, Sequence, Union
import tntorch as tn
import time, logging, sys
from tntorch.maxvol import py_maxvol, py_rect_maxvol

torch.set_default_dtype(torch.float64)
logging.basicConfig(level=logging.ERROR)

import math 

def get_exponential_discretization(xmax=1.0,n=100,sc=1.0,flip=False,device='cpu'):
    '''
    note: applies only for symmetric bounds; (-xmax,xmax)
    Generates discretization non-uniformly: in an exponential manner
    
    sc --> inf => uniform discretization
    sc --> 0 => discretization points are more dense near 0.
    flip=True:
        discretization points are more dense near the boundary
    '''

    xmin = -1*xmax
    even_n = n%2
    n_p = int(n/2) + even_n
    xp = xmax*torch.linspace(0,1,n_p).to(device)
    if not flip:
        yp = -1+torch.exp(sc*xp.abs()/xmax)
    else:
        yp = 1-1/(1+torch.exp(sc*xp.abs()/xmax))
        yp = yp-yp.min()
    yp = xmax*yp/yp.max()
    idx_n = -1*torch.arange(n_p)[even_n:].to(device)
    xn = -1*xp[idx_n]
    yn = -1*yp[idx_n]
    y = torch.cat((yn,yp),dim=-1)
    return y

def idx2domain(I, domain, device): # for any discretization
    ''' Map the index of the tensor/discretization to the domain'''
    X = torch.zeros(I.shape).to(device)
    for i in range(I.shape[1]):
        X[:,i] =  domain[i][I[:, i]]
    return X

def domain2idx(x, domain, device, uniform=False):
    ''' 
    Map x from the domain to the index of the discretization 
    '''
    I = torch.zeros(x.shape).to(device)

    if uniform: # if the discretization is uniform
        for i in range(x.shape[-1]):
            min_i = domain[i][0] # 
            step_i = domain[i][1]-domain[i][0] 
            I[:,i] = ((x[:,i] - min_i)/step_i).round() 
    else: 
        for i in range(x.shape[-1]):
            I[:,i] = torch.argmin(torch.abs(x[:,i].view(-1,1)- domain[i]), dim=1) 
    return I.long()

def get_elements_from_cores(tt_cores, idx):
    '''
    Given the tt_cores and a batch of index get the  elements
    '''
    mat_ = tt_cores[0][:,idx[:,0],:]
    for i in range(1,idx.shape[-1]):
        mat_ = torch.einsum('ijk,kjl->ijl',(mat_,tt_cores[i][:,idx[:,i]]))
    return mat_.view(-1)


def get_elements(tt_model, idx):
    '''
    Given the tt_model in tntorch format and a batch of index get the  elements
    '''
    return get_elements_from_cores(tt_model.tt().cores, idx)


def get_tt_mean(tt_model):
    '''
    Given the tt_model in tntorch format find the mean of the tt-model
    '''

    return get_tt_mean_from_cores(tt_model.tt().cores)


def get_tt_mean_from_cores(tt_cores):
    '''
        find the mean of the tt-model given its cores
    '''
    sum_ = tt_cores[0].sum(dim=1)/tt_cores[0].shape[1]
    for core in tt_cores[1:]:
        sum_ = sum_@core.sum(dim=1)/core.shape[1]
    return sum_.item()



def get_value(tt_model, x,  domain, 
                    n_discretization, max_batch=10**5, device="cpu"):
    ''' 
    Evaluate the tt-model (in tntorch format) at the given state with Linear interpolation between the nodes. Assumes uniform discretization 
    dh_domain : a 1D tensor containing the step size of discretization for each site/mode
    n_discretization: a 1D tensor continginingthe number of discretization points along each mode
    '''
    return get_value_from_cores(tt_model.tt().cores, x,  domain, 
                                    n_discretization,
                                    max_batch, device)

def get_value_from_cores(tt_cores, x,  domain, 
                                    n_discretization=None, max_batch=10**5,
                                     device="cpu"):
    
    if n_discretization is None:
        n_discretization = torch.tensor([len(dom) for dom in domain]).to(device)
    
    def fcn(x_batch):
        idx_1 = domain2idx(x_batch, domain=domain, device=device) # find the closest/floor index of the state (w.r.t to the discretizaton)
        x_1 = idx2domain(idx_1, domain, device=device) 
        dx = (x_batch-x_1)#/dh_domain.view(1,-1) # 
        idx_2 = torch.clip(idx_1+torch.sign(dx),
                                    n_discretization[:x_batch.shape[-1]]*0,
                                    n_discretization[:x_batch.shape[-1]]-1).long() # next index
        x_2 = idx2domain(idx_2, domain, device=device)
        dx = dx.abs()/(1e-6+(x_2-x_1).abs())
        mat_ = tt_cores[0][:,idx_1[:,0],:]+dx[:,0].view(1,-1,1)*(tt_cores[0][:,idx_2[:,0],:]-tt_cores[0][:,idx_1[:,0],:])
        for i in range(1,idx_1.shape[-1]):
            mat = tt_cores[i][:,idx_1[:,i],:]+dx[:,i].view(1,-1,1)*(tt_cores[i][:,idx_2[:,i],:]-tt_cores[i][:,idx_1[:,i],:])
            mat_ = torch.einsum('ijk,kjl->ijl',mat_,mat)
        return mat_.view(-1)
    return fcn_batch_limited(fcn=fcn,max_batch=max_batch, device=device)(x)

def get_value_from_cores_nonbatch(tt_cores, x,  domain, 
                                    n_discretization=None, device="cpu"):
    ''' 
    Evaluate the tt-model (given its tt_cores) at the given state with 
    Linear interpolation between the nodes.  
    dh_domain : a 1D tensor containing the step size of discretization for each site/mode
    n_discretization: a 1D tensor continginingthe number of discretization points along each mode
    '''
    if n_discretization is None:
        n_discretization = torch.tensor([len(dom) for dom in domain]).to(device)

    idx_1 = domain2idx(x, domain=domain, device=device) # find the closest/floor index of the state (w.r.t to the discretizaton)
    x_1 = idx2domain(idx_1, domain, device=device) 
    dx = (x-x_1)#/dh_domain.view(1,-1) # 
    idx_2 = torch.clip(idx_1+torch.sign(dx),
                                n_discretization[:x.shape[-1]]*0,
                                n_discretization[:x.shape[-1]]-1).long() # next index
    x_2 = idx2domain(idx_2, domain, device=device)
    dx = dx.abs()*1/(1e-6+(x_2-x_1).abs())
    mat_ = tt_cores[0][:,idx_1[:,0],:]+dx[:,0].view(1,-1,1)*(tt_cores[0][:,idx_2[:,0],:]-tt_cores[0][:,idx_1[:,0],:])
    for i in range(1,idx_1.shape[-1]):
        mat = tt_cores[i][:,idx_1[:,i],:]+dx[:,i].view(1,-1,1)*(tt_cores[i][:,idx_2[:,i],:]-tt_cores[i][:,idx_1[:,i],:])
        mat_ = torch.einsum('ijk,kjl->ijl',mat_,mat)
    return mat_.view(-1)

def get_value_discrete(tt_model, x, domain, device="cpu"):
    '''
        Evaluate tt-model at the given point (in batch) from the domain. Assuming uniform discretization
        Input: x, batch_size x dim
    '''
    idx_state = domain2idx(x, domain, device) # find the index (w.r.t to the discretizaton)
    return get_elements(tt_model,idx_state).view(-1) #v_model[idx_state].torch() # batch_size x 1


def cross_approximate(fcn, domain, max_batch=10**5, ranks_tt=None,
                        rmax=200, nswp=20, eps=1e-4, verbose=False, val_size=1e5,
                        kickrank=3, return_info=False, device="cpu"):
    ''' 
        TT-Cross Approximation using tntorch's implementation
        eps: accuracy of approximation
    '''
    if return_info:
        tt_model, info = tnt.cross(fcn_batch_limited(fcn, max_batch=max_batch, device=device),
            domain=domain, ranks_tt= ranks_tt,
            max_iter=nswp, eps=eps, rmax=rmax, kickrank=kickrank, 
            function_arg='matrix',device=device,_minimize=False, #ranks_tt=kickrank,
            val_size=val_size, verbose=verbose, return_info=return_info)
        tt_model.round_tt(eps)
        return (tt_model.to(device), info)
    else:

        tt_model = tnt.cross(fcn_batch_limited(fcn, max_batch=max_batch, device=device),
            domain=domain, ranks_tt= ranks_tt,
            max_iter=nswp, eps=eps, rmax=rmax, kickrank=kickrank, 
            function_arg='matrix',device=device,_minimize=False, #ranks_tt=kickrank,
            val_size=val_size, verbose=verbose, return_info=return_info)
        tt_model.round_tt(eps)
        return tt_model.to(device)



def guided_cross_approximate(fcn, domain, max_batch=10**5, ranks_tt=None, tensors=None,
                        rmax=200, nswp=20, eps=1e-4, verbose=False, val_size=1e5,
                        kickrank=3, return_info=False, top_indices=None, Xs_star=None, device="cpu"):
    ''' 
        TT-Cross Approximation using tntorch's implementation
        eps: accuracy of approximation
    '''
    tt_model = guided_tt_cross(fcn_batch_limited(fcn, max_batch=max_batch, device=device),
        domain=domain,ranks_tt=ranks_tt,
        max_iter=nswp, eps=eps, rmax=rmax, kickrank=kickrank, tensors=tensors,
        function_arg='matrix',device=device,_minimize=False, #ranks_tt=kickrank,
        val_size=val_size, verbose=verbose, return_info=return_info, Xs_star=Xs_star)
    tt_model.round_tt(eps)
    return tt_model.to(device)
    

# Initialize left and right interfaces for `tensors`
def init_interfaces(tensors, rsets, N, device):
    t_linterfaces = []
    t_rinterfaces = []
    for t in tensors:
        linterfaces = [torch.ones(1, t.ranks_tt[0]).to(device)] + [None] * (N - 1)
        rinterfaces = [None] * (N - 1) + [torch.ones(t.ranks_tt[t.dim()], 1).to(device)]
        for j in range(N - 1):
            M = torch.ones(t.cores[-1].shape[-1], len(rsets[j])).to(device)
            for n in range(N - 1, j, -1):
                if t.cores[n].dim() == 3:  # TT core
                    M = torch.einsum(
                        "iaj,ja->ia",
                        [t.cores[n][:, rsets[j][:, n - 1 - j], :].to(device), M],
                    )
                else:  # CP factor
                    M = torch.einsum(
                        "ai,ia->ia",
                        [t.cores[n][rsets[j][:, n - 1 - j], :].to(device), M],
                    )
            rinterfaces[j] = M
        t_linterfaces.append(linterfaces)
        t_rinterfaces.append(rinterfaces)
    return t_linterfaces, t_rinterfaces



# @profile
def guided_tt_cross(
    function: Callable = lambda x: x,
    domain=None,
    tensors: Union[Any, Sequence[Any]] = None,
    function_arg: str = "vectors",
    ranks_tt: Union[int, Sequence[int]] = None,
    kickrank: int = 3,
    rmax: int = 100,
    eps: float = 1e-6,
    max_iter: int = 25,
    val_size: int = 1000,
    verbose: bool = True,
    return_info: bool = False,
    record_samples: bool = False,
    _minimize: bool = False,
    device: Any = None,
    suppress_warnings: bool = False,
    detach_evaluations: bool = False,
    Xs_star = None,  # New parameter
):
    """
    Cross-approximation routine that samples a black-box function and returns an N-dimensional tensor train approximating it.

    New: `Xs_star` parameter allows direct initialization of row and column indices from known valid points.
    """
    if device is None and tensors is not None:
        if type(tensors) == list:
            device = tensors[0].cores[0].device
        else:
            device = tensors.cores[0].device

    if verbose:
        print("cross device is", device)

    # try:
    #     import maxvolpy.maxvol

    #     maxvol = maxvolpy.maxvol.maxvol
    #     rect_maxvol = maxvolpy.maxvol.rect_maxvol
    # except ModuleNotFoundError:
    #     print(
    #         "Functions that require cross-approximation can be accelerated with the optional maxvolpy package,"
    #         + " which can be installed by 'pip install maxvolpy'. "
    #         + "More info is available at https://bitbucket.org/muxas/maxvolpy."
    #     )


    maxvol = py_maxvol
    rect_maxvol = py_rect_maxvol

    assert domain is not None or tensors is not None
    assert function_arg in ("vectors", "matrix")
    if function_arg == "matrix":

        def f(*args):
            return function(torch.cat([arg[:, None] for arg in args], dim=1))

    else:
        f = function

    if tensors is None:
        tensors = tn.meshgrid(domain)

    if not hasattr(tensors, "__len__"):
        tensors = [tensors]
    for t in tensors:
        if t.batch:
            raise ValueError("Batched tensors are not supported.")
    tensors = [t.decompress_tucker_factors(_clone=False) for t in tensors]
    Is = list(tensors[0].shape)
    N = len(Is)


    if Xs_star is not None:
        Xs_star = np.stack(Xs_star.cpu().numpy(), axis=0) #Note: Xs_star is the indices within the domain
        # Use Xs_star to initialize row and column indices
        lsets = [np.array([[0]])] + [np.array(Xs_star[:, :n]) for n in range(N - 1)]
        rsets = [np.array(Xs_star[:, n:]) for n in range(1, N)] + [np.array([[0]])]
    else:
        assert False, "Xs_star is required"
        # lsets = [np.array([[0]])] + [None] * (N - 1)
        # randint = np.hstack(
        #     [np.random.randint(0, Is[n + 1], [max(Rs), 1]) for n in range(N - 1)]
        #     + [np.zeros([max(Rs), 1], dtype=int)]
        # )
        # rsets = [randint[: Rs[n + 1], n:] for n in range(N - 1)] + [np.array([[0]])]

    # Process ranks and cap them, if needed
    if ranks_tt is None:
        # ranks_tt = len(Xs_star)
        ranks_tt = 15
    else:
        kickrank = None
    if not hasattr(ranks_tt, "__len__"):
        ranks_tt = [ranks_tt] * (N - 1)
    ranks_tt = [1] + list(ranks_tt) + [1]
    Rs = np.array(ranks_tt)

    for n in list(range(1, N)) + list(range(N - 1, -1, -1)):
        Rs[n] = min(Rs[n - 1] * Is[n - 1], Rs[n], Is[n] * Rs[n + 1], rsets[n - 1].shape[0])

    # Initialize cores at random
    cores = [torch.randn(Rs[n], Is[n], Rs[n + 1]).to(device) for n in range(N)]

    t_linterfaces, t_rinterfaces = init_interfaces(tensors, rsets, N, device)

    # Create a validation set
    Xs_val = [
        torch.as_tensor(np.random.choice(I, int(val_size))).to(device) for I in Is
    ] # Randomly sample from the domain

    # add Xs_star to the validation set
    Xs_star_val = torch.as_tensor(np.stack([Xs_star[:, i] for i in range(len(Is))], axis=1)).to(device) # n_samples x n_dim
    Xs_val = [torch.cat([Xs_val[i], Xs_star_val[:, i]], dim=0) for i in range(len(Is))]

    ys_val = f(*[t[Xs_val].torch() for t in tensors])
    if ys_val.dim() > 1:
        assert ys_val.dim() == 2
        assert ys_val.shape[1] == 1
        ys_val = ys_val[:, 0]

    assert len(ys_val) == val_size + len(Xs_star)
    norm_ys_val = torch.norm(ys_val)

    if verbose:
        print(
            "Cross-approximation over a {}D domain containing {:g} grid points:".format(
                N, tensors[0].numel()
            )
        )

    start = time.time()
    converged = False

    info = {"nsamples": 0, "eval_time": 0, "val_epss": [], "min": 0, "argmin": None}
    if record_samples:
        info["sample_positions"] = torch.zeros(0, N).to(device)
        info["sample_values"] = torch.zeros(0).to(device)

    def evaluate_function(
        j,
    ):  # Evaluate function over Rs[j] x Rs[j+1] fibers, each of size I[j]
        Xs = []
        for k, t in enumerate(tensors):
            if tensors[k].cores[j].dim() == 3:  # TT core
                V = torch.einsum(
                    "ai,ibj,jc->abc",
                    [t_linterfaces[k][j], t.cores[j], t_rinterfaces[k][j]],
                )
            else:  # CP factor
                V = torch.einsum(
                    "ai,bi,ic->abc",
                    [t_linterfaces[k][j], t.cores[j], t_rinterfaces[k][j]],
                )
            Xs.append(V.flatten())

        eval_start = time.time()
        evaluation = f(*Xs)
        if record_samples:
            info["sample_positions"] = torch.cat(
                (info["sample_positions"], torch.cat([x[:, None] for x in Xs], dim=1)),
                dim=0,
            )
            info["sample_values"] = torch.cat((info["sample_values"], evaluation))
        info["eval_time"] += time.time() - eval_start
        if _minimize:
            evaluation = np.pi / 2 - torch.atan(
                (evaluation - info["min"])
            )  # Function used by I. Oseledets for TT minimization in ttpy
            evaluation_argmax = torch.argmax(evaluation)
            eval_min = (
                torch.tan(np.pi / 2 - evaluation[evaluation_argmax]) + info["min"]
            )
            if info["min"] == 0 or eval_min < info["min"]:
                coords = np.unravel_index(
                    evaluation_argmax.cpu(), [Rs[j], Is[j], Rs[j + 1]]
                )
                info["min"] = eval_min
                info["argmin"] = (
                    tuple(lsets[j][coords[0]][1:])
                    + tuple([coords[1]])
                    + tuple(rsets[j][coords[2]][:-1])
                )

        # Check for nan/inf values
        if evaluation.dim() == 2:
            evaluation = evaluation[:, 0]
        invalid = torch.nonzero(torch.isnan(evaluation) | torch.isinf(evaluation))
        if len(invalid) > 0:
            invalid = invalid[0].item()
            raise ValueError(
                "Invalid return value for function {}: f({}) = {}".format(
                    function,
                    ", ".join(
                        "{:g}".format(x[invalid].detach().cpu().numpy()) for x in Xs
                    ),
                    f(*[x[invalid : invalid + 1][:, None] for x in Xs]).item(),
                )
            )

        V = torch.reshape(evaluation, [Rs[j]*Is[j], -1])
        Q, R = torch.linalg.qr(V)
        V = Q @ R[:, :Rs[j + 1]]
        # Q = Q[:, :Rs[j + 1]]
        # V = torch.linalg.lstsq(Q.t(), Q.t()).solution.t()
        
        info["nsamples"] += V.numel()
        return V.reshape(Rs[j], Is[j], Rs[j + 1])


    # Sweeps (main loop remains the same, but uses `lsets` and `rsets` initialized by `Xs_star`)
    for i in range(max_iter):

        if verbose:
            print("iter: {: <{}}".format(i, len("{}".format(max_iter)) + 1), end="")
            sys.stdout.flush()

        left_locals = []

        # Left-to-right
        for j in range(N - 1):

            # Update tensors for current indices

            V = evaluate_function(j) # bigger self.max_batch for faster computation

            # QR + maxvol towards the right
            if j == 0:
                V = torch.reshape(V, [Is[j], -1])  # Left unfolding
            else:
                V = torch.reshape(V, [-1, Rs[j + 1]])
                # V = torch.reshape(V, [Rs[j] * Rs[j + 1], -1])
            Q, _ = torch.linalg.qr(V)

            if _minimize:
                local, _ = rect_maxvol(Q.detach().cpu().numpy(), maxK=Q.shape[1])
            else:
                local, _ = maxvol(Q.detach().cpu().numpy())

            V = torch.linalg.lstsq(Q[local, :].t(), Q.t()).solution.t()
            cores[j] = torch.reshape(V, [Rs[j], Is[j], Rs[j + 1]])
            left_locals.append(local)

            # Map local indices to global ones
            local_r, local_i = np.unravel_index(local, [Rs[j], Is[j]])
            lsets[j + 1] = np.c_[lsets[j][local_r, :], local_i]
            for k, t in enumerate(tensors):
                if t.cores[j].dim() == 3:  # TT core
                    t_linterfaces[k][j + 1] = torch.einsum(
                        "ai,iaj->aj",
                        [t_linterfaces[k][j][local_r, :], t.cores[j][:, local_i, :]],
                    )
                else:  # CP factor
                    t_linterfaces[k][j + 1] = torch.einsum(
                        "ai,ai->ai",
                        [t_linterfaces[k][j][local_r, :], t.cores[j][local_i, :]],
                    )


        # Right-to-left sweep
        for j in range(N - 1, 0, -1):

            # Update tensors for current indices
            V = evaluate_function(j)

            # QR + maxvol towards the left
            V = torch.reshape(V, [Rs[j], -1])  # Right unfolding
            Q, _ = torch.linalg.qr(V.t())
            if _minimize:
                local, _ = rect_maxvol(Q.detach().cpu().numpy(), maxK=Q.shape[1])
            else:
                local, _ = maxvol(Q.detach().cpu().numpy())
            V = torch.linalg.lstsq(Q[local, :].t(), Q.t()).solution
            cores[j] = torch.reshape(torch.as_tensor(V), [Rs[j], Is[j], Rs[j + 1]])

            # Map local indices to global ones
            local_i, local_r = np.unravel_index(local, [Is[j], Rs[j + 1]])
            rsets[j - 1] = np.c_[local_i, rsets[j][local_r, :]]

            for k, t in enumerate(tensors):
                if t.cores[j].dim() == 3:  # TT core
                    t_rinterfaces[k][j - 1] = torch.einsum(
                        "iaj,ja->ia",
                        [t.cores[j][:, local_i, :], t_rinterfaces[k][j][:, local_r]],
                    )
                else:  # CP factor
                    t_rinterfaces[k][j - 1] = torch.einsum(
                        "ai,ia->ia",
                        [t.cores[j][local_i, :], t_rinterfaces[k][j][:, local_r]],
                    )


        # Leave the first core ready
        V = evaluate_function(0)
        # cores[0] = V
        cores[0] = torch.reshape(V, [Rs[0], Is[0], Rs[1]])

        # Evaluate validation error
        val_eps = torch.norm(ys_val - tn.Tensor(cores)[Xs_val].torch()) / (norm_ys_val+1e-6)
        info["val_epss"].append(val_eps)
        if val_eps < eps:
            converged = True
        if verbose:  # Print status
            if _minimize:
                print("| best: {:.8g}".format(info["min"]), end="")
            else:
                print("| eps: {:.3e}".format(val_eps), end="")
            print(
                " | time: {:8.4f} | largest rank: {:3d}".format(
                    time.time() - start, max(Rs)
                ),
                end="",
            )
            if converged:
                print(" <- converged: eps < {}".format(eps))
            elif i == max_iter - 1:
                print(" <- max_iter was reached: {}".format(max_iter))
            else:
                print()
        if converged:
            break
        elif i < max_iter - 1 and kickrank is not None:  # Augment ranks
            newRs = Rs.copy()
            newRs[1:-1] = np.minimum(rmax, newRs[1:-1] + kickrank)
            for n in list(range(1, N)) + list(range(N - 1, 0, -1)):
                newRs[n] = min(newRs[n - 1] * Is[n - 1], newRs[n], Is[n] * newRs[n + 1])
            extra = np.hstack(
                [np.random.randint(0, Is[n + 1], [max(newRs), 1]) for n in range(N - 1)]
                + [np.zeros([max(newRs), 1], dtype=int)]
            )
            for n in range(N - 1):
                if newRs[n + 1] > Rs[n + 1]:
                    rsets[n] = np.vstack(
                        [rsets[n], extra[: newRs[n + 1] - Rs[n + 1], n:]]
                    )
            Rs = newRs
            t_linterfaces, t_rinterfaces = init_interfaces(
                tensors, rsets, N, device
            )  # Recompute interfaces

    if val_eps > eps and not _minimize and not suppress_warnings:
        logging.warning(
            "eps={:g} (larger than {}) when cross-approximating {}".format(
                val_eps, eps, function
            )
        )

    if verbose:
        print(
            "Did {} function evaluations, which took {:.4g}s ({:.4g} evals/s)".format(
                info["nsamples"],
                info["eval_time"],
                info["nsamples"] / info["eval_time"],
            )
        )
        print()

    ret = tn.Tensor(
        [c if isinstance(c, torch.Tensor) else torch.tensor(c) for c in cores]
    )

    if return_info:
        info["lsets"] = lsets
        info["rsets"] = rsets
        info["Rs"] = Rs
        info["left_locals"] = left_locals
        info["total_time"] = time.time() - start
        info["val_eps"] = val_eps
        return ret, info
    else:
        return ret



def tnt_cross(
    function: Callable = lambda x: x,
    domain=None,
    tensors: Union[Any, Sequence[Any]] = None,
    function_arg: str = "vectors",
    ranks_tt: Union[int, Sequence[int]] = None,
    kickrank: int = 3,
    rmax: int = 100,
    eps: float = 1e-6,
    max_iter: int = 25,
    val_size: int = 1000,
    verbose: bool = True,
    return_info: bool = False,
    record_samples: bool = False,
    _minimize: bool = False,
    device: Any = None,
    suppress_warnings: bool = False,
    detach_evaluations: bool = False,
):
    """
    Cross-approximation routine that samples a black-box function and returns an N-dimensional tensor train approximating it. It accepts either:

    - A domain (tensor product of :math:`N` given arrays) and a function :math:`\\mathbb{R}^N \\to \\mathbb{R}`
    - A list of :math:`K` tensors of dimension :math:`N` and equal shape and a function :math:`\\mathbb{R}^K \\to \\mathbb{R}`

    :Examples:

    >>> tn.cross(function=lambda x: x**2, tensors=[t])  # Compute the element-wise square of `t` using 5 TT-ranks

    >>> domain = [torch.linspace(-1, 1, 32)]*5
    >>> tn.cross(function=lambda x, y, z, t, w: x**2 + y*z + torch.cos(t + w), domain=domain)  # Approximate a function over the rectangle :math:`[-1, 1]^5`

    >>> tn.cross(function=lambda x: torch.sum(x**2, dim=1), domain=domain, function_arg='matrix')  # An example where the function accepts a matrix

    References:

    - I. Oseledets, E. Tyrtyshnikov: `"TT-cross Approximation for Multidimensional Arrays" (2009) <http://www.mat.uniroma2.it/~tvmsscho/papers/Tyrtyshnikov5.pdf>`_
    - D. Savostyanov, I. Oseledets: `"Fast Adaptive Interpolation of Multi-dimensional Arrays in Tensor Train Format" (2011) <https://ieeexplore.ieee.org/document/6076873>`_
    - S. Dolgov, R. Scheichl: `"A Hybrid Alternating Least Squares - TT Cross Algorithm for Parametric PDEs" (2018) <https://arxiv.org/pdf/1707.04562.pdf>`_
    - A. Mikhalev's `maxvolpy package <https://bitbucket.org/muxas/maxvolpy>`_
    - I. Oseledets (and others)'s `ttpy package <https://github.com/oseledets/ttpy>`_

    :param function: should produce a vector of :math:`P` elements. Accepts either :math:`N` comma-separated vectors, or a matrix (see `function_arg`)
    :param domain: a list of :math:`N` vectors (incompatible with `tensors`)
    :param tensors: a :class:`Tensor` or list thereof (incompatible with `domain`)
    :param function_arg: if 'vectors', `function` accepts :math:`N` vectors of length :math:`P` each. If 'matrix', a matrix of shape :math:`P \\times N`.
    :param ranks_tt: int or list of :math:`N-1` ints. If None, will be determined adaptively
    :param kickrank: when adaptively found, ranks will be increased by this amount after every iteration (full sweep left-to-right and right-to-left)
    :param rmax: this rank will not be surpassed
    :param eps: the procedure will stop after this validation error is met (as measured after each iteration)
    :param max_iter: int
    :param val_size: size of the validation set
    :param verbose: default is True
    :param return_info: if True, will also return a dictionary with informative metrics about the algorithm's outcome
    :param device: PyTorch device
    :param suppress_warnings: Boolean, if True, will hide the message about insufficient accuracy
    :param detach_evaluations: Boolean, if True, will remove gradient buffers for the function

    :return: an N-dimensional TT :class:`Tensor` (if `return_info`=True, also a dictionary)
    """
    if device is None and tensors is not None:
        if type(tensors) == list:
            device = tensors[0].cores[0].device
        else:
            device = tensors.cores[0].device

    if verbose:
        print("cross device is", device)

    try:
        import maxvolpy.maxvol

        maxvol = maxvolpy.maxvol.maxvol
        rect_maxvol = maxvolpy.maxvol.rect_maxvol
    except ModuleNotFoundError:
        print(
            "Functions that require cross-approximation can be accelerated with the optional maxvolpy package,"
            + " which can be installed by 'pip install maxvolpy'. "
            + "More info is available at https://bitbucket.org/muxas/maxvolpy."
        )
        from tntorch.maxvol import py_maxvol, py_rect_maxvol

        maxvol = py_maxvol
        rect_maxvol = py_rect_maxvol

    assert domain is not None or tensors is not None
    assert function_arg in ("vectors", "matrix")
    if function_arg == "matrix":

        def f(*args):
            return function(torch.cat([arg[:, None] for arg in args], dim=1))

    else:
        f = function

    
    if tensors is None:
        tensors = tn.meshgrid(domain)

    if not hasattr(tensors, "__len__"):
        tensors = [tensors]
    for t in tensors:
        if t.batch:
            raise ValueError("Batched tensors are not supported.")
    tensors = [t.decompress_tucker_factors(_clone=False) for t in tensors]
    Is = list(tensors[0].shape)
    N = len(Is)

    # Process ranks and cap them, if needed
    if ranks_tt is None:
        ranks_tt = len(Xs_star)
    else:
        kickrank = None
    if not hasattr(ranks_tt, "__len__"):
        ranks_tt = [ranks_tt] * (N - 1)
    ranks_tt = [1] + list(ranks_tt) + [1]
    Rs = np.array(ranks_tt)

    for n in list(range(1, N)) + list(range(N - 1, -1, -1)):
        Rs[n] = min(Rs[n - 1] * Is[n - 1], Rs[n], Is[n] * Rs[n + 1])

    # Initialize cores at random
    cores = [torch.randn(Rs[n], Is[n], Rs[n + 1]).to(device) for n in range(N)]

    # Prepare left and right sets
    lsets = [np.array([[0]])] + [None] * (N - 1)
    randint = np.hstack(
        [np.random.randint(0, Is[n + 1], [max(Rs), 1]) for n in range(N - 1)]
        + [np.zeros([max(Rs), 1], dtype=int)]
    )
    rsets = [randint[: Rs[n + 1], n:] for n in range(N - 1)] + [np.array([[0]])]

    t_linterfaces, t_rinterfaces = init_interfaces(tensors, rsets, N, device)

    # Create a validation set
    Xs_val = [
        torch.as_tensor(np.random.choice(I, int(val_size))).to(device) for I in Is
    ]
    ys_val = f(*[t[Xs_val].torch() for t in tensors])
    if ys_val.dim() > 1:
        assert ys_val.dim() == 2
        assert ys_val.shape[1] == 1
        ys_val = ys_val[:, 0]

    assert len(ys_val) == val_size
    norm_ys_val = torch.norm(ys_val)

    if verbose:
        print(
            "Cross-approximation over a {}D domain containing {:g} grid points:".format(
                N, tensors[0].numel()
            )
        )
    start = time.time()
    converged = False

    info = {"nsamples": 0, "eval_time": 0, "val_epss": [], "min": 0, "argmin": None}
    if record_samples:
        info["sample_positions"] = torch.zeros(0, N).to(device)
        info["sample_values"] = torch.zeros(0).to(device)

    def evaluate_function(
        j,
    ):  # Evaluate function over Rs[j] x Rs[j+1] fibers, each of size I[j]
        Xs = []
        for k, t in enumerate(tensors):
            if tensors[k].cores[j].dim() == 3:  # TT core
                V = torch.einsum(
                    "ai,ibj,jc->abc",
                    [t_linterfaces[k][j], t.cores[j], t_rinterfaces[k][j]],
                )
            else:  # CP factor
                V = torch.einsum(
                    "ai,bi,ic->abc",
                    [t_linterfaces[k][j], t.cores[j], t_rinterfaces[k][j]],
                )
            Xs.append(V.flatten())

        eval_start = time.time()
        evaluation = f(*Xs)
        if record_samples:
            info["sample_positions"] = torch.cat(
                (info["sample_positions"], torch.cat([x[:, None] for x in Xs], dim=1)),
                dim=0,
            )
            info["sample_values"] = torch.cat((info["sample_values"], evaluation))
        info["eval_time"] += time.time() - eval_start
        if _minimize:
            evaluation = np.pi / 2 - torch.atan(
                (evaluation - info["min"])
            )  # Function used by I. Oseledets for TT minimization in ttpy
            evaluation_argmax = torch.argmax(evaluation)
            eval_min = (
                torch.tan(np.pi / 2 - evaluation[evaluation_argmax]) + info["min"]
            )
            if info["min"] == 0 or eval_min < info["min"]:
                coords = np.unravel_index(
                    evaluation_argmax.cpu(), [Rs[j], Is[j], Rs[j + 1]]
                )
                info["min"] = eval_min
                info["argmin"] = (
                    tuple(lsets[j][coords[0]][1:])
                    + tuple([coords[1]])
                    + tuple(rsets[j][coords[2]][:-1])
                )

        # Check for nan/inf values
        if evaluation.dim() == 2:
            evaluation = evaluation[:, 0]
        invalid = torch.nonzero(torch.isnan(evaluation) | torch.isinf(evaluation))
        if len(invalid) > 0:
            invalid = invalid[0].item()
            raise ValueError(
                "Invalid return value for function {}: f({}) = {}".format(
                    function,
                    ", ".join(
                        "{:g}".format(x[invalid].detach().cpu().numpy()) for x in Xs
                    ),
                    f(*[x[invalid : invalid + 1][:, None] for x in Xs]).item(),
                )
            )

        V = torch.reshape(evaluation, [Rs[j], Is[j], Rs[j + 1]])
        info["nsamples"] += V.numel()
        return V

    # Sweeps
    for i in range(max_iter):

        if verbose:
            print("iter: {: <{}}".format(i, len("{}".format(max_iter)) + 1), end="")
            sys.stdout.flush()

        left_locals = []

        # Left-to-right
        for j in range(N - 1):

            # Update tensors for current indices
            V = evaluate_function(j)

            # QR + maxvol towards the right
            V = torch.reshape(V, [-1, Rs[j + 1]])  # Left unfolding
            Q, _ = torch.linalg.qr(V)
            if _minimize:
                local, _ = rect_maxvol(Q.detach().cpu().numpy(), maxK=Q.shape[1])
            else:
                local, _ = maxvol(Q.detach().cpu().numpy())
            V = torch.linalg.lstsq(Q[local, :].t(), Q.t()).solution.t()
            cores[j] = torch.reshape(V, [Rs[j], Is[j], Rs[j + 1]])
            left_locals.append(local)

            # Map local indices to global ones
            local_r, local_i = np.unravel_index(local, [Rs[j], Is[j]])
            lsets[j + 1] = np.c_[lsets[j][local_r, :], local_i]
            for k, t in enumerate(tensors):
                if t.cores[j].dim() == 3:  # TT core
                    t_linterfaces[k][j + 1] = torch.einsum(
                        "ai,iaj->aj",
                        [t_linterfaces[k][j][local_r, :], t.cores[j][:, local_i, :]],
                    )
                else:  # CP factor
                    t_linterfaces[k][j + 1] = torch.einsum(
                        "ai,ai->ai",
                        [t_linterfaces[k][j][local_r, :], t.cores[j][local_i, :]],
                    )

        # Right-to-left sweep
        for j in range(N - 1, 0, -1):

            # Update tensors for current indices
            V = evaluate_function(j)

            # QR + maxvol towards the left
            V = torch.reshape(V, [Rs[j], -1])  # Right unfolding
            Q, _ = torch.linalg.qr(V.t())
            if _minimize:
                local, _ = rect_maxvol(Q.detach().cpu().numpy(), maxK=Q.shape[1])
            else:
                local, _ = maxvol(Q.detach().cpu().numpy())
            V = torch.linalg.lstsq(Q[local, :].t(), Q.t()).solution
            cores[j] = torch.reshape(torch.as_tensor(V), [Rs[j], Is[j], Rs[j + 1]])

            # Map local indices to global ones
            local_i, local_r = np.unravel_index(local, [Is[j], Rs[j + 1]])
            rsets[j - 1] = np.c_[local_i, rsets[j][local_r, :]]
            for k, t in enumerate(tensors):
                if t.cores[j].dim() == 3:  # TT core
                    t_rinterfaces[k][j - 1] = torch.einsum(
                        "iaj,ja->ia",
                        [t.cores[j][:, local_i, :], t_rinterfaces[k][j][:, local_r]],
                    )
                else:  # CP factor
                    t_rinterfaces[k][j - 1] = torch.einsum(
                        "ai,ia->ia",
                        [t.cores[j][local_i, :], t_rinterfaces[k][j][:, local_r]],
                    )

        # Leave the first core ready
        V = evaluate_function(0)
        cores[0] = V

        # Evaluate validation error
        val_eps = torch.norm(ys_val - tn.Tensor(cores)[Xs_val].torch()) / norm_ys_val
        info["val_epss"].append(val_eps)
        if val_eps < eps:
            converged = True
        if verbose:  # Print status
            if _minimize:
                print("| best: {:.8g}".format(info["min"]), end="")
            else:
                print("| eps: {:.3e}".format(val_eps), end="")
            print(
                " | time: {:8.4f} | largest rank: {:3d}".format(
                    time.time() - start, max(Rs)
                ),
                end="",
            )
            if converged:
                print(" <- converged: eps < {}".format(eps))
            elif i == max_iter - 1:
                print(" <- max_iter was reached: {}".format(max_iter))
            else:
                print()
        if converged:
            break
        elif i < max_iter - 1 and kickrank is not None:  # Augment ranks
            newRs = Rs.copy()
            newRs[1:-1] = np.minimum(rmax, newRs[1:-1] + kickrank)
            for n in list(range(1, N)) + list(range(N - 1, 0, -1)):
                newRs[n] = min(newRs[n - 1] * Is[n - 1], newRs[n], Is[n] * newRs[n + 1])
            extra = np.hstack(
                [np.random.randint(0, Is[n + 1], [max(newRs), 1]) for n in range(N - 1)]
                + [np.zeros([max(newRs), 1], dtype=int)]
            )
            for n in range(N - 1):
                if newRs[n + 1] > Rs[n + 1]:
                    rsets[n] = np.vstack(
                        [rsets[n], extra[: newRs[n + 1] - Rs[n + 1], n:]]
                    )
            Rs = newRs
            t_linterfaces, t_rinterfaces = init_interfaces(
                tensors, rsets, N, device
            )  # Recompute interfaces

    if val_eps > eps and not _minimize and not suppress_warnings:
        logging.warning(
            "eps={:g} (larger than {}) when cross-approximating {}".format(
                val_eps, eps, function
            )
        )

    if verbose:
        print(
            "Did {} function evaluations, which took {:.4g}s ({:.4g} evals/s)".format(
                info["nsamples"],
                info["eval_time"],
                info["nsamples"] / info["eval_time"],
            )
        )
        print()

    ret = tn.Tensor(
        [c if isinstance(c, torch.Tensor) else torch.tensor(c) for c in cores]
    )
    if return_info:
        info["lsets"] = lsets
        info["rsets"] = rsets
        info["Rs"] = Rs
        info["left_locals"] = left_locals
        info["total_time"] = time.time() - start
        info["val_eps"] = val_eps
        return ret, info
    else:
        return ret



def fcn_batch_limited(fcn, max_batch=10**5, device="cpu"):
    ''' 
    To avoid memorry issues with large batch processing, 
    reduce computation into smaller batches 
    '''   
    def fcn_batch_truncated(x):
        batch_size = x.shape[0]
        fcn_values = torch.empty(batch_size).to(device)
        num_batch = batch_size//max_batch
        end_idx = 0
        for i in range(num_batch):
            start_idx = i*max_batch
            end_idx = (i+1)*max_batch
            fcn_values[start_idx:end_idx] = fcn(x[start_idx:end_idx].view(-1,x.shape[1]))
        if batch_size>end_idx:          
            fcn_values[end_idx:batch_size] = fcn(x[end_idx:batch_size].view(-1,x.shape[1]))
        return fcn_values
    return fcn_batch_truncated


def sample_random(batch_size, n_samples, domain, device="cpu"):
    ''' sample from the uniform distribution from the domain '''
    samples = torch.empty((batch_size,n_samples)).to(device)
    for i in range(len(domain)):
        samples[:,i] = domain[i][0] + (domain[i][-1]-domain[i][0])*torch.rand(size=(batch_size,n_samples))
    return samples


def stochastic_choice(M, alpha=0.99, rand_state=None, device="cpu"):
    '''
        Given pmf get the prioritized samples
        M: batch_size x n_samples x n  
        Treat each row of a matrix M[:,i,:] as a PMF and select a column per row according to it
    '''
    
    #filtering low pmf samples
    if rand_state is not None:
        torch.random.manual_seed(torch.randn(1).data)
    M= torch.abs(M) # batch_size x n_samples x n_site
    M_max, _ = torch.max(M,dim=-1) # batch_size x n_samples
    M_min, _ = torch.min(M,dim=-1)
    M_mean = M.mean(dim=-1)
    
    M_threshold = M_mean + alpha*(M_max-M_mean)
    
    M_max  = M_max[:,:,None].expand(-1,-1,M.shape[-1]) # batch_size x n_samples x n_site
    M_min  = M_min[:,:,None].expand(-1,-1,M.shape[-1]) # batch_size x n_samples x n_site
    M_mean  = M_mean[:,:,None].expand(-1,-1,M.shape[-1]) # batch_size x n_samples x n_site
    M_threshold  = M_threshold[:,:,None].expand(-1,-1,M.shape[-1]) # batch_size x n_samples x n_site
    
    M = M*(M>M_threshold)        
    M= M/(1e-9+M_max) # batch_size x n_samples x n_site

    M=M**(1/(1e-9+1-alpha))  # higher density is given higher importance
    M=M+1e-9
    M = M/(torch.sum(M, dim=-1)[:,:, None]) + 1e-9  # Normalize the pdf, batch_size x n_samples x n_site
    samples = torch.multinomial(M.view(-1,M.shape[-1]),1).view(M.shape[0],-1) # (batch_size*n_samples) x 1        
    if rand_state is not None:
        torch.random.set_rng_state(rand_state)
    return samples # batch_size x n_samples


def deterministic_choice(M,n_samples,idx_site, device="cpu"):
    """
        M: batch_size x n_samples x n_site

    """
    idx_site[:,1:] = (idx_site[:,1:]-idx_site[:,:-1]).abs()>0 
    idx_site[:,0] = 1
    
    bs = M.shape[0]
    n_site = M.shape[2]
    M = M*idx_site[:,:,None].expand(-1,-1,n_site) # make pmf corresponding to repeated indices to be zeo
    next_site = torch.zeros(bs,n_samples).to(device)
    previous_sample_id = torch.zeros(bs,n_samples).to(device)
    M2d = M.view(bs,-1) # bs x (n_samples*n_site)
    idx_k = torch.topk(M2d, k=n_samples, dim=-1)[1] # bs x n_samples
    next_site[:,:n_samples] = (idx_k).fmod(n_site) # which site next, bs x n_samples
    previous_sample_id[:,:n_samples] = (idx_k/n_site).long() # previous sample_id, b_size x n_samples
    return next_site.long(), previous_sample_id.long()


def contract_sites(tt_model, site_x, pro_x, device, eps=1e-6):
    '''
    Contract the cores of the tt-model given the weights for each discretization point 
    corresponding to each of the contracted site.
    p_x: a list of 1D tensor (probaility of each index of the site) 
    Return a contracted model 
    '''
    p_x = pro_x.clone()
    tt_cores = [core for core in tt_model.tt().cores[:]] # r_k x n_k x r_kn
    mat =  (tt_cores[site_x[0]]*(p_x[0].view(1,-1,1))).sum(dim=1) # r_k x 1 x r_kn
    for i,site in enumerate(site_x[1:]):
        mat_i = (tt_cores[site]*(p_x[1+i].view(1,-1,1))).sum(dim=1)
        mat = mat@mat_i
    state_id = site_x[-1]+1
    tt_cores_c = tt_cores[:site_x[0]] + tt_cores[state_id:]
    if site_x[-1] < len(tt_cores_c):
        tt_cores_c[site_x[0]] = torch.einsum('ij,jkl->ikl',mat,tt_cores_c[site_x[0]])
    else:
        last_state_id = site_x[0]-1
        tt_cores_c[last_state_id] = torch.einsum('ikj,jl->ikl',tt_cores_c[last_state_id],mat)

    tt_c_model = tnt.Tensor(tt_cores_c)
    # tt_c_model.round_tt(eps=eps)
    return tt_c_model.to(device)


def get_prob_x(mean_id, site_x, n_param, sigma=0.1, length=1, flag = 'uniform', device='cpu'):
    """
    param: 
    mean_id: index of the true parameter within parameter domain
    site_x: the dimension of the site to be contracted
    n_param: number of discretization points in parameter domain, (num_param x n_param)

    Given a rough guess of the true parameter, we assume the true parameter respects a probability distribution around the guess
    sigma: covariance if gaussian distribution
    length: width range if uniform distribution
    flag: 'gaussian' or 'uniform' #default is uniform
    
    return: p_x: probability of each discretization point in the parameter domain

    """

    num_param = len(site_x) 
    p_x = torch.zeros(num_param, n_param).to(device)

    if flag == 'gaussian':
        for id in range(num_param):
            mu = mean_id[id].cpu().numpy()
            values = np.arange(0, n_param.cpu().numpy())

            # compute probabilities and normalize
            probabilities = scipy.stats.norm.pdf(values, mu, sigma)
            probabilities /= np.sum(probabilities)
            p_x[id, :] = torch.tensor(probabilities).to(device)

    elif flag == 'uniform':
        length = max(int(length), 1)
        for id in range(num_param):
            mu = mean_id[id].cpu().numpy()
            probabilities = np.zeros(n_param)
            probabilities[mu:mu+length] = 1
            probabilities /= np.sum(probabilities)

            p_x[id, :] = torch.tensor(probabilities).to(device)
    return p_x


def prob_sites(tt_model, site_x, p_x, device, eps=1e-6):
    '''
    Contract the cores of the tt-model given the weights for each discretization point 
    corresponding to each of the contracted site.
    p_x: a list of 1D tensor (probaility of each index of the site) 
    Return a contracted model 
    '''

    tt_cores = [core for core in tt_model.tt().cores[:]] # r_k x n_k x r_kn
    mat =  (tt_cores[site_x[0]]*(p_x[0].view(1,-1,1))).sum(dim=1) # r_k x 1 x r_kn
    for i,site in enumerate(site_x[1:]):
        mat_i = (tt_cores[site]*(p_x[1+i].view(1,-1,1))).sum(dim=1)
        mat = mat@mat_i
    state_id = site_x[-1]+1
    tt_cores_c = tt_cores[:site_x[0]] + tt_cores[state_id:]
    # if site_x[-1] < len(tt_cores):
    tt_cores_c[site_x[0]] = torch.einsum('ij,jkl->ikl',mat,tt_cores_c[site_x[0]])
    # else:
    #     tt_cores_c[-1] = torch.einsum('ikj,jl->ikl',tt_cores_c[-1],mat)
    tt_c_model = tnt.Tensor(tt_cores_c)
    # tt_c_model.round_tt(eps=eps)
    return tt_c_model.to(device)


def contract_site(tt_model, site_x, p_x, device, eps=1e-6):
    '''
    Contract the cores of the tt-model given the weights for each discretization point 
    corresponding to each of the contracted site.
    p_x: a list of 1D tensor (probaility of each index of the site) 
    Return a contracted model 
    '''

    tt_cores = [core for core in tt_model.tt().cores[:]] # r_k x n_k x r_kn
    mat =  (tt_cores[site_x[0]](p_x[1+i].view(1,-1,1))).sum(dim=1) 
    for i,site in enumerate(site_x[1:]):
        mat_i = (tt_cores[site]*(p_x[1+i].view(1,-1,1))).sum(dim=1)
        mat = mat@mat_i
    tt_cores_c = tt_cores[:site_x[0]] + tt_cores[site_x[-1]:]
    if site_x[-1] < len(tt_cores):
        tt_cores_c[site_x[-1]] = torch.einsum('ij,jkl->ikl',mat,tt_cores_c[site_x[-1]])
    else:
        tt_cores_c[-1] = torch.einsum('ikj,jl->ikl',tt_cores_c[-1],mat)
    return tnt.Tensor(tt_cores_c).round_tt(eps=eps).to(device)



def condition_site(tt_cores, x, domain_x, n_discretization_x,device):
    '''
    Condition (or slicing) the cores of the tt-model given the values corresponding to a site. 
    Assumes x: batch_size x dim_x correspond to the first few cores
    Return the conditioned model:  tt_cores of shape batch_size x r_i x n_i x r_i' 
    '''
    batch_size = x.shape[0]
    dim_x = x.shape[1]
    # interpolate to find the corresponding slice for x
    idx_x = domain2idx(x,domain_x,device).view(batch_size,-1) # batch_size x dim_state
    x_1 = idx2domain(idx_x,domain_x,device) 
    dx = (x - x_1)
    idx_x_next = torch.clip(idx_x+torch.sign(dx),n_discretization_x*0,n_discretization_x-1).long() # next index (w.r.t disctretization)
    x_2 = idx2domain(idx_x_next,domain_x,device) 
    dx = torch.abs(dx)*1.0/(1e-6+(x_2-x_1).abs())
    # interpolate between the adjacent slices 
    for site in range(x.shape[-1]):
        tt_cores[site] = (tt_cores[site][:,idx_x[:,site],:]+dx[:,site].view(1,-1,1)*(tt_cores[site][:,idx_x_next[:,site],:]-tt_cores[site][:,idx_x[:,site],:]))
    # tranform cores so that it is: batch_size x r_k x -1 x r_kn     
    tt_cores_ext = [tt_cores[site][None,:,:,:].permute(2,1,0,3) for site in range(dim_x)]+[tt_cores[site][None,:,:,:].expand(batch_size,-1,-1,-1) for site in range(dim_x,len(tt_cores))]
    # Merge the slices corresponding to x into one core of size: b_state x 1 x 1 x r and then merge it to the non-sliced core b_state x 1 x n_a x r_a
    core_state = tt_cores_ext[0]
    for site in range(1,dim_x):
        core_state = torch.einsum('bijk,bkjl->bijl',core_state,tt_cores_ext[site])
    tt_cores_ext[dim_x] = torch.einsum('bi,ijk->bjk',core_state[:,0,0,:],tt_cores[dim_x])[:,None,:,:] # b_state x 1 x n_1 x r
    tt_cores_ext = tt_cores_ext[dim_x:]
    return tt_cores_ext # each core is of shape barch_size x r_ x n_ x r and the number of cores is len(tt_cores)-dim_x

def get_rights(tt_cores_ext, device):
    batch_size = tt_cores_ext[0].shape[0]
    # batch_size x r_k x r_kn
    tt_cores_action_summed =[torch.sum(core,dim=2) for core in tt_cores_ext] # batch_size x r_k x r_kn 
    rights = [torch.ones(batch_size,1).to(device).view(-1,1)] # each element is batch_size x r_k
    for site, summed_core in enumerate(tt_cores_action_summed[::-1]):
        r_ = torch.einsum('ijk,ik->ij',summed_core, rights[-1])
        rights.append(r_) # batch_size x r_k : batch_size x (r_k x r_kn) times (batch_size x r_kn)
    rights = rights[::-1] # batch_size x r_k
    return rights


def stochastic_top_k(tt_cores, domain, 
                         n_discretization_x=None, x=None, n_samples = 1, 
                         alpha=0.9, device="cpu", train=True):
    '''
    Consider x to be continuous (linear interpolation between tt-nodes)
    state: batch_size x dim_state
    Generate n_samples points from Q-function (treated as a joint PDF distribution ) 
    '''
    dim = len(tt_cores)
    
    if x is None: # no task variable means no conditioning
        batch_size = 1
        tt_cores_ext = [core[None,:,:,:] for core in tt_cores]
    else:
        if n_discretization_x is None:
            n_discretization_x = torch.tensor([len(domain[i]) for i in range(x.shape[-1])]).to(device)
        batch_size = x.shape[0]
        tt_cores_ext = condition_site(tt_cores=tt_cores[:], x=x, 
                            domain_x=domain[:x.shape[1]], 
                            n_discretization_x=n_discretization_x, 
                            device=device)

    rights = get_rights(tt_cores_ext,device=device)

    samples_idx = torch.zeros([batch_size, n_samples, len(tt_cores_ext)]).long().to(device) #
    lefts = torch.ones([batch_size, n_samples, 1]).to(device) # batch_size x n_samples x 1
    for site in range(len(tt_cores_ext)):
        fiber = torch.einsum('ijkl,il->ijk', (tt_cores_ext[site], rights[site+1])) # batch_size x r_k x n_k
        pmf = torch.einsum('ijk,ikl->ijl', (lefts, fiber)) # batch_size x n_samples x n_site 
        samples_idx[:,:, site] = stochastic_choice(M=pmf, alpha=alpha, rand_state=None, device=device ) # batch_size x n_samples
        core_sliced = (tt_cores_ext[site].permute([0,2,1,3])[torch.arange(tt_cores_ext[site].shape[0]).unsqueeze(1),samples_idx[:,:, site]]).permute([0,2,1,3])
        lefts = torch.einsum('ijk,ikjl->ijl', (lefts, core_sliced))

        
    samples = idx2domain(samples_idx.flatten(0,1),domain[-len(tt_cores_ext):], device).view(batch_size,n_samples,len(tt_cores_ext))
    if x is not None:
        samples_concat = torch.concat((x[:,None,:].expand(-1,n_samples,-1),samples),dim=-1)
    else:
        samples_concat = samples
    return samples_concat

def get_conditioned_tt(tt_cores, x, domain, device):

    n_discretization_x = torch.tensor([len(domain[i]) for i in range(x.shape[-1])]).to(device)

    # dim = x.shape[-1]
    batch_size = x.shape[0]
    domain_x=domain[:x.shape[1]]


    batch_size = x.shape[0]
    dim_x = x.shape[1]
    # interpolate to find the corresponding slice for x
    idx_x = domain2idx(x,domain_x,device).view(batch_size,-1) # batch_size x dim_state
    x_1 = idx2domain(idx_x,domain_x,device) 
    dx = (x - x_1)
    idx_x_next = torch.clip(idx_x+torch.sign(dx),n_discretization_x*0,n_discretization_x-1).long() # next index (w.r.t disctretization)
    x_2 = idx2domain(idx_x_next,domain_x,device) 
    dx = torch.abs(dx)*1.0/(1e-6+(x_2-x_1).abs())
    
    # interpolate between the adjacent slices 
    for site in range(x.shape[-1]):
        tt_cores[site] = (tt_cores[site][:,idx_x[:,site],:]+dx[:,site].view(1,-1,1)*(tt_cores[site][:,idx_x_next[:,site],:]-tt_cores[site][:,idx_x[:,site],:]))
    
    core_state = tt_cores[0] #1 x 1 x r_1
    for site in range(1, dim_x):
        core_state = torch.einsum('ijk,kml->iml',core_state, tt_cores[site]) # 1 x 1 x r_n
    
    tt_cores[dim_x] = torch.einsum('ijk,kml->iml',core_state, tt_cores[dim_x]) # 1 x n_k x r_k
    tt_cores = tt_cores[dim_x:]

    return tnt.Tensor(tt_cores).to(device)

        
        # # tranform cores so that it is: batch_size x r_k x -1 x r_kn     
        # tt_cores_ext = [tt_cores[site][None,:,:,:].permute(2,1,0,3) for site in range(dim_x)]+[tt_cores[site][None,:,:,:].expand(batch_size,-1,-1,-1) for site in range(dim_x,len(tt_cores))]
        # # Merge the slices corresponding to x into one core of size: b_state x 1 x 1 x r and then merge it to the non-sliced core b_state x 1 x n_a x r_a
        # core_state = tt_cores_ext[0]
        # for site in range(1,dim_x):
        #     core_state = torch.einsum('bijk,bkjl->bijl',core_state,tt_cores_ext[site])
        # tt_cores_ext[dim_x] = torch.einsum('bi,ijk->bjk',core_state[:,0,0,:],tt_cores[dim_x])[:,None,:,:] # b_state x 1 x n_1 x r
        # tt_cores_ext = tt_cores_ext[dim_x:]
        # return tt_cores_ext
        

    tt_cores_ext = condition_site(tt_cores=tt_cores[:], x=x, 
                            domain_x=domain[:x.shape[1]], 
                            n_discretization_x=n_discretization_x, 
                            device=device)
    return tt_cores_ext


def stochastic_top_k_sol(tt_cores, domain, 
                         n_discretization_x=None, x=None, n_samples = 1, 
                         alpha=0.9, device="cpu", train=True):
    '''
    Consider x to be continuous (linear interpolation between tt-nodes)
    state: batch_size x dim_state
    Generate n_samples points from Q-function (treated as a joint PDF distribution ) 
    
    Return both the samples and the index of the samples
    '''
    dim = len(tt_cores)
    
    if x is None: # no task variable means no conditioning
        batch_size = 1
        tt_cores_ext = [core[None,:,:,:] for core in tt_cores]
    else:
        if n_discretization_x is None:
            n_discretization_x = torch.tensor([len(domain[i]) for i in range(x.shape[-1])]).to(device)
        batch_size = x.shape[0]
        tt_cores_ext = condition_site(tt_cores=tt_cores[:], x=x, 
                            domain_x=domain[:x.shape[1]], 
                            n_discretization_x=n_discretization_x, 
                            device=device)

    rights = get_rights(tt_cores_ext,device=device)

    samples_idx = torch.zeros([batch_size, n_samples, len(tt_cores_ext)]).long().to(device) #
    lefts = torch.ones([batch_size, n_samples, 1]).to(device) # batch_size x n_samples x 1
    for site in range(len(tt_cores_ext)):
        fiber = torch.einsum('ijkl,il->ijk', (tt_cores_ext[site], rights[site+1])) # batch_size x r_k x n_k
        pmf = torch.einsum('ijk,ikl->ijl', (lefts, fiber)) # batch_size x n_samples x n_site 
        samples_idx[:,:, site] = stochastic_choice(M=pmf, alpha=alpha, rand_state=None, device=device ) # batch_size x n_samples
        core_sliced = (tt_cores_ext[site].permute([0,2,1,3])[torch.arange(tt_cores_ext[site].shape[0]).unsqueeze(1),samples_idx[:,:, site]]).permute([0,2,1,3])
        lefts = torch.einsum('ijk,ikjl->ijl', (lefts, core_sliced))

        
    samples = idx2domain(samples_idx.flatten(0,1),domain[-len(tt_cores_ext):], device).view(batch_size,n_samples,len(tt_cores_ext))
    if x is not None:
        samples_concat = torch.concat((x[:,None,:].expand(-1,n_samples,-1),samples),dim=-1)
    else:
        samples_concat = samples
    return samples_concat.flatten(0,1), samples_idx.flatten(0,1) # (batch_size* n_samples) x n_site

def compute_tt_visit(tt_cores, x_prefix, target_dim):
    """
    Compute partial sums for a TT tensor for given prefix values in dimensions before target_dim.

    Args:
        tt_tensor (tn.Tensor): The TT-tensor object.
        x_prefix (list): A list of fixed values for dimensions before target_dim.
        target_dim (int): The dimension to analyze.

    Returns:
        torch.Tensor: A tensor of size N (dimension size), where each element contains
                      the total sum of values for the corresponding target_dim index.
    """
    # Ensure the prefix length matches the target dimension
    # assert x_prefix.dim()-1 == target_dim, "Length of x_prefix must match target_dim"

    # Step 1: Forward propagation for dimensions up to target_dim
    if target_dim == 0:
        forward_middle_site = tt_cores[0] # batch_size x num_nodes x N x r_1
    else:
        index = x_prefix[:, :, 0].squeeze() # batch_size x 1 x num_nodes x 1

        if index.dim()==0: # for shape keeping
            index = index.unsqueeze(0)

        # forward_site = torch.gather(tt_cores[0], dim=2, index=index.long()) # batch_size x 1 x num_nodes x r_1
        forward_site = tt_cores[0][:, :, index,:] # batch_size x 1 x num_nodes x r_1
        for dim in range(1, target_dim):
            # G = tt_cores[dim]  # Get TT-core for current dimension
            index = x_prefix[:, :, dim].squeeze() # batch_size x 1 x num_nodes x 1
            if index.dim()==0: # for shape keeping
                index = index.unsqueeze(0)
            # G_slice = torch.gather(tt_cores[dim], dim=2, index=index.long())
            G_slice = tt_cores[dim][:, :, index,:] # batch_size x r_{dim-1} x num_nodes x r_{dim}
            forward_site = torch.einsum("bijr,brjl->bijl", forward_site, G_slice) # batch_size x 1 x num_nodes x r_{dim+1}

        forward_middle_site = torch.einsum("bijr,brkl->bjkl", forward_site, tt_cores[target_dim])  # batch_size x num_nodes x N x r_{target_dim}

    return forward_middle_site  # batch_size x n_samples x N x 1


def compute_tt_value(tt_cores, x_prefix, target_dim):
    """
    Compute partial sums for a TT tensor for given prefix values in dimensions before target_dim.

    Args:
        tt_tensor (tn.Tensor): The TT-tensor object.
        x_prefix (list): A list of fixed values for dimensions before target_dim.
        target_dim (int): The dimension to analyze.

    Returns:
        torch.Tensor: A tensor of size N (dimension size), where each element contains
                      the total sum of values for the corresponding target_dim index.
    """
    # Ensure the prefix length matches the target dimension
    # assert x_prefix.dim()-1 == target_dim, "Length of x_prefix must match target_dim"

    # Step 1: Forward propagation for dimensions up to target_dim
    if target_dim == 0:
        forward_middle_site = tt_cores[0] # batch_size x num_nodes x N x r_1
    else:
        index = x_prefix[:, :, 0].squeeze() # batch_size x 1 x num_nodes x 1

        if index.dim()==0: # for shape keeping
            index = index.unsqueeze(0)

        # forward_site = torch.gather(tt_cores[0], dim=2, index=index.long()) # batch_size x 1 x num_nodes x r_1
        forward_site = tt_cores[0][:, :, index,:] # batch_size x 1 x num_nodes x r_1
        for dim in range(1, target_dim):
            # G = tt_cores[dim]  # Get TT-core for current dimension
            index = x_prefix[:, :, dim].squeeze() # batch_size x 1 x num_nodes x 1
            # G_slice = torch.gather(tt_cores[dim], dim=2, index=index.long())
            if index.dim()==0: # for shape keeping
                index = index.unsqueeze(0)
            G_slice = tt_cores[dim][:, :, index,:] # batch_size x r_{dim-1} x num_nodes x r_{dim}
            forward_site = torch.einsum("bijr,brjl->bijl", forward_site, G_slice) # batch_size x 1 x num_nodes x r_{dim+1}

        forward_middle_site = torch.einsum("bijr,brkl->bjkl", forward_site, tt_cores[target_dim])  # batch_size x num_nodes x N x r_{target_dim}


    # Step 2: Backward propagation for dimensions after target_dim
    if target_dim == len(tt_cores) - 1:
        condition_site = forward_middle_site
    else:
        backward_site = tt_cores[-1]  # batch_size x r_{D-1} x N x 1
        backward_site = backward_site.sum(dim=2)  # batch_size x r_{D-1} x 1
        for dim in range(len(tt_cores) - 2, target_dim, -1):
            G = tt_cores[dim]  # Get TT-core for current dimension
            backward_site = torch.einsum("bijk,bkm->bijm", G, backward_site) # batch_size x r_{dim} x N x 1
            backward_site = backward_site.sum(dim=2)  # batch_size x r_{dim} x 1

        condition_site = torch.einsum("bijk,bkm->bijm", forward_middle_site, backward_site)  # batch_size x n_samples x N x 1


    return condition_site  # batch_size x n_samples x N x 1

def ucb_select(ttts, value_tt, visit_tt, domain=[], param_C = 1,
                x=None, n_samples=100, simulation_samples =10,
                n_discretization_x=None, alpha=0,
                device="cpu"):
    '''
    Leveraging the UCB formula to select the top-k ids

    Consider the states to be continuous (linear interpolation between tt-nodes)
    x: batch_size x dim_x (task variables)


    Return both the optimal solution and the corresponding index
    '''
    value_tt_cores = value_tt.tt().cores
    # visit_tt_cores = visit_tt.tt().cores
    dim = len(value_tt_cores)
    if x is None: # no task variable means no conditioning
        batch_size = 1
        value_tt_cores_ext = [core[None,:,:,:] for core in value_tt_cores]
        # visit_tt_cores_ext = [core[None,:,:,:] for core in visit_tt_cores]
    else:
        if n_discretization_x is None:
            n_discretization_x = torch.tensor([len(domain[i]) for i in range(x.shape[-1])]).to(device)
        batch_size = x.shape[0]
        value_tt_cores_ext = condition_site(tt_cores=value_tt_cores[:], x=x, 
                            domain_x=domain[:x.shape[-1]], 
                            n_discretization_x=n_discretization_x, 
                            device=device)

    if x is None:
        x_prefix = torch.zeros(batch_size,n_samples, len(value_tt_cores_ext)).to(device).long()
    else:
        x_prefix = x

    

    # sample_list = -1*torch.ones((batch_size, n_samples, len(value_tt_cores_ext))).long().to(device) # -1 as symbol for not updated
    leaf_list = []
    edge_list = torch.tensor([]).long()
    terminate_flag = False # check whether all branches are leaf nodes
    for site in range(0,len(value_tt_cores_ext)):
        # sample_list[:,original_edge_indices,site] = sample_branch_ids[None, :] #
        # sample_list.append(sample_branch_ids[None, :])
        
        # parent_visits = parent_visits.repeat(1, 1, values.shape[2], 1) # batch_size x 1 x n_i x 1
        n_sites = value_tt_cores_ext[site].shape[-2]
        values = compute_tt_value(value_tt_cores_ext, x_prefix[:, :, :site+1], site).round().long() # batch_size x n_samples x n_i x 1
        # visits = compute_tt_value(visit_tt_cores_ext, x_prefix[:, :, :site+1], site).long() # batch_size x n_samples x n_i x 1
        
        visit_tt_cores_ext = [core[None,:,:,:] for core in ttts.visit_tt[site].tt().cores]
        visits = compute_tt_visit(visit_tt_cores_ext, x_prefix[:, :, :site+1], site).round().long() # batch_size x n_samples x n_i x 1
        if site > 0:
            parent_visits = visits.sum(2).unsqueeze(2).expand(-1, -1, n_sites, -1) # batch_size x n_samples x n_i x 1
        else:
            parent_visits = ttts.visit_tt[0].sum() #n_samples #TODO: for conditioning case, should be value_tt[x].torch().sum()

        ucb_values = values / (visits + 1e-6) + param_C * torch.sqrt(2 * torch.log(parent_visits + 1) / (visits + 1e-6)) # batch_size x n_samples x n_i x 1
        ucb_values = ucb_values.flatten(1, 2) # batch_size x (n_samples * n_i) x 1
        
        # #Trick: adding a small random noise to break ties
        epsilon = 1e-6  # Small value to ensure the noise is minimal
        random_noise = torch.rand_like(ucb_values) * epsilon
        ucb_values = ucb_values + random_noise
        
        # Select top-k indices based on UCB values
        top_k = min(n_samples, ucb_values.shape[1])
        # Every retained parent keeps its best child (so that no selected branch is dropped before it is
        # counted); the remaining top_k - P slots are filled by the highest UCB values over all children.
        n_parents = values.shape[1]
        ucb_flat = ucb_values.squeeze(-1) # batch_size x (n_parents * n_i)
        best_idx = ucb_flat.view(batch_size, n_parents, n_sites).argmax(-1) \
            + torch.arange(n_parents, device=ucb_flat.device) * n_sites # batch_size x n_parents
        rest_flat = ucb_flat.clone()
        rest_flat.scatter_(1, best_idx, -float('inf'))
        rest_idx = torch.topk(rest_flat, k=top_k - n_parents, dim=-1)[1]
        idx_k = torch.cat([best_idx, rest_idx], dim=1).long() # batch_size x n_samples


        # Separate leaf nodes and edge nodes
        if torch.all(visits.flatten(1, 2)[:, idx_k[0], 0] == 0):
            leaf_ids = (visits.flatten(1, 2)[:, idx_k[0], 0] == 0).nonzero()[:, 1]
            terminate_flag = True
            edge_ids = torch.tensor([]).long()
        elif torch.all(visits.flatten(1, 2)[:, idx_k[0], 0] > 0):
            leaf_ids = torch.tensor([]).long()
            edge_ids = (visits.flatten(1, 2)[:, idx_k[0], 0] > 0).nonzero()[:, 1]
        else:
            leaf_ids = (visits.flatten(1, 2)[:, idx_k[0], 0] == 0).nonzero()[:, 1]  # n_leaf
            edge_ids = (visits.flatten(1, 2)[:, idx_k[0], 0] > 0).nonzero()[:, 1]  # n_edge

        

        # Process leaf nodes
        # samples_idx = torch.zeros([batch_size, simulation_samples, len(value_tt_cores_ext)]).long().to(device) #

        # map converted indices back to original indices
        site_id = idx_k.fmod(n_sites).long() #bs x n_samples
        

        global_leaf_id = idx_k[:, leaf_ids].fmod(n_sites).long()[0]
        global_edge_id = idx_k[:, edge_ids].fmod(n_sites).long()[0]

        
        # samples_idx[:, edge_idx, :site]=  samples_idx[:, edge_idx, :site][torch.arange(batch_size).unsqueeze(1),leaf_prev_id]
        if site>0:
            leaf_prev_id  = (idx_k[:, leaf_ids]/n_sites).long()
            edge_prev_id = (idx_k[:, edge_ids]/n_sites).long()
            site_prev_id = (idx_k/n_sites).long()
            # edge_list = torch.zeros([batch_size, n_samples, site+1]).long().to(device)

            global_edge_prev_id=  edge_list[:,:site][edge_prev_id[0]] #n_samples x sites
            global_leaf_prev_id = edge_list[:,:site][leaf_prev_id[0]] #n_samples x sites

            leaf_indices = torch.cat((global_leaf_prev_id, global_leaf_id[:, None]), dim=1)
            edge_list = torch.cat((global_edge_prev_id, global_edge_id[:, None]), dim=1)

            if len(leaf_indices) > 0:
                leaf_list.append(leaf_indices) #terminated node
        else:
            edge_list = global_edge_id[:, None]
            if len(global_leaf_id) > 0:
                leaf_list.append(global_leaf_id)
        


        # sample_branch_ids = sample_branch_ids[~torch.isin(sample_branch_ids, global_leaf_id)] # remove leaf indices

        x_prefix = edge_list
        x_prefix = x_prefix.repeat(batch_size, 1, 1) # batch_size x n_samples x ...

        num_branches = sum(len(sublist) for sublist in leaf_list)
        if terminate_flag or num_branches >= n_samples:
            # edge_indices = edge_list[None, :, :]
            break

    def create_random_simulation_samples(leaf_list):
        simulation_indices = []
        for leaf_idx in leaf_list:
            site = leaf_idx.shape[1] if leaf_idx.dim() > 1 else 1 # site index
            simulation_idx = leaf_idx.repeat_interleave(simulation_samples, 0)[:, None] if leaf_idx.dim() == 1 else leaf_idx.repeat_interleave(simulation_samples, 0)
            for dim in range(site, len(value_tt_cores_ext)):
                random_indices = torch.randint(0, len(domain[dim]), (leaf_idx.shape[0] * simulation_samples,)).to(device)
                # simulation_idx[:, leaf_idx.repeat_interleave(simulation_samples), dim] = random_indices
                
                simulation_idx = torch.cat((simulation_idx, random_indices[:, None]), dim=-1)
            simulation_indices.append(simulation_idx)

        simulation_indices = torch.cat(simulation_indices, dim=0)
        return simulation_indices # n_branches x n_dim
    
    def create_tt_simulation_samples(leaf_list):
        simulation_indices = []
        for leaf_idx in leaf_list:
            site = leaf_idx.shape[1] if leaf_idx.dim() > 1 else 1 # site index
            if site >= len(ttts.domain):
                simulation_indices.append(leaf_idx)
                break   
            leaf_samples = ttts.domain[site][leaf_idx]
            if leaf_idx.dim() == 1:
                leaf_samples = leaf_samples[:, None]
            _, sto_indices = stochastic_top_k_sol(tt_cores=value_tt.tt().cores, domain=ttts.domain,
                                                   x=leaf_samples, n_samples=simulation_samples, 
                                                   alpha=alpha, device=device)

            simulation_idx = leaf_idx.repeat_interleave(simulation_samples, 0)[:, None] if leaf_idx.dim() == 1 else leaf_idx.repeat_interleave(simulation_samples, 0)
            simulation_idx = torch.cat((simulation_idx, sto_indices), dim=-1)
            # for dim in range(site, len(value_tt_cores_ext)):
            #     random_indices = torch.randint(0, len(domain[dim]), (leaf_idx.shape[0] * simulation_samples,)).to(device)
            #     # simulation_idx[:, leaf_idx.repeat_interleave(simulation_samples), dim] = random_indices
                
            #     simulation_idx = torch.cat((simulation_idx, random_indices[:, None]), dim=-1)
            simulation_indices.append(simulation_idx)

        simulation_indices = torch.cat(simulation_indices, dim=0)
        return simulation_indices # n_branches x n_dim

    
    def create_block_simulation_samples(leaf_list):
        # simulation_indices = []
        n_dim = len(ttts.domain)

        # leaf_idx = leaf_list[0]
        leaf_idx = leaf_list[torch.randint(0, len(leaf_list), (1,))]
        site = leaf_idx.shape[1] if leaf_idx.dim() > 1 else 1
        rand_leaf_idx =  leaf_idx[torch.randint(0, len(leaf_idx), (1,))]


        n_free_dim = min(len(ttts.domain) - site, ttts.free_block_dim)

        n_fixed_dim = n_dim - n_free_dim - site

        fixed_dim = torch.arange(site, n_dim)[torch.randperm(n_dim - site)[:n_fixed_dim]]
        fixed_x = torch.randint(0, ttts.dim_state[0].long(), (n_fixed_dim,)).to(device)


        num_visits = ttts.dim_state[0]**n_free_dim
        block_indices = torch.zeros(num_visits, n_dim).long().to(device)
        block_indices[:, fixed_dim] = fixed_x.repeat(num_visits, 1)
        block_indices[:, :site] = rand_leaf_idx.repeat(num_visits, 1)
        free_dim = torch.tensor([i for i in range(len(ttts.domain)) if i not in torch.cat((torch.arange(0, site), fixed_dim))])
    
        # Generate block indices using torch.meshgrid

        free_ids = torch.cartesian_prod(*[torch.arange(ttts.dim_state[dim]) for dim in free_dim]).to(device)
        block_indices[:, free_dim] = free_ids[:, None] if len(free_dim) == 1 else free_ids

        # simulation_indices.append(block_indices)
        
        # simulation_indices = torch.cat(simulation_indices, dim=0)
        return block_indices

    # complete the simulation samples from leaf nodes (value=-1) in each instance in sample_list
    # simulation_indices = create_random_simulation_samples(leaf_list)[None, :, :]

    simulation_indices = create_tt_simulation_samples(leaf_list)[None, :, :]



    indices = torch.cat((edge_list[None, :, :], simulation_indices), dim=1) if (edge_list.shape[-1] == ttts.dim) else simulation_indices    
    samples = idx2domain(indices.flatten(0,1),domain[-len(value_tt_cores_ext):], device).view(batch_size,-1,len(value_tt_cores_ext))
    

    return leaf_list, edge_list, samples[0], indices[0] # n_samples x n_site


def ucb_select_test(value_tt, visit_tt, domain=[], param_C = 1,
                x=None, n_samples=100, 
                n_discretization_x=None, 
                device="cpu"):
    '''
    Leveraging the UCB formula to select the top-k ids

    Consider the states to be continuous (linear interpolation between tt-nodes)
    x: batch_size x dim_x (task variables)


    Return both the optimal solution and the corresponding index
    '''
    value_tt_cores = value_tt.tt().cores
    visit_tt_cores = visit_tt.tt().cores
    dim = len(value_tt_cores)
    if x is None: # no task variable means no conditioning
        batch_size = 1
        value_tt_cores_ext = [core[None,:,:,:] for core in value_tt_cores]
        visit_tt_cores_ext = [core[None,:,:,:] for core in visit_tt_cores]
    else:
        if n_discretization_x is None:
            n_discretization_x = torch.tensor([len(domain[i]) for i in range(x.shape[-1])]).to(device)
        batch_size = x.shape[0]
        value_tt_cores_ext = condition_site(tt_cores=value_tt_cores[:], x=x, 
                            domain_x=domain[:x.shape[-1]], 
                            n_discretization_x=n_discretization_x, 
                            device=device)
        
        visit_tt_cores_ext = condition_site(tt_cores=visit_tt_cores[:], x=x,
                            domain_x=domain[:x.shape[-1]],
                            n_discretization_x=n_discretization_x,
                            device=device)


    # rights = get_rights(tt_cores_ext, device=device)
    samples_idx = torch.zeros([batch_size, n_samples, len(value_tt_cores_ext)]).long().to(device) #

    if x is None:
        x_prefix = torch.zeros(batch_size,n_samples, len(value_tt_cores_ext)).to(device).long()
    else:
        x_prefix = x

    # parent_visits = torch.zeros(batch_size, 1, 1).to(device) # batch_size x 1 x 1
    for site in range(0,len(value_tt_cores_ext)):
        # parent_visits = parent_visits.repeat(1, 1, values.shape[2], 1) # batch_size x 1 x n_i x 1
        n_sites = value_tt_cores_ext[site].shape[-2]
        values = compute_tt_value(value_tt_cores_ext, x_prefix[:, :, :site+1], site) # batch_size x n_samples x n_i x 1
        visits = compute_tt_value(visit_tt_cores_ext, x_prefix[:, :, :site+1], site) # batch_size x n_samples x n_i x 1
        if site > 0:
            parent_visits = visits.sum(2).unsqueeze(2).expand(-1, -1, n_sites, -1) # batch_size x n_samples x n_i x 1
        else:
            parent_visits = visit_tt.sum() #n_samples #TODO: for conditioning case, should be value_tt[x].torch().sum()

        ucb_values = values / (visits + 1e-6) + param_C * torch.sqrt(2 * torch.log(parent_visits + 1) / (visits + 1e-6)) # batch_size x n_samples x n_i x 1
        ucb_values = ucb_values.flatten(1, 2) # batch_size x (n_samples * n_i) x 1
        
        #Trick: adding a small random noise to break ties
        epsilon = 1e-6  # Small value to ensure the noise is minimal
        random_noise = torch.rand_like(ucb_values) * epsilon
        ucb_values = ucb_values + random_noise
        

        idx_k = torch.topk(ucb_values, k=min(ucb_values.shape[1], n_samples), dim=-2)[1].long().squeeze(-1) # batch_size x n_samples
        # samples_idx[:,:,site] = idx_k

        site_idx = idx_k.fmod(n_sites).long().repeat(1,int(n_samples/n_sites)+1)[:,:n_samples]#((idx_k)/n_samples).floor().long()#( # top-k indices from the site, bs x n_samples
        samples_idx[:,:,site] = site_idx #batch_size x n_samples
        samples_prev_id  = (site_idx/n_sites).long()#(idx_k).fmod(n_samples).long()#idx_k - samples_idx[:,:,site]*n_sites # ((idx_k-1)/n_sites).long() # update previous site index
        samples_idx[:,:,:site]= samples_idx[:,:,:site][torch.arange(batch_size).unsqueeze(1),samples_prev_id]

        x_prefix[:,:,:site+1] = samples_idx[:,:,:site+1]
        # x_prefix[:,:,site] = idx_k

        # parent_visits = torch.take_along_dim(visits.squeeze(), idx_k[:, :, None], dim=2) # batch_size x n_samples x 1


        # parent_visits = visits[idx_k]

    samples = idx2domain(samples_idx.flatten(0,1),domain[-len(value_tt_cores_ext):], device).view(batch_size,n_samples,len(value_tt_cores_ext))
    # return samples, samples_idx # batch_size (n_task) x n_samples x n_site
    return samples[0], samples_idx[0] #, ucb_values[0][:n_samples] # n_samples x n_site, n_samples x n_site, n_samples x 1


def deterministic_top_k(tt_cores, domain=[], 
                x=None, n_samples=100, 
                n_discretization_x=None, 
                device="cpu", train=True):
    '''
    Consider the states to be continuous (linear interpolation between tt-nodes)
    x: batch_size x dim_x (task variables)
    Generate n_samples points from tt-model (treated as a joint PDF distribution ) corresponding to top-k max values
    The tt_cores are not assumed to be right orthogonalized (orthogonal model).
    If not, call canonlicalize(tt_model) prior to calling this method 
    This will speed up the process 
    '''
    dim = len(tt_cores)
    if x is None: # no task variable means no conditioning
        batch_size = 1
        tt_cores_ext = [core[None,:,:,:] for core in tt_cores]
    else:
        if n_discretization_x is None:
            n_discretization_x = torch.tensor([len(domain[i]) for i in range(x.shape[-1])]).to(device)
        batch_size = x.shape[0]
        tt_cores_ext = condition_site(tt_cores=tt_cores[:], x=x, 
                            domain_x=domain[:x.shape[-1]], 
                            n_discretization_x=n_discretization_x, 
                            device=device)


    # rights = get_rights(tt_cores_ext, device=device)
    samples_idx = torch.zeros([batch_size, n_samples, len(tt_cores_ext)]).long().to(device) #

    # pmf:  batch_size x 1 x n
    pmf = torch.linalg.norm(tt_cores_ext[0],dim=-1) # tt_cores_ext[0]: batch_size X 1 X n X r_1
    # pmf = torch.einsum('ijkr,ir->ijk',tt_cores_ext[0],rights[1]).abs()

    n_site_0 = tt_cores_ext[0].shape[-2]
    # samples_site:  batch_size x min(n_samples,n_site) 
    idx_k = torch.topk(pmf.view(batch_size,-1),k=min(n_samples,n_site_0),dim=-1)[1].fmod(n_site_0).long()
    if n_site_0 < n_samples: 
        samples_idx[:,:,0] = idx_k.repeat(1,int(n_samples/n_site_0)+1)[:,:n_samples] #batch_size x n_samples
    else:
        samples_idx[:,:,0] = idx_k
    # p_cum: batch_size x n_samples x r_1
    p_cum = (tt_cores_ext[0].permute([0,2,1,3])[torch.arange(batch_size).unsqueeze(1),idx_k]).permute([0,2,1,3])[:,0,:,:]

    for site in range(1,len(tt_cores_ext)):
        n_sites = tt_cores_ext[site].shape[-2]

        pmf_pre = torch.einsum('ijk,iklm->ijlm', (p_cum, tt_cores_ext[site])).flatten(1,2)#.view(batch_size,-1,tt_cores_ext[site].shape[-1]) # batch x n_site*n_samples x r_site
        pmf = torch.linalg.norm(pmf_pre,dim=-1) # batch x (n_site*n_samples) 
        # pmf = torch.einsum('ijr,ir->ij',pmf_pre,rights[site+1]).abs()
        idx_k = torch.topk(pmf, k=n_samples, dim=-1)[1].long() # bs x n_samples
        
        samples_idx[:,:,site] = idx_k.fmod(n_sites).long()#((idx_k)/n_samples).floor().long()#( # top-k indices from the site, bs x n_samples
        samples_prev_id  = (idx_k/n_sites).long()#(idx_k).fmod(n_samples).long()#idx_k - samples_idx[:,:,site]*n_sites # ((idx_k-1)/n_sites).long() # update previous site index
        samples_idx[:,:,:site]= samples_idx[:,:,:site][torch.arange(batch_size).unsqueeze(1),samples_prev_id]
        # p_cum: batch_size x  n_samples  x r_site 
        p_cum = pmf_pre[torch.arange(batch_size).unsqueeze(1),idx_k]

    
    samples = idx2domain(samples_idx.flatten(0,1),domain[-len(tt_cores_ext):], device).view(batch_size,n_samples,len(tt_cores_ext))
    if x is not None:
        samples_concat = torch.concat((x[:,None,:].expand(-1,n_samples,-1),samples),dim=-1)
    else:
        samples_concat = samples

    return samples_concat



def deterministic_top_k_sol(tt_cores, domain=[], 
                x=None, n_samples=100, 
                n_discretization_x=None, 
                device="cpu", train=True):
    '''
    Consider the states to be continuous (linear interpolation between tt-nodes)
    x: batch_size x dim_x (task variables)
    Generate n_samples points from tt-model (treated as a joint PDF distribution ) corresponding to top-k max values
    The tt_cores are not assumed to be right orthogonalized (orthogonal model).
    If not, call canonlicalize(tt_model) prior to calling this method 
    This will speed up the process 

    Return both the optimal solution and the corresponding index
    '''
    dim = len(tt_cores)
    if x is None: # no task variable means no conditioning
        batch_size = 1
        tt_cores_ext = [core[None,:,:,:] for core in tt_cores]
    else:
        if n_discretization_x is None:
            n_discretization_x = torch.tensor([len(domain[i]) for i in range(x.shape[-1])]).to(device)
        batch_size = x.shape[0]
        tt_cores_ext = condition_site(tt_cores=tt_cores[:], x=x, 
                            domain_x=domain[:x.shape[-1]], 
                            n_discretization_x=n_discretization_x, 
                            device=device)


    # rights = get_rights(tt_cores_ext, device=device)
    samples_idx = torch.zeros([batch_size, n_samples, len(tt_cores_ext)]).long().to(device) #

    # pmf:  batch_size x 1 x n
    pmf = torch.linalg.norm(tt_cores_ext[0],dim=-1) # tt_cores_ext[0]: batch_size X 1 X n X r_1
    # pmf = torch.einsum('ijkr,ir->ijk',tt_cores_ext[0],rights[1]).abs()

    n_site_0 = tt_cores_ext[0].shape[-2]
    # samples_site:  batch_size x min(n_samples,n_site) 
    idx_k = torch.topk(pmf.view(batch_size,-1),k=min(n_samples,n_site_0),dim=-1)[1].fmod(n_site_0).long()
    if n_site_0 < n_samples: 
        samples_idx[:,:,0] = idx_k.repeat(1,int(n_samples/n_site_0)+1)[:,:n_samples] #batch_size x n_samples
    else:
        samples_idx[:,:,0] = idx_k
    # p_cum: batch_size x n_samples x r_1
    p_cum = (tt_cores_ext[0].permute([0,2,1,3])[torch.arange(batch_size).unsqueeze(1),idx_k]).permute([0,2,1,3])[:,0,:,:]

    for site in range(1,len(tt_cores_ext)):
        n_sites = tt_cores_ext[site].shape[-2]

        pmf_pre = torch.einsum('ijk,iklm->ijlm', (p_cum, tt_cores_ext[site])).flatten(1,2)#.view(batch_size,-1,tt_cores_ext[site].shape[-1]) # batch x n_site*n_samples x r_site
        pmf = torch.linalg.norm(pmf_pre,dim=-1) # batch x (n_site*n_samples) 
        # pmf = torch.einsum('ijr,ir->ij',pmf_pre,rights[site+1]).abs()
        idx_k = torch.topk(pmf, k=n_samples, dim=-1)[1].long() # bs x n_samples
        
        samples_idx[:,:,site] = idx_k.fmod(n_sites).long()#((idx_k)/n_samples).floor().long()#( # top-k indices from the site, bs x n_samples
        samples_prev_id  = (idx_k/n_sites).long()#(idx_k).fmod(n_samples).long()#idx_k - samples_idx[:,:,site]*n_sites # ((idx_k-1)/n_sites).long() # update previous site index
        samples_idx[:,:,:site]= samples_idx[:,:,:site][torch.arange(batch_size).unsqueeze(1),samples_prev_id]
        # p_cum: batch_size x  n_samples  x r_site 
        p_cum = pmf_pre[torch.arange(batch_size).unsqueeze(1),idx_k]

    
    samples = idx2domain(samples_idx.flatten(0,1),domain[-len(tt_cores_ext):], device).view(batch_size,n_samples,len(tt_cores_ext))
    if x is not None:
        samples_concat = torch.concat((x[:,None,:].expand(-1,n_samples,-1),samples),dim=-1)
    else:
        samples_concat = samples

    return samples_concat.flatten(0,1), samples_idx.flatten(0,1) # (batch_size* n_samples) x n_site

def deterministic_top_k_id(tt_cores, domain=[], 
                x=None, n_samples=100, 
                n_discretization_x=None, 
                device="cpu", train=True):
    '''
    Consider the states to be continuous (linear interpolation between tt-nodes)
    x: batch_size x dim_x (task variables)
    Generate n_samples points from tt-model (treated as a joint PDF distribution ) corresponding to top-k max values
    The tt_cores are not assumed to be right orthogonalized (orthogonal model).
    If not, call canonlicalize(tt_model) prior to calling this method 
    This will speed up the process 
    '''
    dim = len(tt_cores)
    if x is None: # no task variable means no conditioning
        batch_size = 1
        tt_cores_ext = [core[None,:,:,:] for core in tt_cores]
    else:
        if n_discretization_x is None:
            n_discretization_x = torch.tensor([len(domain[i]) for i in range(x.shape[-1])]).to(device)
        batch_size = x.shape[0]
        tt_cores_ext = condition_site(tt_cores=tt_cores[:], x=x, 
                            domain_x=domain[:x.shape[-1]], 
                            n_discretization_x=n_discretization_x, 
                            device=device)


    # rights = get_rights(tt_cores_ext, device=device)
    samples_idx = torch.zeros([batch_size, n_samples, len(tt_cores_ext)]).long().to(device) #

    # pmf:  batch_size x 1 x n
    pmf = torch.linalg.norm(tt_cores_ext[0],dim=-1) # tt_cores_ext[0]: batch_size X 1 X n X r_1
    # pmf = torch.einsum('ijkr,ir->ijk',tt_cores_ext[0],rights[1]).abs()

    n_site_0 = tt_cores_ext[0].shape[-2]
    # samples_site:  batch_size x min(n_samples,n_site) 
    idx_k = torch.topk(pmf.view(batch_size,-1),k=min(n_samples,n_site_0),dim=-1)[1].fmod(n_site_0).long()
    if n_site_0 < n_samples: 
        samples_idx[:,:,0] = idx_k.repeat(1,int(n_samples/n_site_0)+1)[:,:n_samples] #batch_size x n_samples
    else:
        samples_idx[:,:,0] = idx_k
    # p_cum: batch_size x n_samples x r_1
    p_cum = (tt_cores_ext[0].permute([0,2,1,3])[torch.arange(batch_size).unsqueeze(1),idx_k]).permute([0,2,1,3])[:,0,:,:]

    for site in range(1,len(tt_cores_ext)):
        n_sites = tt_cores_ext[site].shape[-2]

        pmf_pre = torch.einsum('ijk,iklm->ijlm', (p_cum, tt_cores_ext[site])).flatten(1,2)#.view(batch_size,-1,tt_cores_ext[site].shape[-1]) # batch x n_site*n_samples x r_site
        pmf = torch.linalg.norm(pmf_pre,dim=-1) # batch x (n_site*n_samples) 
        # pmf = torch.einsum('ijr,ir->ij',pmf_pre,rights[site+1]).abs()
        idx_k = torch.topk(pmf, k=n_samples, dim=-1)[1].long() # bs x n_samples
        
        samples_idx[:,:,site] = idx_k.fmod(n_sites).long()#((idx_k)/n_samples).floor().long()#( # top-k indices from the site, bs x n_samples
        samples_prev_id  = (idx_k/n_sites).long()#(idx_k).fmod(n_samples).long()#idx_k - samples_idx[:,:,site]*n_sites # ((idx_k-1)/n_sites).long() # update previous site index
        samples_idx[:,:,:site]= samples_idx[:,:,:site][torch.arange(batch_size).unsqueeze(1),samples_prev_id]
        # p_cum: batch_size x  n_samples  x r_site 
        p_cum = pmf_pre[torch.arange(batch_size).unsqueeze(1),idx_k]

    
    return samples_idx.flatten(0,1) # (batch_size* n_samples) x n_site


def deterministic_top_k_TS(tt_cores, domain=[], 
                x=None, n_samples=100, 
                n_discretization_x=None, 
                device="cpu", train=True):
    '''
    Consider the states to be continuous (linear interpolation between tt-nodes)
    x: batch_size x dim_x (task variables)
    Generate n_samples points from tt-model (treated as a joint PDF distribution ) corresponding to top-k max values
    The tt_cores are not assumed to be right orthogonalized (orthogonal model).
    If not, call canonlicalize(tt_model) prior to calling this method 
    This will speed up the process 
    '''
    dim = len(tt_cores)
    if x is None: # no task variable means no conditioning
        batch_size = 1
        tt_cores_ext = [core[None,:,:,:] for core in tt_cores]
    else:
        if n_discretization_x is None:
            n_discretization_x = torch.tensor([len(domain[i]) for i in range(x.shape[-1])]).to(device)
        batch_size = x.shape[0]
        tt_cores_ext = condition_site(tt_cores=tt_cores[:], x=x, 
                            domain_x=domain[:x.shape[-1]], 
                            n_discretization_x=n_discretization_x, 
                            device=device)


    # rights = get_rights(tt_cores_ext, device=device)
    samples_idx = torch.zeros([batch_size, n_samples, len(tt_cores_ext)]).long().to(device) #

    # pmf:  batch_size x 1 x n
    pmf = torch.linalg.norm(tt_cores_ext[0],dim=-1) # tt_cores_ext[0]: batch_size X 1 X n X r_1
    # pmf = torch.einsum('ijkr,ir->ijk',tt_cores_ext[0],rights[1]).abs()

    n_site_0 = tt_cores_ext[0].shape[-2]
    # samples_site:  batch_size x min(n_samples,n_site) 
    idx_k = torch.topk(pmf.view(batch_size,-1),k=min(n_samples,n_site_0),dim=-1)[1].fmod(n_site_0).long()
    if n_site_0 < n_samples: 
        samples_idx[:,:,0] = idx_k.repeat(1,int(n_samples/n_site_0)+1)[:,:n_samples] #batch_size x n_samples
    else:
        samples_idx[:,:,0] = idx_k
    # p_cum: batch_size x n_samples x r_1
    p_cum = (tt_cores_ext[0].permute([0,2,1,3])[torch.arange(batch_size).unsqueeze(1),idx_k]).permute([0,2,1,3])[:,0,:,:]

    tmp_x = x.repeat(batch_size*n_samples,1) # (batch_size*n_samples) x dim_x

    for site in range(1,len(tt_cores_ext)):
        samples = idx2domain(idx_k[:, :, None].flatten(0,1),domain[:site], device).view(batch_size*n_samples, 1)
        #TODO: ensure x: batch_size is always 1
        tmp_x = torch.cat((tmp_x,samples),dim=-1).view(batch_size*n_samples,-1)
        n_discretization_tmp_x = torch.tensor([len(domain[i]) for i in range(tmp_x.shape[-1])]).to(device)
        tt_core = condition_site(tt_cores=tt_cores[:], x=tmp_x,
                            domain_x=domain[:tmp_x.shape[-1]], 
                            n_discretization_x=n_discretization_tmp_x, 
                            device=device)[0]
        
        n_sites = tt_core.shape[-2]
        pmf_pre = tt_core.view(-1,n_sites,tt_core.shape[-1])

        # n_sites = tt_core.shape[-2]

        pmf = torch.linalg.norm(pmf_pre,dim=-1) # (batch * n_samples) x n_site
        _, idx_k = torch.max(pmf, dim=1)
        idx_k = idx_k.view(batch_size,n_samples)
        
        samples_idx[:,:,site] = idx_k#.fmod(n_sites).long()#((idx_k)/n_samples).floor().long()#( # top-k indices from the site, bs x n_samples

    
    samples = idx2domain(samples_idx.flatten(0,1),domain[-len(tt_cores_ext):], device).view(batch_size,n_samples,len(tt_cores_ext))
    if x is not None:
        samples_concat = torch.concat((x[:,None,:].expand(-1,n_samples,-1),samples),dim=-1)
    else:
        samples_concat = samples

    # return samples_concat
    return samples_concat.flatten(0,1), samples_idx.flatten(0,1) # (batch_size* n_samples) x n_site


def concat_tt_model(tt_model, tt_model_2, device='cpu'):
    '''
    Concatenate two tt-models
    '''
    tt_cores = tt_model.tt().cores[:]
    tt_cores_2 = tt_model_2.tt().cores[:]
    
    for i in range(len(tt_cores)):
        tt_cores[i] = torch.cat((tt_cores[i],tt_cores_2[i]),dim=-1)
    
    return tnt.Tensor(tt_cores).to(device)


def get_tt_max(tt_model, domain, n_samples=100, deterministic=True, alpha=0.9, device="cpu"):
    '''
    Note: max is w.r.t the absolute value
    find the pseudo-max and argmax of a tt-model (absolute max) in a stochastic way
    '''
    tt_model_o =  tt_canonicalize(tt_model)
    tt_cores = tt_model_o.tt().cores[:]
    # Warm-up for mass sampling
    if deterministic:
        samples = deterministic_top_k(tt_cores=tt_cores, 
                        n_samples=n_samples, 
                        domain=domain, 
                        device=device)
    else:
        samples = stochastic_top_k(tt_cores=tt_cores, 
                        n_samples=n_samples, alpha=alpha,
                        domain=domain, device=device)
    samples_idx = domain2idx(samples.flatten(0,1),domain,device)
    values = get_elements(tt_model_o,samples_idx)
    idx = torch.argmax(torch.abs(values)) # batch_size 
    best_value = values[idx]

    return best_value, samples_idx[idx].view(-1) # max, argmax


def extend_cores(tt_cores, site, n_cores, d,  device='cpu'):
        ''' 
        Given a list of tt_cores add n_cores of identity cores starting at the given site
        d is a list containing dimension of those cores modes (size: n_cores)
        '''
        site = min(site,len(tt_cores))
        if site==0:
            r = 1
            # base_cores_left = []
            # base_cores_right = tt_cores[:]

        elif (site == (len(tt_cores))) or (site==-1):
            r = 1
            site=len(tt_cores)

        else:
            r = tt_cores[site-1].shape[-1]

        base_cores_left = tt_cores[:site]
        base_cores_right = tt_cores[site:]        
        id_action = torch.eye(r)[:,None,:].to(device)
        dummy_cores = [id_action.expand(-1, d[i],-1).to(device) for i in range(n_cores)]
        cores = base_cores_left + dummy_cores + base_cores_right
        return cores

def extend_model(tt_model, site, n_cores, d, device='cpu'):
        ''' 
        Given a list of tt_cores add n_cores of identity cores starting at the given site
        d is a list containing dimension of those cores modes (size: n_cores)
        '''
        tt_cores = tt_model.tt().cores[:]
        cores =  extend_cores(tt_cores, site, n_cores, d,  device)
        return tnt.Tensor(cores).to(device)

def get_tt_bounds(tt_model,domain,device="cpu"):
    tt_model_1 = tt_model.clone()
    bound_1, idx_1 = get_tt_max(tt_model, domain, device=device)
    bound_1  =  get_elements(tt_model,idx_1.view(1,-1)).item()
    tt_model_2 = tt_model_1-bound_1
    tt_model_2.round_tt(eps=1e-9)
    bound_2, idx_2 = get_tt_max(tt_model_2.to(device),domain, device=device)
    bound_2  = get_elements(tt_model,idx_2.view(1,-1)).item()
    upper_bound = bound_1 if (bound_1>bound_2) else bound_2
    lower_bound = bound_1 if (bound_1<bound_2) else bound_2
    return (lower_bound,upper_bound)


def normalize_tt(tt_model, domain, lb=1., ub=100., 
                    auto_bound=True,canonicalize=True,
                    device="cpu"):
    lower_bound, upper_bound  = get_tt_bounds(tt_model, domain, device=device)
    if auto_bound:
        lb = 1 + upper_bound - lower_bound
        tt_model_out = lb + (tt_model.to("cpu")-lower_bound)
    else:
        tt_model_out = lb + (tt_model.to("cpu")-lower_bound)*((ub-lb)/(upper_bound-lower_bound))
    if canonicalize:
        tt_model_out = tt_canonicalize(tt_model_out)
    else:
        tt_model_out.round_tt(eps=1e-9) # not necessary
    return tt_model_out.to(device)

def tt_canonicalize(tt_model,site=0):
    ''' 
    Return an  orthogonalized tt-model at site. 
    For i>site, torch.einsum('ijk,ljk->il',Core[i],Core[i]) will be identity matrix
    '''
    tt_model_o = tt_model.clone()
    tt_model_o.orthogonalize(site)
    return tt_model_o
