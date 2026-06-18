#
# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Teng Xue <teng.xue@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import torch
import numpy as np
torch.set_default_dtype(torch.float64)
device = "cuda" if torch.cuda.is_available() else "cpu"

def Rosenbrock_2D(a=1, b=100, alpha=1):
    '''
        a 2D version of the Rosenbrock function with fixed coefficients (a,b)
        https://en.wikipedia.org/wiki/Rosenbrock_function
    '''
    def cost(x): 
        result = b*(x[:,1]-x[:,0]**2)**2 + (x[:,0]-a)**2
        return result

    def pdf(x):
        return torch.exp(-alpha*cost(x))

    return pdf, cost


def Rosenbrock_4D(alpha=1):
    '''
        a 4D version of the Rosenbrock function with coefficients considered as variables of the function
        a=1, b=100 represents the standard 2D Rosenbrock function
        https://en.wikipedia.org/wiki/Rosenbrock_function
    '''
    def cost(x): 
        result = x[:,1]*(x[:,3]-x[:,2]**2)**2 + (x[:,2]-x[:,0])**2
        return result

    def pdf(x):
        return torch.exp(-alpha*cost(x)) 

    return pdf, cost


def Rosenbrock_nD(a, b, n=2,alpha=1):
    '''
        nD version of Rosenbrock function: https://en.wikipedia.org/wiki/Rosenbrock_function
        actual minima is at (a, a^2, a, a^2,...,a, a^2). 
        Domain: [-2,2]
    '''

    def cost(x):
        # a = x[:,0]
        # b = x[:,1]
        # y = x[:,2:] 
        y = x
        result = 0.
        for i in range(y.shape[1]-1):
            result = result+b*(y[:,i+1]-y[:,i]**2)**2 + (y[:,i]-a)**2
        return result

    def pdf(x):
        return torch.exp(-alpha*cost(x))

    return pdf, cost

def Rosenbrock_nD_2(n=2,alpha=1):
    '''
        nD version of Rosenbrock function: https://en.wikipedia.org/wiki/Rosenbrock_function
        actual minima is at (a, a^2, a, a^2,...,a, a^2). 
        Domain: [-2,2]
    '''
    assert ((n%2)==0 and n>2), 'n has to be even number greater than 2'

    def cost(x):
        a = x[:,0]
        b = x[:,1]
        y = x[:,2:] 
        result = 0.
        for i in range(int(y.shape[1]/2)):
            result = result+b*(y[:,2*i+1]-y[:,2*i]**2)**2 + (y[:,2*i]-a)**2
        return result

    def pdf(x):
        return torch.exp(-alpha*cost(x))+1e-9

    return pdf, cost

def diag_func_2D(alpha=1):
    def cost(x):
        # flag = (x[:, 0] == x[:, 1]).float() + (x[:, 0] == -1*x[:, 1]).float()
        # flag = (abs(x[:, 0] - x[:, 1])<0.5).float()
        # result = (1-flag.float())*(x[:, 0]**2 + x[:, 1]**2)
        # return result

        x0, x1 = x[:, 0], x[:, 1]
        # return 0.5 * (x0**2 - 2*x0*x1 + x1**2) + 0.1 * (x0**2 + x1**2)


        x_proj = (x0 + x1) / np.sqrt(2)
        res1 = torch.where(
            x_proj < 0,
            -(0.8*x_proj + 2)**2,
            -(0.8*x_proj - 2)**2
        )

        return 0.5 * res1 + 0.1 * (x0**2 + x1**2) +2

    def pdf(x):
        result = torch.exp(-cost(x))
        return result
    return pdf, cost


def mix_func_2D(alpha=1):
    def cost(x):
        # flag = (x[:, 0] == x[:, 1]).float() + (x[:, 0] == -1*x[:, 1]).float()
        # flag = (abs(x[:, 0] - x[:, 1])<0.5).float()
        # result = (1-flag.float())*(x[:, 0]**2 + x[:, 1]**2)
        # return result

        x0, x1 = x[:, 0], x[:, 1]
        # return 0.5 * (x0**2 - 2*x0*x1 + x1**2) + 0.1 * (x0**2 + x1**2)

        res1 = abs(-(x0- 1)**2 + 1) + abs(-(x0)**2 + 1)
        return 0.5*res1 + 0.1 * (x0**2 + x1**2) +8

    def pdf(x):
        result = torch.exp(-cost(x))
        return result
    return pdf, cost

def diag_func_2D_2(alpha=1):
    def cost(x):
        x0_raw, x1_raw = x[:, 0], x[:, 1]
        sqrt2_inv = 1 / np.sqrt(2)

        # 45° rotation
        x0_45 = sqrt2_inv * (x0_raw - x1_raw)

        cond1 = (x0_45 >= -7) & (x0_45 < -3)
        cond2 = (x0_45 >= -3) & (x0_45 < -1)
        cond3 = (x0_45 >= -1) & (x0_45 <= 1)
        cond4 = (x0_45 > 1) & (x0_45 <= 3)
        cond5 = (x0_45 > 3) & (x0_45 <= 7)

        valley_45 = torch.zeros_like(x0_45)
        valley_45[cond1] = torch.abs(-(x0_45[cond1] + 5)**2)
        valley_45[cond2] = torch.abs(-(x0_45[cond2] + 2)**2)
        valley_45[cond3] = torch.abs(-(x0_45[cond3] - 1)**2)
        valley_45[cond4] = torch.abs(-(x0_45[cond4] - 2)**2)
        valley_45[cond5] = torch.abs(-(x0_45[cond5] - 5)**2)

        # 135° rotation
        x0_135 = sqrt2_inv * (x0_raw + x1_raw)

        cond1b = (x0_135 >= -7) & (x0_135 < -3)
        cond2b = (x0_135 >= -3) & (x0_135 < -1)
        cond3b = (x0_135 >= -1) & (x0_135 <= 1)
        cond4b = (x0_135 > 1) & (x0_135 <= 3)
        cond5b = (x0_135 > 3) & (x0_135 <= 7)

        valley_135 = torch.zeros_like(x0_135)
        valley_135[cond1b] = torch.abs((x0_135[cond1b] + 5)**2)
        valley_135[cond2b] = torch.abs((x0_135[cond2b] + 2)**2 )
        valley_135[cond3b] = torch.abs((x0_135[cond3b] - 1)**2)
        valley_135[cond4b] = torch.abs((x0_135[cond4b] - 2)**2)
        valley_135[cond5b] = torch.abs((x0_135[cond5b] - 5)**2)

        valley = valley_45 + 0.5*valley_135


        regularizer = 0.1 * (x0_45**2 + x0_135**2)

        return 0.5 * valley + regularizer + 8

    def pdf(x):
        return torch.exp(-cost(x))

    return pdf, cost



def diag_func_2D_1(alpha=1):
    def cost(x):
        x0_raw, x1_raw = x[:, 0], x[:, 1]
        sqrt2_inv = 1 / np.sqrt(2)

        # Apply 45° rotation
        x0 = sqrt2_inv * (x0_raw - x1_raw)
        x1 = sqrt2_inv * (x0_raw + x1_raw)

        # Piecewise conditions on rotated x0
        cond1 = (x0 >= -7) & (x0 < -3)
        cond2 = (x0 >= -3) & (x0 < -1)
        cond3 = (x0 >= -1) & (x0 <= 1)
        cond4 = (x0 > 1) & (x0 <= 3)
        cond5 = (x0 > 3) & (x0 <= 7)

        valley = torch.zeros_like(x0)

        # Use torch.abs, not Python abs!
        valley[cond1] = torch.abs(-(x0[cond1] + 5)**2 + 5)
        valley[cond2] = torch.abs(-(x0[cond2] + 2)**2 + 5)
        valley[cond3] = torch.abs(-(x0[cond3]-1)**2 +5)
        valley[cond4] = torch.abs(-(x0[cond4] - 2)**2 + 5)
        valley[cond5] = torch.abs(-(x0[cond5] - 5)**2 + 5)

        

        return 0.5 * valley + 0.1 * (x0**2 + x1**2) + 8

    def pdf(x):
        return torch.exp(-cost(x))

    return pdf, cost



def mix_func_2D(alpha=1):
    import math

    def cost(x):
        x0_raw, x1_raw = x[:, 0], x[:, 1]

        # Apply 45-degree rotation
        sqrt2_inv = 1 / math.sqrt(2)
        x0 = sqrt2_inv * (x0_raw - x1_raw)
        x1 = sqrt2_inv * (x0_raw + x1_raw)

        # Piecewise conditions on rotated x0
        cond1 = (x0 >= -7) & (x0 < -3)
        cond2 = (x0 >= -3) & (x0 < 1)
        cond3 = (x0 >= 1) & (x0 <= 5)
        cond4 = (x0 > 5) & (x0 <= 9)
        cond5 = (x0 > 9) & (x0 <= 13)

        valley = torch.zeros_like(x0)
        noise = torch.zeros_like(x0)

        valley[cond1] = torch.abs(-(x0[cond1] + 5)**2 + 8) 
        valley[cond2] = torch.abs(-(x0[cond2] + 1)**2 + 3) 
        valley[cond3] = torch.abs(-(x0[cond3] - 3)**2 + 4) 
        valley[cond4] = torch.abs(-(x0[cond4] - 7)**2 + 5) 
        valley[cond5] = torch.abs(-(x0[cond5] - 11)**2 + 10) 

        noise[cond1] = ((x0[cond1] + 5)**2 + (x1[cond1] + 5)**2)
        noise[cond2] = ((x0[cond2] + 1)**2 + (x1[cond2] + 1)**2)
        noise[cond3] = ((x0[cond3] - 3)**2 + (x1[cond3] - 3)**2)
        noise[cond4] = ((x0[cond4] - 7)**2 + (x1[cond4] - 7)**2)
        noise[cond5] = ((x0[cond5] - 11)**2 + (x1[cond5] - 11)**2)

        return 0.5 * valley + 0.1 * noise


    def pdf(x):
        return torch.exp(-cost(x))

    return pdf, cost




def Himmelblaue_2D(alpha=1,a=11,b=7):
    '''
        a 2D function: https://en.wikipedia.org/wiki/Himmelblau%27s_function
        cost(x,y)=(x^2+y-11)^2+(x+y^2-7)^2
        Domain: [-5,5]
    '''
    def cost(x): 
        result = (x[:,0]**2+x[:,1]-a)**2 + (x[:,0]+x[:,1]**2-b)**2
        return result

    def pdf(x): # Cost-to-PDF transformation
        return torch.exp(-alpha*cost(x)) # or use:  1/(eps+cost(x))
    
    return pdf, cost



def Himmelblaue_4D(alpha=1):
    '''
        a 4D version of the Himmelblaue2D function with coefficients considered as variables of the function
        cost(a,b,x,y)=(x^2+y-a)^2+(x+y^2-b)^2
        a=11, b=7 represents the standard 2D Himmelblaue function
    '''
    def cost(x): 
        result = (x[:,2]**2+x[:,3]-x[:,0])**2 + (x[:,2]+x[:,3]**2-x[:,1])**2 #11, 7
        return result

    def pdf(x):
        return torch.exp(-alpha*cost(x)) + 1e-9 #or use: 1/(eps+cost(x))#

    return pdf, cost


def gmm(n=2,nmix=3,L=1,mx_coef=None,mu=None,s=0.1,device='cpu'):
    """
        Mixture of spherical Gaussians (un-normalized)
        nmix: number of mixture coefficients
        n: dimension of the domain
        s: variance
        mu: the centers assumed to be in : [-L,L]^n
    """
    n_sqrt = torch.sqrt(torch.tensor([n])).to(device)
    if mx_coef is None: # if centers and mixture coef are not given, generate them randomly
        mx_coef = torch.rand(nmix)
        mx_coef = mx_coef/torch.sum(mx_coef)
        mu = ((torch.rand(nmix,n)-0.5)*2*L).to(device)

    def pdf(x):
        result = torch.tensor([0]).to(device)
        for k in range(nmix):
            l = torch.linalg.norm(mu[k]-x, dim=1)/n_sqrt
            result = result + mx_coef[k]*torch.exp(-(l/s)**2)
        return result 

    def cost(x):
        return 1.-pdf(x)

    return pdf, cost


def sine_nD(n=2, alpha=1, device='cpu'):
    "an nD sinusoidal surface"
    n_sqrt = torch.sqrt(torch.tensor([n])).to(device)
    def pdf(x): 
        return 0.49*(1.001+torch.sin(4*torch.pi*torch.linalg.norm(x,dim=1)/n_sqrt))

    def cost(x):
        return 1.-pdf(x)

    return pdf, cost


############################################################################################################
# Other functions
############################################################################################################

import torch

def Ackley_2D(a=20, b=0.2, c=2 * torch.pi, alpha=1, device='cpu'):
    '''
        A 2D version of the Ackley function.
        https://en.wikipedia.org/wiki/Ackley_function
    '''
    def cost(x):
        term1 = -a * torch.exp(-b * torch.sqrt(0.5 * (x[:, 0]**2 + x[:, 1]**2)))
        term2 = -torch.exp(0.5 * (torch.cos(c * x[:, 0]) + torch.cos(c * x[:, 1])))
        result = term1 + term2 + a + torch.exp(torch.tensor(1.0))
        return result

    def pdf(x):
        return torch.exp(-alpha * cost(x))

    def get_optima():
        optimal_x = torch.tensor([[0.0, 0.0]])
        optimal_value = cost(optimal_x)
        return optimal_x, optimal_value

    return pdf, cost, get_optima

def Ackley_nD(n, a=20, b=0.2, c=2 * torch.pi, alpha=1):
    '''
        nD version of the Ackley function.
        Actual minima is at (0,0,...,0).
        Domain: [-32.768, 32.768]
    '''
    def cost(x):
        n = x.shape[1]
        term1 = -a * torch.exp(-b * torch.sqrt(torch.sum(x**2, dim=1) / n))
        term2 = -torch.exp(torch.sum(torch.cos(c * x), dim=1) / n)
        result = term1 + term2 + a + torch.exp(torch.tensor(1.0))
        return result

    def pdf(x):
        return torch.exp(-alpha * cost(x))

    def get_optima():
        optimal_x = torch.zeros(1, n)
        optimal_value = cost(optimal_x)
        return optimal_x, optimal_value

    return pdf, cost, get_optima

def Griewank_2D(alpha=1, device='cpu'):
    '''
        A 2D version of the Griewank function.
        https://en.wikipedia.org/wiki/Griewank_function
    '''
    def cost(x):
        term1 = (x[:, 0]**2 + x[:, 1]**2) / 4000
        term2 = torch.cos(x[:, 0]) * torch.cos(x[:, 1] / torch.sqrt(torch.tensor(2.0)))
        result = term1 - term2 + 1
        return result

    def pdf(x):
        return torch.exp(-alpha * cost(x))

    def get_optima():
        optimal_x = torch.tensor([[0.0, 0.0]])
        optimal_value = cost(optimal_x)
        return optimal_x, optimal_value

    return pdf, cost, get_optima

def Griewank_nD(n, alpha=1, device='cpu'):
    '''
        nD version of the Griewank function.
        Actual minima is at (0, 0, ..., 0).
        Domain: [-600, 600]
    '''
    def cost(x):
        term1 = torch.sum(x**2, dim=1).to(device) / 4000
        term2 = torch.prod(torch.cos(x / torch.sqrt(torch.arange(1, x.shape[1] + 1)).to(x.device)).to(device), dim=1)
        result = term1 - term2 + 1
        return result

    def pdf(x):
        return torch.exp(-alpha * cost(x))

    def get_optima():
        optimal_x = torch.zeros(1, n)
        optimal_value = cost(optimal_x)
        return optimal_x, optimal_value

    return pdf, cost, get_optima

def Rastrigin_2D(A=10, alpha=1, device='cpu'):
    '''
        A 2D version of the Rastrigin function.
        https://en.wikipedia.org/wiki/Rastrigin_function
    '''
    def cost(x):
        result = A * 2 + (x[:, 0]**2 - A * torch.cos(2 * torch.pi * x[:, 0])) + \
                 (x[:, 1]**2 - A * torch.cos(2 * torch.pi * x[:, 1]))
        return result

    def pdf(x):
        return torch.exp(-alpha * cost(x))

    def get_optima():
        optimal_x = torch.tensor([[0.0, 0.0]])
        optimal_value = cost(optimal_x)
        return optimal_x, optimal_value

    return pdf, cost, get_optima

def Rastrigin_nD(n, A=10, alpha=1, device='cpu'):
    '''
        nD version of the Rastrigin function.
        Actual minima is at (0, 0, ..., 0).
        Domain: [-5.12, 5.12]
    '''
    def cost(x):
        result = A * x.shape[1] + torch.sum(x**2 - A * torch.cos(2 * torch.pi * x), dim=1)
        return result

    def pdf(x):
        return torch.exp(-alpha * cost(x))

    def get_optima():
        optimal_x = torch.zeros(1, n)
        optimal_value = cost(optimal_x)
        return optimal_x, optimal_value

    return pdf, cost, get_optima

def Levy_2D(alpha=1, devic='cpu'):
    '''
        A 2D version of the Levy function.
        https://en.wikipedia.org/wiki/Test_functions_for_optimization
    '''
    def cost(x):
        w1 = 1 + (x[:, 0] - 1) / 4
        w2 = 1 + (x[:, 1] - 1) / 4
        term1 = torch.sin(torch.pi * w1)**2
        term2 = (w1 - 1)**2 * (1 + 10 * torch.sin(torch.pi * w1 + 1)**2)
        term3 = (w2 - 1)**2 * (1 + torch.sin(2 * torch.pi * w2)**2)
        result = term1 + term2 + term3
        return result

    def pdf(x):
        return torch.exp(-alpha * cost(x))

    def get_optima():
        optimal_x = torch.tensor([[1.0, 1.0]])
        optimal_value = cost(optimal_x)
        return optimal_x, optimal_value

    return pdf, cost, get_optima

def Levy_nD(n, alpha=1, device='cpu'):
    '''
        nD version of the Levy function.
        Actual minima is at (1, 1, ..., 1).
        Domain: [-10, 10]
    '''
    def cost(x):
        w = 1 + (x - 1) / 4
        term1 = torch.sin(torch.pi * w[:, 0])**2
        term2 = torch.sum((w[:, :-1] - 1)**2 * (1 + 10 * torch.sin(torch.pi * w[:, :-1] + 1)**2), dim=1)
        term3 = (w[:, -1] - 1)**2 * (1 + torch.sin(2 * torch.pi * w[:, -1])**2)
        result = term1 + term2 + term3
        return result

    def pdf(x):
        return torch.exp(-alpha * cost(x))

    def get_optima():
        optimal_x = torch.ones(1, n)
        optimal_value = cost(optimal_x)
        return optimal_x, optimal_value

    return pdf, cost, get_optima

