#
# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Teng Xue <teng.xue@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import torch
import numpy as np
import sys

cur_path = sys.path[0]
sys.path.append(cur_path + "/..")
import tt_utils
from src.ttts import TTTS

from fcn_plotting_utils import plot_surf, plot_contour, plot_surface_3D_interactive

np.set_printoptions(precision=3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device
##### Define the objective function (a pdf, need not be normalized)


from test_fcns import diag_func_2D
L=5
n = 2 # Dimension of the domain
a=11; b=7;
pdf, cost = diag_func_2D(alpha=1.0)
print("This has multiple global optima")
z_max = 2000 # for plotting
log_norm = True # for plotting
d = 1000

domain = [torch.linspace(-L,L,d).to(device)]*2
ttts = TTTS(func = pdf, domain = domain, value_k=5, visit_k=5, max_tt=2, num_mcts_batch=5,
            warm_start=True, verbose=False, num_mcts_sample=1, cross_max_iter=2)
sol = ttts.iterate(max_iters = 500, param_C=3) # K x n
