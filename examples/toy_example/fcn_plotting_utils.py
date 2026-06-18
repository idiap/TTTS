#
# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Teng Xue <teng.xue@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#


import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LinearLocator
import warnings
from matplotlib.colors import LogNorm
from matplotlib import ticker, cm
warnings.filterwarnings("ignore")

def plot_surf(x,y,cost,data=None,zlim=(0,1000),figsize=10, view_angle=(45,45),markersize=3):
   
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    fig.set_size_inches(figsize,figsize)

    Z = np.empty((len(x),len(y)))

    X, Y = np.meshgrid(x, y)
    XY = np.array([X.reshape(-1,),Y.reshape(-1,)]).T
    Z = cost(XY).reshape(X.shape[0],X.shape[1])

    cmap = sns.cm.rocket_r
    surf = ax.plot_surface(X, Y, Z,cmap=cmap,
                           linewidth=0, antialiased=False, zorder=0, alpha=1)
    # Customize the z axis.
    ax.set_zlim(zlim[0], zlim[1])
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter('{x:.1f}')

    if not (data is None):
        data_z = 0
        if len(data.shape)==3:
            data_z = data[:,2] 
        ax.plot(data[:,0],data[:,1],data_z,'ob', markersize=markersize, zorder=10)
        
    ax.view_init(view_angle[0], view_angle[1])
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.4, aspect=5)
    
    return plt

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm

# def plot_surface_3D(x, y, cost, data=None, elev=45, azim=135, figsize=(8, 6), markersize=3, log_norm=True):
#     X, Y = np.meshgrid(x, y)
#     XY = np.stack([X.ravel(), Y.ravel()], axis=1)
#     Z = cost(XY).reshape(X.shape)

#     fig = plt.figure(figsize=figsize)
#     ax = fig.add_subplot(111, projection='3d')

#     # Log normalization if requested
#     if log_norm:
#         Z_plot = np.log10(Z + 1e-12)
#     else:
#         Z_plot = Z

#     # Surface plot
#     surf = ax.plot_surface(X, Y, Z_plot, cmap=cm.viridis, edgecolor='k', linewidth=0.1, alpha=0.9)
#     fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

#     if data is not None:
#         ax.scatter(data[:, 0], data[:, 1], np.zeros_like(data[:, 0]), color='blue', s=markersize**2)

#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('cost')
#     ax.view_init(elev=elev, azim=azim)
#     plt.tight_layout()
#     plt.show()

#     return plt
import numpy as np
import plotly.graph_objects as go


def plot_surface_3D_interactive(x, y, cost, data1=None, data2=None, data3=None, width=1000, height=800):

    X, Y = np.meshgrid(x, y)
    XY = np.stack([X.ravel(), Y.ravel()], axis=1)
    Z = cost(XY).reshape(X.shape)
    print("min Z:", Z.min(), "max Z:", Z.max())

    fig = go.Figure()
    
    fontsize = 35
    ticksize = 17

    # Surface plot
    fig.add_trace(go.Surface(
        z=Z, x=X, y=Y,
        colorscale='magma_r', #'Viridis',
        opacity=0.7,
        showscale=False,
        name='Cost surface'
    ))

    # Optional scatter points
    if data3 is not None:
        est_x = data3[:, 0]
        est_y = data3[:, 1]
        est_z = cost(data3)
        fig.add_trace(go.Scatter3d(
            x=est_x, y=est_y, z=est_z,
            mode='markers',
            marker=dict(size=6, color='red', symbol='x', line=dict(width=1, color='black')),
            name='Ground Truth',
            showlegend=True
        ))

    if data1 is not None:
        est_x = data1[:, 0]
        est_y = data1[:, 1]
        est_z = cost(data1)
        fig.add_trace(go.Scatter3d(
            x=est_x, y=est_y, z=est_z,
            mode='markers',
            marker=dict(size=8, color='cyan', symbol='diamond', line=dict(width=1, color='black')),
            name='TTGO result',
            showlegend=True
        ))

    if data2 is not None:
        est_x = data2[:, 0]
        est_y = data2[:, 1]
        est_z = cost(data2)
        fig.add_trace(go.Scatter3d(
            x=est_x, y=est_y, z=est_z,
            mode='markers',
            marker=dict(size=12, color='green', symbol='circle', line=dict(width=1, color='black')),
            name='TTTS result',
            showlegend=True
        ))

    # Layout
    fig.update_layout(
        title="Interactive 3D Surface with Estimates",
        # scene=dict(
        #     xaxis=dict(
        #         title=dict(text='x₁', font=dict(size=fontsize, family='Arial', color='black')),
        #         tickfont=dict(size=ticksize, family='Arial', color='black')
        #     ),
        #     yaxis=dict(
        #         title=dict(text='x₂', font=dict(size=fontsize, family='Arial', color='black')),
        #         tickfont=dict(size=ticksize, family='Arial', color='black'),
        #     ),
        #     zaxis=dict(
        #         title=dict(text='cost   ', font=dict(size=fontsize, family='Arial', color='black')),
        #         tickfont=dict(size=ticksize, family='Arial', color='black')
        #     ),
        #     aspectmode='manual',
        #     aspectratio=dict(x=1, y=1, z=1),
        #     camera=dict(
        #         eye=dict(x=-1.25, y=1.25, z=1.25)  # key to isometric-like view
        #     )
        # ),
        scene=dict(
            xaxis=dict(
                visible=False,
                showbackground=False,
                showticklabels=False,
                title=None
            ),
            yaxis=dict(
                visible=False,
                showbackground=False,
                showticklabels=False,
                title=None
            ),
            zaxis=dict(
                visible=False,
                showbackground=False,
                showticklabels=False,
                title=None
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(
                eye=dict(x=-1.25, y=1.25, z=1.25)
            )
        ),

        legend=dict(
            font=dict(size=20, family='Arial', color='black'),
            x=0.80,
            y=0.98,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.7)',
            borderwidth=1
        ),
        width=width,
        height=height,
        margin=dict(l=10, r=10, b=10, t=40)
    )


    fig.show()
    return fig


def plot_surface_3D_interactive_2(x, y, cost, data1=None, data2=None, data3=None, width=1000, height=800):
    X, Y = np.meshgrid(x, y)
    XY = np.stack([X.ravel(), Y.ravel()], axis=1)
    Z = cost(XY).reshape(X.shape)
    print("min Z:", Z.min(), "max Z:", Z.max())
    fig = go.Figure()

    fontsize = 35
    ticksize = 17

    # Surface plot
    fig.add_trace(go.Surface(
        z=Z, x=X, y=Y,
        colorscale='magma_r', #'Viridis',
        opacity=0.7,
        showscale=False,
        name='Cost surface'
    ))

    # Plot curves at integer x0 = 0 to 10
    for x0_fixed in range(10):
        x1_vals = Y[:, 0]

        # curve_points = np.column_stack([
        #     np.full_like(x1_vals, x0_fixed),
        #     x1_vals
        # ])
        # z_vals = cost(curve_points)
        fig.add_trace(go.Scatter3d(
            x=X[:, x0_fixed],
            y=x1_vals,
            z=Z[:, x0_fixed],
            mode='lines',
            line=dict(width=8, color='black'),
            name=f'x₀={x0_fixed}',
            showlegend=False
        ))
    for x0_fixed in range(10):
        x1_vals = Y[:, 0]

        # curve_points = np.column_stack([
        #     np.full_like(x1_vals, x0_fixed),
        #     x1_vals
        # ])
        # z_vals = cost(curve_points)
        fig.add_trace(go.Scatter3d(
            x=X[:, x0_fixed],
            y=x1_vals,
            z=Z[:, x0_fixed]*0,
            mode='lines',
            line=dict(width=8, color='blue', dash='dot'),
            name=f'x₀={x0_fixed}',
            showlegend=False
        ))


    # Optional scatter points
    # Optional scatter points
    if data3 is not None:
        est_x = data3[:, 0]
        est_y = data3[:, 1]
        est_z = cost(data3)
        fig.add_trace(go.Scatter3d(
            x=est_x, y=est_y, z=est_z,
            mode='markers',
            marker=dict(size=6, color='red', symbol='x', line=dict(width=1, color='black')),
            name='Ground Truth',
            showlegend=True
        ))

    if data1 is not None:
        est_x = data1[:, 0]
        est_y = data1[:, 1]
        est_z = cost(data1)
        fig.add_trace(go.Scatter3d(
            x=est_x, y=est_y, z=est_z,
            mode='markers',
            marker=dict(size=8, color='cyan', symbol='diamond', line=dict(width=1, color='black')),
            name='TTGO result',
            showlegend=True
        ))

    if data2 is not None:
        est_x = data2[:, 0]
        est_y = data2[:, 1]
        est_z = cost(data2)
        fig.add_trace(go.Scatter3d(
            x=est_x, y=est_y, z=est_z,
            mode='markers',
            marker=dict(size=12, color='green', symbol='circle', line=dict(width=1, color='black')),
            name='TTTS result',
            showlegend=True
        ))

    # Layout
    fig.update_layout(
        title="Interactive 3D Surface with Estimates",
        # scene=dict(
        #     xaxis=dict(
        #         title=dict(text='x   ', font=dict(size=fontsize, family='Arial', color='black')),
        #         tickfont=dict(size=ticksize, family='Arial', color='black')
        #     ),
        #     yaxis=dict(
        #         title=dict(text='y   ', font=dict(size=fontsize, family='Arial', color='black')),
        #         tickfont=dict(size=ticksize, family='Arial', color='black')
        #     ),
        #     zaxis=dict(
        #         title=dict(text='cost   ', font=dict(size=fontsize, family='Arial', color='black')),
        #         tickfont=dict(size=ticksize, family='Arial', color='black'),
        #     )
        # ),
        scene=dict(
            xaxis=dict(
                visible=False,
                showbackground=False,
                showticklabels=False,
                title=None
            ),
            yaxis=dict(
                visible=False,
                showbackground=False,
                showticklabels=False,
                title=None
            ),
            zaxis=dict(
                visible=False,
                showbackground=False,
                showticklabels=False,
                title=None
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(
                eye=dict(x=-1.25, y=1.25, z=1.25)
            )
        ),

        # legend=dict(
        #     font=dict(size=fontsize, family='Arial', color='black')
        # ),
        legend=dict(
            font=dict(size=20, family='Arial', color='black'),
            x=0.80,  # Position the legend inside the 3D plot
            y=0.98,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.7)',  # Add a semi-transparent background
            # bordercolor='black',
            borderwidth=1
        ),
        width=width,
        height=height,
        margin=dict(l=10, r=10, b=10, t=40),
    )

    fig.show()
    return fig


def plot_surface_2D(x, y, cost, data1=None, data2=None, data3=None, name='None', figsize=(10, 10), task='mix'):
    X, Y = np.meshgrid(x, y)
    XY = np.stack([X.ravel(), Y.ravel()], axis=1)
    Z = cost(XY).reshape(X.shape)
    
    print("Z range:", Z.min(), Z.max())

    fig, ax = plt.subplots(figsize=figsize)

    ticksize = 25
    markersize = 350


    contour = ax.contourf(X, Y, Z, levels=50, cmap='magma_r')
    # plt.colorbar(contour, ax=ax, label='Cost')

    if task=='mix':
        for x0_fixed in range(10):
            x0_vals = np.full_like(y, x0_fixed)
            x1_vals = y
            curve_points = np.column_stack([x0_vals, x1_vals])
            z_vals = cost(curve_points)
            ax.plot(x0_vals, x1_vals, color='black', linewidth=2)


            ax.plot(x0_vals, x1_vals, linestyle='-', color='blue', linewidth=2)



    if data1 is not None:
        ax.scatter(data1[:, 0], data1[:, 1], c='cyan', s=markersize, label='TTGO result', marker='D')

    if data2 is not None:
        ax.scatter(data2[:, 0], data2[:, 1], c='green', s=markersize, label='TTTS result')

    if data3 is not None:
        ax.scatter(data3[:, 0], data3[:, 1], c='red', s=markersize, label='Ground Truth', marker='x')


    ax.set_title(name, fontsize=ticksize)
    ax.set_xlabel('x₁', fontsize=ticksize)
    ax.set_ylabel('x₂', fontsize=ticksize)
    ax.legend(fontsize=ticksize)
    ax.grid(True)
    x_range = x.max() - x.min()
    ax.set_xlim(x.min() - 0.05 * x_range, x.max() + 0.05 * x_range)
    y_range = y.max() - y.min()
    ax.set_ylim(y.min() - 0.05 * y_range, y.max() + 0.05 * y_range)
    ax.tick_params(axis='both', labelsize=ticksize)
    

    plt.tight_layout()
    plt.show()
    return fig



def plot_contour(x,y,cost, data=None, contour_scale=100, figsize=10, markersize=3,log_norm=True):
    # plt.style.use('seaborn-white')
    Z = np.empty((len(x),len(y)))
    X, Y = np.meshgrid(x, y)
    XY = np.array([X.reshape(-1,),Y.reshape(-1,)]).T
    Z = cost(XY).reshape(X.shape[0],X.shape[1])
    sns.set_style("white")
    cmap = 'binary_r'    
    if log_norm == True:
        levels = 10**(0.25*np.arange(-6,14))
        cs = plt.contour(X, Y, Z, contour_scale, cmap=cmap, shade=True,locator=ticker.LogLocator(),
            levels=levels, norm=LogNorm(), alpha=1);
    else:
        cs = plt.contour(X, Y, Z, contour_scale, cmap=cmap, shade=True,alpha=1);

    plt.colorbar(cs);
    if not (data is None):
        plt.plot(data[:,0],data[:,1],'ob', markersize=markersize)
    plt.rcParams["figure.figsize"] = (figsize, figsize)

    return plt
