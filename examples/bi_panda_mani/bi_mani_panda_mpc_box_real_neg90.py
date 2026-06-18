#
# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Teng Xue <teng.xue@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import torch
import numpy as np
import genesis as gs

import sys, os
import time

cur_path = os.path.dirname(__file__)
sys.path.append(cur_path)

parent_path = os.path.dirname(cur_path)
sys.path.append(parent_path)

pparent_path = os.path.dirname(parent_path)
sys.path.append(pparent_path)


from bi_panda_mani.config import absjoin, BIMANNUAL_XML_REAL, REALBOX_URDF
from bi_panda_mani.env_panda_pivot import PandaPivot

from src.ttts_augment import TTTS
from robot_utils import Point2PointMotion

BOX_SIZE = 0.4
MAX_BATCH = 500

def env_infos_gen():
    """
    This function generates the environment information 
    useful for defining the simulation environment.
    """
    robot_info = {
            "name": "bi_panda",
            "xml": BIMANNUAL_XML_REAL,
            "base_pose": [0, 0, 0.0, 0, 0, 0],
            # "conf": [np.pi/8, np.pi/8, np.pi/2.1, np.pi/2.1, -np.pi/2, np.pi/2, 
            #         -1.0, -1.0, 0, np.pi, np.pi, np.pi, np.pi/4, np.pi/4, 
            #         # 0.06, 0.06, 0.06, 0.06
            #         ],
            "conf": [0.0, 1.6, 1.7163346012349714,  1.6819582263712298, 
                1.6098622649351406,  -1.593042766403684, -1.3949898984220739, -1.4514206011086177, 
                -0.23821908311049236, 0.06636745597918826, 2.432812159862783, 2.3819313348035016, 
                -2.274967811495568, 0.7672042183099919], #up
                # -0.704171484700671, -0.8035921084849047], #side
            

            "jnt_names": ["panda0_joint1", "panda1_joint1", "panda0_joint2", "panda1_joint2",
                "panda0_joint3", "panda1_joint3", "panda0_joint4", "panda1_joint4",
                "panda0_joint5", "panda1_joint5", "panda0_joint6",  "panda1_joint6",
                "panda0_joint7", "panda1_joint7", 
                # "panda0_finger_joint1", "panda0_finger_joint2", 
                # "panda1_finger_joint1", "panda1_finger_joint2", 
                ],
            "active_jnts": ["panda0_joint1", "panda1_joint1",
                "panda0_joint4", "panda1_joint4", 
                "panda0_joint6",  "panda1_joint6",
            ],


            "kp": np.array([4500, 4500, 4500, 4500, 3500, 3500, 3500, 3500, 
                            2000, 2000, 2000, 2000, 2000, 2000,
                            # 100, 100, 100, 100
                            ]),

            "kv": np.array([450, 450, 450, 450, 350, 350, 350, 350,
                            200, 200, 200, 200, 200, 200, 
                            # 10, 10, 10, 10
                            ]),

            "force_range": {
                "lower": np.array([-87, -87, -87, -87, -87, -87, -87, -87, 
                                   -12, -12, -12, -12, -12, -12,
                                #    -100, -100, -100, -100
                                   ]),
                "upper": np.array([87,  87,  87, 87,  87,  87,  87, 87,
                                   12,  12, 12,  12,  12, 12, 
                                #    100,  100, 100,  100
                                   ]),
            },
    }
    
    init__angle = 45

    goal_info = {
        "init": init__angle,
        "target_orn": [0, 0, -45],
        "target_pos": [0.80, -0.20, 0.32],
    }

    # box_info = {
    #     "name": "box",
    #     "urdf": BOX_URDF,
    #     "size": [BOX_SIZE, BOX_SIZE, BOX_SIZE],
    #     "base_pose": [0.9, 0.1, BOX_SIZE/2+0.05, 0, 0, init__angle],
    # }

    box_info = {
        "name": "box",
        "urdf": REALBOX_URDF,
        "size": [BOX_SIZE, BOX_SIZE, BOX_SIZE],
        "base_pose": [0.70, 0.20, 0.32, 0, 0, init__angle],
    }

    env_infos = {
        "robot": robot_info,
        "box": box_info,
        "goal": goal_info,
    }

    return env_infos

def cost_func(box_poss, box_orns, theta_t, box_contacts):
    box_target_orn = env_infos["goal"]["target_orn"]
    target_pos = env_infos["goal"]["target_pos"]
    orn_final_cost = torch.linalg.norm(box_orns[:, -1]/180-torch.tensor(box_target_orn).to(device)/180, dim=-1).to(device)
    pos_final_cost = torch.linalg.norm(box_poss[:, -1] - torch.tensor(target_pos).to(device), dim=-1).to(device)
    # x_final = box_poss[:, -1, 0]


    if not isinstance(theta_t, torch.Tensor):
        theta_t = torch.tensor(theta_t, device=device, dtype=torch.float32)

    """cost contact"""
    lr_contacts = box_contacts[:, :, 0] * box_contacts[:, :, 1] #n_envs x stime_steps x 1
    binary_tr_contacts = (lr_contacts > 0).squeeze() #n_envs x stime_steps
    d_contact = torch.sum(binary_tr_contacts, dim=-1).to(device) #n_envs x 1

    control_cost = torch.linalg.norm(theta_t[:, 1:, :] - theta_t[:, :-1, :], dim=-1).sum(dim=-1).to(device)
    eef0 = env_infos["robot"]["id"].get_link("panda0_grasping_target_hand")
    eef1 = env_infos["robot"]["id"].get_link("panda1_grasping_target_hand")
    eef_dis = torch.linalg.norm(eef0.get_pos() - eef1.get_pos(), dim=-1).to(device)

    cost_total = 1*pos_final_cost + 50*orn_final_cost + 0.1* control_cost - 1*d_contact + 5*(eef_dis<0.3).to(torch.float) #+ 0.1*orn_cost
    # cost_total = 10*(x_final<0.7)*abs(x_final - 0.7).to(torch.float) + 1*orn_final_cost + 0.1* control_cost - 1*d_contact + 1*(eef_dis<0.3).to(torch.float) #+ 0.1*orn_cost

    return cost_total


evaluate_time = 0.0

def augment_objective_func(xs):
    global env_infos, evaluate_time

    xs_pre = xs.clone()
    if xs.shape[0] < MAX_BATCH:
        xs = torch.cat([xs, torch.zeros((MAX_BATCH - xs.shape[0], xs.shape[1]), device=xs.device)], dim=0)

    # only joint 1 4 6 are active
    idx = [0, 1, 6, 7, 10, 11]
    init_conf = env_infos["robot"]["conf"]
    obj_pos = xs[:, :3]
    x = xs[:, 3:] # weights

    # box_pos =  [obj_pos[:, 0], obj_pos[:, 1], BOX_SIZE/2+0.05, 0, 0, obj_pos[:, 2]]

    box_pos = torch.tile(torch.tensor(env_infos["box"]["base_pose"], dtype=torch.double, device=device),
                           (xs.shape[0], 1))
    id = [0, 1, -1]
    box_pos[:, id] = obj_pos
    

    env.box.set_pos(box_pos[:, :3])
    env.box.set_quat(gs.xyz_to_quat(box_pos[:, 3:]))

    
    # env_infos["box"]["base_pose"] = [obj_pos[:, 0], obj_pos[:, 1], BOX_SIZE/2+0.05, 0, 0, obj_pos[:, 2]],

    # env.load_world(env_infos)

    batch_init_conf = torch.tile(torch.tensor(init_conf, dtype=torch.double,device=device),
                         (x.shape[0], 1))
    theta_0 = batch_init_conf[:, idx]

    # generate trajectories given the wights
    theta_t = env.get_traj_p2(x, theta_0).to(device)
   
    test_traj = torch.tile(torch.tensor(init_conf, dtype=torch.double, device=device),
                           (theta_t.shape[0], theta_t.shape[1], 1))
    test_traj[:, :, idx] = theta_t

        # propage the joint configurations
    _t0 = time.time()
    box_contacts, box_orns, box_poss = env.forward_configuration_full(env_infos, test_traj) #n_env x stime_steps x 2,
    evaluate_time += time.time() - _t0


    d_total = cost_func(box_poss, box_orns, theta_t, box_contacts)

    # # minimize the orn errors and maximize robot-box contact
    # d_total = 0.01*orn_final_cost + 100*pos_final_cost + 0.1* control_cost # + orn_cost

    losses = torch.exp(-d_total)

    if xs_pre.shape[0] < MAX_BATCH:
        losses = losses[:xs_pre.shape[0]]

    return losses

def test_contact():
    env = PandaPivot(n_envs=1, gpu=True, use_gui=True, p2p_motion=p2p_motion)
    env.load_world(env_infos)
    env.reset(env_infos)

    time.sleep(100)

    # for i in range(1000):
    #     comd = torch.tensor(conf_tarj[:, i], device=device)
    #     env.step(comd.unsqueeze(0))
        
    print("done")



if __name__ == "__main__":
    # optimize = True # retrain or only visualize the optimal trajectory
    optimize = False # retrain or only visualize the optimal trajectory

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Define CMA-ES parameters
    num_guess = 5
    popsize = 25
    num_iterations = 2
    # n_envs = num_guess * popsize
    n_envs = MAX_BATCH
    K = 2

    # Define p2p motion

    T = 0.1
    dt = 0.01
    n_joints = 6
    basis = 'bs'
    bounds = None
    p2p_motion = Point2PointMotion(T=T, dt=dt, basis=basis, n=n_joints,K=2,
                                bounds=bounds, device=device)

    # Generate the environment information
    env_infos = env_infos_gen()
    
    """Test the contact dynamics"""
    # test_contact()

    """Optimize the trajectory"""
    if optimize:
        env = PandaPivot(n_envs=n_envs, gpu=True, use_gui=False,
                        p2p_motion=p2p_motion, n_joints=n_joints)
        env.load_world(env_infos)


        # initial guess of target robot configuration
        idx = [0, 1, 6, 7, 10, 11]
        init1 = np.asarray([np.pi/8, -np.pi/4, np.pi/2.1, np.pi/2.1, -np.pi/2, np.pi/2, 
                    -1.0, -1.0, 0, np.pi, np.pi, np.pi, np.pi/4, np.pi/4,])
        initial_sol = torch.tile(torch.tensor(init1[idx], device=device),
                                (num_guess, 1))
        initial_w = torch.rand((num_guess, K*n_joints), device=device)
        initial_guesses = torch.cat((initial_sol, initial_w), dim=-1).to(device)

        # parallel CMAES optimization
        # TODO domain is not specified correctly here
        max_theta = torch.pi
        min_theta = -1*max_theta
        d0_theta = 20; 
        pos_max = [1.1, 0.4, 180]
        pos_min = [0.5, -0.4, -180]

        init_conf = env_infos["robot"]["conf"]
        theta_delta = 20



        domain_pos = [torch.linspace(pos_min[i], pos_max[i], d0_theta).to(device) for i in range(3)] 

        # min_theta = list(np.array(init_conf)[idx] - theta_delta)
        # max_theta = list(np.array(init_conf)[idx] + theta_delta)
        # domain_theta = [torch.linspace(min_theta[i], max_theta[i], d0_theta).to(device) for i in range(len(min_theta))]
        # domain_w_theta = domain_theta*K
        domain_theta = [torch.linspace(min_theta, max_theta, d0_theta).to(device)]*n_joints
        domain_w_theta = [torch.linspace(min_theta, max_theta, d0_theta).to(device)]*(K*n_joints)
        domain = domain_pos + domain_w_theta


        model_folder = cur_path + "/tt_models/"
        file_name = 'bi_manipulation_box_real'
        # file_name = 'bi_panda_pivot'
        save_path = os.path.join(model_folder, file_name)
        evaluate_time = 0.0
        tt_apprx_st = time.time()
        ttts = TTTS(func = augment_objective_func, domain = domain, value_k=1, visit_k=1,
                    cross_max_iter=2, kickrank=20, max_tt=100, max_batch=MAX_BATCH, num_mcts_sample=10,
                warm_start=True, verbose=False, save_path = save_path)#, test_mode=True)

        tt_apprx_wall = time.time() - tt_apprx_st
        tt_apprx_net = tt_apprx_wall - evaluate_time
        print(f"TT Approximation Time: wall={tt_apprx_wall:.2f}s  net={tt_apprx_net:.2f}s")
        
        # MPC
        theta_t_exec = []
        box_pose = []
        for i in range(int(2/T)):
            print(f"iteration {i}")
            #condition on the current state, get the TT model
            ttts.domain = domain
            # full_state, state = env.get_box_state()
            xy = env_infos["box"]["base_pose"][:2]
            pos = env_infos["box"]["base_pose"][-1]

            ttts.task_var = torch.tensor([xy[0], xy[1], pos], device=device).unsqueeze(0)
            print("current object pose", ttts.task_var)

            # env_infos["box"]["base_pose"] = full_state 
            # env_infos["robot"]["conf"] = env.robot.get_qpos()[:, :n_joints]    

            sol = ttts.iterate(max_iters = 3, param_C=3)
            # #cma-es finetune
            # sol = ttts.parallel_cmaes_gpu(initial_guesses=sol_1, sigma=0.1,
            #                     popsize=MAX_BATCH, num_iterations=20)



            idx = [0, 1, 6, 7, 10, 11]
            init_conf = env_infos["robot"]["conf"]
            batch_init_conf = torch.tile(torch.tensor(init_conf, device=device),
                                (sol.shape[0], 1))
            theta_0 = batch_init_conf[:, idx]

            # generate trajectories given the wights
            theta_t = env.get_traj_p2(sol, theta_0).to(device)
        
            test_traj = torch.tile(torch.tensor(init_conf, device=device),
                                (theta_t.shape[0], theta_t.shape[1], 1)).to(torch.double)
            test_traj[:, :, idx] = theta_t

            if test_traj.shape[0] < MAX_BATCH:
                test_traj = torch.cat([test_traj, torch.zeros((MAX_BATCH - test_traj.shape[0], test_traj.shape[1], test_traj.shape[2]), device=device)], dim=0)

            env_infos, box_traj, best_theta_t = env.forward_execution(env_infos, test_traj, cost_func, record=False)


            # aug_theta_t = best_theta_t[:15].repeat((test_traj.shape[0], 1, 1))
            # env_infos, box_traj, best_theta_t = env.forward_execution(env_infos, aug_theta_t, cost_func, record=False)

            
            theta_t_exec.append(best_theta_t.cpu().numpy())
            box_pose.append(box_traj.cpu().numpy())

            if abs(pos-env_infos["goal"]["target_orn"][-1]) < 5:
                break   

        # theta_t_exec = torch.cat(theta_t_exec, dim=0)
        # save the trajectory
        np.save(sys.path[0] +"/data/bi_mani_box_neg90.npy", {"robot_q": theta_t_exec, "box_q": box_pose})
    else:
        """Visualize the optimal trajectory"""
        # load the optimal trajectory and visualize it
        import os
        if os.path.exists(sys.path[0] +"/data/bi_mani_box_neg90.npy"):
            data = np.load(sys.path[0] +"/data/bi_mani_box_neg90.npy", allow_pickle=True)
            opti_traj = data.item()["robot_q"]
            box_traj = data.item()["box_q"]
        else:
            raise ValueError("Optimal solution not found")
        env_infos = env_infos_gen()
        # Define CMA-ES parameters
        env = PandaPivot(n_envs=1, gpu=True, use_gui=True,
                        p2p_motion=p2p_motion, n_joints=n_joints)
        env.load_world(env_infos)
        robot = env_infos["robot"]["id"]

        env.cam.start_recording()
        for i in range(len(opti_traj)):
            print(f"the {i}th stage")
            init_conf = env_infos["robot"]["conf"]
            bounded_traj = torch.tile(torch.tensor(init_conf, dtype=torch.double, device=device),
                        (opti_traj[i].shape[0], 1))
            idx = [0, 1, 6, 7, 10, 11]
            bounded_traj[:, idx] = torch.tensor(opti_traj[i][:, idx], device=device)
            print("robot configuration trajectory", opti_traj[i][:, idx])
            env.set_robot_box_config(bounded_traj, box_traj[i], dt=0.05, record=True)
            # env.reset(env_infos)
            # env_infos, box_orns, best_theta_t = env.forward_execution(env_infos, opti_traj[i][None, :, :], cost_func, record=True)
            # print("final box orns", box_orns[:, -1])
            xy = env_infos["box"]["base_pose"][:2]
            pos = env_infos["box"]["base_pose"][-1]

            # cur_pose = torch.tensor([xy[0], xy[1], pos], device=device).unsqueeze(0)
            print("current object pose", env.get_box_state()[1])
        # qpos = env.robot.get_qpos()
        # env.robot.set_qpos(qpos)

        env.cam.stop_recording(save_to_filename = sys.path[0] +"/video/bi_mani_traj_box_real_neg90.mp4", fps=20)
        
