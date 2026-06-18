# Monte Carlo Tree Search with Tensor Factorization

This repository provides the official implementation of **Tensor Train Tree Search (TTTS)**, a novel algorithm that integrates Monte Carlo Tree Search (MCTS) with Tensor Train (TT) factorization. TTTS efficiently handles complex decision-making problems in robotics by combining global search, local refinement, and a low-rank representation of the decision tree.

TTTS enables scalable and parallelizable planning across a variety of tasks, including nonlinear dynamics, hybrid discrete-continuous optimization, non-convex constraints, and multi-modal solution discovery.

---

## 🔬 Features

- Combines MCTS with tensor factorization to reduce combinatorial complexity.
- Supports non-linear kinematics, non-convex constraints, mixed-integer programming and complex contact-rich dynamics.
-  seamless integration with black-box simulators (e.g., Genesis).
- Supports GPU-based parallel computation.
---

## 📁 Experiments

In the ```examples``` folder, you can find the following examples:

### 1. Toy Examples

- Continuous Non-Convex Optimization  
- Mixed-Integer Optimization

If you are interested in a comparison between Tensor Trains (TTs) and neural networks (NNs) as function approximators, check out tt_vs_nn.ipynb.

### 2. Inverse Kinematics

- Solves collision-free IK for multi-joint manipulators  

- Demonstrates TTTS’s ability to find multiple solutions  

### 3. Motion Planning Around Obstacles

- 3-joint Manipulator: Reaches a narrow target via long-horizon planning  
- 7-joint Panda Arm: Moves between shelves while avoiding collisions  

### 4. Bimanual Whole-Body Manipulation

- Real-world task using whole-body contact  
- Solves complex control with real-time MPC using TTTS  

---
## 🚀 Getting Started

```bash
pip install -r requirements.txt
``` 

## 📬 Contact
For questions or contributions, please contact:

Teng Xue
[teng.xue@idiap.ch]
