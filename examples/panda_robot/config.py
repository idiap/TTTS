#
# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Teng Xue <teng.xue@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import sys
from os.path import join, abspath, dirname

def absjoin(*args):
    return abspath(join(*args))

import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = absjoin(dirname(__file__), '../../')
TTTS_PATH = absjoin(PROJECT_DIR, 'examples')
BENCHMARK_PATH = absjoin(PROJECT_DIR, 'examples', 'panda_robot')


PANDA_XML = absjoin(TTTS_PATH, 'panda_robot/models/franka_panda/panda.xml')
SHELF_URDF = absjoin(TTTS_PATH, 'panda_robot/models/shelf/shelf.urdf')
BASE_URDF = absjoin(TTTS_PATH, 'panda_robot/models/cube_base/cube_base.urdf')

sys.path.append(PROJECT_DIR)
sys.path.append(TTTS_PATH)