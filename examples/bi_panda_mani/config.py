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

TTTS_PATH = absjoin(dirname(__file__), '../..')


PANDA_URDF = absjoin(TTTS_PATH, 'models/franka_panda/panda.urdf')
PANDA_ARM_URDF = absjoin(TTTS_PATH, 'models/franka_description/robots/panda_arm.urdf')
PANDA_XML = absjoin(TTTS_PATH, 'models/franka_panda/panda.xml')
BIMANNUAL_XML = absjoin(TTTS_PATH, 'models/franka_panda/bimannual.xml')
BIMANNUAL_XML_REAL = absjoin(TTTS_PATH, 'models/franka_panda/bimannual_real.xml')
TABLE_URDF= absjoin(TTTS_PATH, 'models/table/table.urdf')
SHELF_URDF = absjoin(TTTS_PATH, 'models/shelf/shelf.urdf')
BASE_URDF = absjoin(TTTS_PATH, 'models/cube_base/cube_base.urdf')
BOX_URDF = absjoin(TTTS_PATH, 'models/object/cube.urdf')
REALBOX_URDF = absjoin(TTTS_PATH, 'models/object/cube_real.urdf')
SMALLBOX_URDF = absjoin(TTTS_PATH, 'models/object/smallcube.urdf')
CYLINDER_URDF = absjoin(TTTS_PATH, 'models/object/cylinder.urdf')
CYLINDER_URDF_REAL = absjoin(TTTS_PATH, 'models/object/cylinder_real.urdf')

# sys.path.append(PROJECT_DIR)
sys.path.append(TTTS_PATH)