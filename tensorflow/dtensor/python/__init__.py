# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""DTensor Python API."""

from tensorflow.dtensor.python import gen_dtensor_ops as ops
from tensorflow.dtensor.python import mesh_util
from tensorflow.dtensor.python import tpu_util

from tensorflow.dtensor.python.api import call_with_layout
from tensorflow.dtensor.python.api import check_layout
from tensorflow.dtensor.python.api import copy_to_mesh
from tensorflow.dtensor.python.api import device_name
from tensorflow.dtensor.python.api import fetch_layout
from tensorflow.dtensor.python.api import local_devices
from tensorflow.dtensor.python.api import num_global_devices
from tensorflow.dtensor.python.api import num_local_devices
from tensorflow.dtensor.python.api import pack
from tensorflow.dtensor.python.api import relayout
from tensorflow.dtensor.python.api import run_on
from tensorflow.dtensor.python.api import unpack
from tensorflow.dtensor.python.config import client_id
from tensorflow.dtensor.python.config import full_job_name
from tensorflow.dtensor.python.config import heartbeat_enabled
from tensorflow.dtensor.python.config import job_name
from tensorflow.dtensor.python.config import jobs
from tensorflow.dtensor.python.config import num_clients
from tensorflow.dtensor.python.d_checkpoint import DTensorCheckpoint
from tensorflow.dtensor.python.d_variable import DVariable
from tensorflow.dtensor.python.input_util import DTensorDataset
from tensorflow.dtensor.python.input_util import TFDataServiceConfig
from tensorflow.dtensor.python.layout import Layout
from tensorflow.dtensor.python.layout import MATCH
from tensorflow.dtensor.python.layout import Mesh
from tensorflow.dtensor.python.layout import UNSHARDED
from tensorflow.dtensor.python.save_restore import enable_save_as_bf16
from tensorflow.dtensor.python.save_restore import name_based_restore
from tensorflow.dtensor.python.save_restore import name_based_save
from tensorflow.dtensor.python.save_restore import sharded_save
