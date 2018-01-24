# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import platform

if platform.system() != "Windows":
  # pylint: disable=wildcard-import,unused-import,g-import-not-at-top
  from tensorflow.contrib.tensorrt.ops.gen_trt_engine_op import *

  from tensorflow.contrib.util import loader
  from tensorflow.python.platform import resource_loader
  # pylint: enable=wildcard-import,unused-import,g-import-not-at-top

  _trt_engine_op = loader.load_op_library(
      resource_loader.get_path_to_datafile("_trt_engine_op.so"))
else:
  raise RuntimeError("Windows platforms are not supported")


