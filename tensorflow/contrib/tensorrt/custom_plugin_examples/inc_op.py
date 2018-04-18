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
import os

if platform.system() != "Windows":
  from tensorflow.contrib.util import loader
  from tensorflow.python.platform import resource_loader

  _inc_op = loader.load_op_library(
      os.path.join(os.path.dirname(os.path.realpath(__file__)),"_inc_op.so"))
else:
  raise RuntimeError("Windows not supported")
