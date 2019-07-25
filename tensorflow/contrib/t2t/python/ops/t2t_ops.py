# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow
import tensorflow.python
from tensorflow.python import ops

from tensorflow.contrib.t2t.python.ops import gen_t2t_ops
from tensorflow.contrib.util import loader
from tensorflow.python.platform import resource_loader

_t2t_ops_so = loader.load_op_library(resource_loader.get_path_to_datafile("_t2t_ops.so"))

@ops.RegisterGradient("CustomL2Norm")
def _custom_l2_norm_grad(op, grad, name=None):
  return [gen_t2t_ops.custom_l2_norm_grad(op.inputs[0], op.inputs[1], op.inputs[2], op.inputs[3], grad), None, None, None]

@ops.RegisterGradient("CustomDropout")
def _custom_dropout_grad(op, grad, name=None):
  return [gen_t2t_ops.custom_dropout(grad, op.inputs[1], op.inputs[2]), None, None]
