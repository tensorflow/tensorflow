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
"""
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.compat import compat as fwd_compat
from tensorflow.python.eager import context
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_sparse_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_math_ops import *
# pylint: enable=wildcard-import
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
"""
import tensorflow
import tensorflow.python
from tensorflow.python import ops

from tensorflow.contrib.t2t.python.ops import gen_t2t_ops
from tensorflow.contrib.util import loader
from tensorflow.python.platform import resource_loader

_t2t_ops_so = loader.load_op_library(resource_loader.get_path_to_datafile("_t2t_ops.so"))

@ops.RegisterGradient("CustomL2Norm")
def _custom_l2_norm_grad(op, grad):
  return [gen_t2t_ops.custom_l2_norm_grad(op.inputs[0], op.inputs[1], op.inputs[2], op.inputs[3], grad), None, None, None]
  #return [custom_l2_norm_grad(op.inputs[0], op.inputs[1], op.inputs[2], op.inputs[3], grad), 0.0, 0.0, 0.0]
