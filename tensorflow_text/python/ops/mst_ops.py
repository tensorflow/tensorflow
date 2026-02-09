# coding=utf-8
# Copyright 2025 TF.Text Authors.
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

"""TensorFlow ops for maximum spanning tree problems."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import standard_ops

# pylint: disable=g-bad-import-order
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
gen_mst_ops = load_library.load_op_library(resource_loader.get_path_to_datafile('_mst_ops.so'))

# Re-export the generated MST op.
max_spanning_tree = gen_mst_ops.max_spanning_tree


@ops.RegisterGradient("MaxSpanningTree")
def max_spanning_tree_gradient(mst_op, d_loss_d_max_scores, *_):
  """Returns a subgradient of the MaximumSpanningTree op.

  Note that MaximumSpanningTree is only differentiable w.r.t. its |scores| input
  and its |max_scores| output.

  Args:
    mst_op: The MaximumSpanningTree op being differentiated.
    d_loss_d_max_scores: [B] vector where entry b is the gradient of the network
      loss w.r.t. entry b of the |max_scores| output of the |mst_op|.
    *_: The gradients w.r.t. the other outputs; ignored.

  Returns:
    1. None, since the op is not differentiable w.r.t. its |num_nodes| input.
    2. [B,M,M] tensor where entry b,t,s is a subgradient of the network loss
       w.r.t. entry b,t,s of the |scores| input, with the same dtype as
       |d_loss_d_max_scores|.
  """
  dtype = d_loss_d_max_scores.dtype.base_dtype
  if dtype is None:
    raise errors.InvalidArgumentError("Expected (%s) is not None" % dtype)

  argmax_sources_bxm = mst_op.outputs[1]
  input_dim = array_ops.shape(argmax_sources_bxm)[1]  # M in the docstring

  # The one-hot argmax is a subgradient of max.  Convert the batch of maximal
  # spanning trees into 0/1 indicators, then scale them by the relevant output
  # gradients from |d_loss_d_max_scores|.  Note that |d_loss_d_max_scores| must
  # be reshaped in order for it to broadcast across the batch dimension.
  indicators_bxmxm = standard_ops.one_hot(
      argmax_sources_bxm, input_dim, dtype=dtype)
  d_loss_d_max_scores_bx1 = array_ops.expand_dims(d_loss_d_max_scores, -1)
  d_loss_d_max_scores_bx1x1 = array_ops.expand_dims(d_loss_d_max_scores_bx1, -1)
  d_loss_d_scores_bxmxm = indicators_bxmxm * d_loss_d_max_scores_bx1x1
  return None, d_loss_d_scores_bxmxm
