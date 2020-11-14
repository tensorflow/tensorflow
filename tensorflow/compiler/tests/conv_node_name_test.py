# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Convolution node name match via the XLA JIT.

The canned results in these tests are created by running each test using the
Tensorflow CPU device and saving the output.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import ops
from tensorflow.python.layers import layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.platform import googletest


class ConvolutionNodeNameTest(xla_test.XLATestCase):
  """Verify convolution node name match.

  Verify convolution node names on TPU and CPU match with dilation > 1.
  """

  def _verifyNodeNameMatch(self, layer, input_sizes, filter_sizes, strides,
                           dilations):

    def _GetNodeNames(use_xla):
      with self.session():
        input_tensor = array_ops.placeholder(np.float32, shape=input_sizes)

        if use_xla:
          with self.test_scope():
            # pylint: disable=protected-access
            graph = ops.get_default_graph()
            graph._set_control_flow_context(
                control_flow_ops.XLAControlFlowContext())
            # pylint: enable=protected-access
            conv2d_op = layer(
                filters=64,
                kernel_size=filter_sizes,
                dilation_rate=dilations,
                padding="same")
            _ = conv2d_op(input_tensor)
            return [n.name for n in ops.get_default_graph().as_graph_def().node]
        else:
          with ops.device("CPU"):
            conv2d_op = layer(
                filters=64,
                kernel_size=filter_sizes,
                dilation_rate=dilations,
                padding="same")
            _ = conv2d_op(input_tensor)
            names = [
                n.name for n in ops.get_default_graph().as_graph_def().node
            ]
            # filter out space to depth ops.
            return [
                name for name in names
                if "space" not in name and "Space" not in name
            ]

    xla_names = _GetNodeNames(use_xla=True)
    no_xla_names = _GetNodeNames(use_xla=False)

    # CPU path creates some additional nodes to handle dilations.
    # TODO(b/138804006): Remove this when CPU & GPU support dilations.
    filtered_no_xla_names = []
    for name in no_xla_names:
      if ("dilation_rate" in name or "filter_shape" in name or "stack" in name):
        continue
      else:
        filtered_no_xla_names.append(name)

    self.assertListEqual(xla_names, filtered_no_xla_names)

  def testConv1DNodeNameMatch(self):
    input_sizes = [8, 16, 3]
    filter_sizes = [7]
    strides = 1
    dilations = [2]
    layer = layers.Conv1D
    self._verifyNodeNameMatch(layer, input_sizes, filter_sizes, strides,
                              dilations)

  def testConv2DNodeNameMatch(self):
    input_sizes = [8, 16, 16, 3]
    filter_sizes = [7, 7]
    strides = 1
    dilations = [2, 2]
    layer = layers.Conv2D
    self._verifyNodeNameMatch(layer, input_sizes, filter_sizes, strides,
                              dilations)

  def testConv3DNodeNameMatch(self):
    input_sizes = [8, 16, 16, 16, 3]
    filter_sizes = [7, 7, 7]
    strides = 1
    dilations = [2, 2, 2]
    layer = layers.Conv3D
    self._verifyNodeNameMatch(layer, input_sizes, filter_sizes, strides,
                              dilations)


if __name__ == "__main__":
  googletest.main()
