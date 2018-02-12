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
# ==============================================================================
"""Script to test TF-TensorRT integration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
# normally we should do import tensorflow as tf and then
# tf.placeholder, tf.constant, tf.nn.conv2d etc but
# it looks like internal builds don't like it so
# importing every module individually

from tensorflow.contrib import tensorrt as trt
from tensorflow.core.protobuf import config_pb2 as cpb2
from tensorflow.python.client import session as csess
from tensorflow.python.framework import constant_op as cop
from tensorflow.python.framework import dtypes as dtypes
from tensorflow.python.framework import importer as importer
from tensorflow.python.framework import ops as ops
from tensorflow.python.ops import array_ops as aops
from tensorflow.python.ops import nn as nn
from tensorflow.python.ops import nn_ops as nn_ops


def get_simple_graph_def():
  """Create a simple graph and return its graph_def"""
  g = ops.Graph()
  with g.as_default():
    a = aops.placeholder(dtype=dtypes.float32, shape=(None, 24, 24, 2), name="input")
    e = cop.constant(
        [[[[1., 0.5, 4., 6., 0.5, 1.], [1., 0.5, 1., 1., 0.5, 1.]]]],
        name="weights",
        dtype=dtypes.float32)
    conv = nn.conv2d(
        input=a, filter=e, strides=[1, 2, 2, 1], padding="SAME", name="conv")
    b = cop.constant([4., 1.5, 2., 3., 5., 7.], name="bias", dtype=dtypes.float32)
    t = nn.bias_add(conv, b, name="biasAdd")
    relu = nn.relu(t, "relu")
    idty = aops.identity(relu, "ID")
    v = nn_ops.max_pool(
        idty, [1, 2, 2, 1], [1, 2, 2, 1], "VALID", name="max_pool")
    aops.squeeze(v, name="output")
  return g.as_graph_def()


def run_graph(gdef, dumm_inp):
  gpu_options = cpb2.GPUOptions(per_process_gpu_memory_fraction=0.50)
  ops.reset_default_graph()
  g = ops.Graph()
  with g.as_default():
    inp, out = importer.import_graph_def(
        graph_def=gdef, return_elements=["input", "output"])
    inp = inp.outputs[0]
    out = out.outputs[0]
  with csess.Session(
      config=cpb2.ConfigProto(gpu_options=gpu_options), graph=g) as sess:
    val = sess.run(out, {inp: dumm_inp})
  return val


if "__main__" in __name__:
  inp_dims = (100, 24, 24, 2)
  dummy_input = np.random.random_sample(inp_dims)
  gdef = get_simple_graph_def()
  # Get optimized graph
  trt_graph = trt.create_inference_graph(gdef, ["output"], inp_dims[0])
  o1 = run_graph(gdef, dummy_input)
  o2 = run_graph(trt_graph, dummy_input)
  o3 = run_graph(trt_graph, dummy_input)
  assert np.array_equal(o1, o2)
  assert np.array_equal(o3, o2)  # sanity check
  print("Pass")
