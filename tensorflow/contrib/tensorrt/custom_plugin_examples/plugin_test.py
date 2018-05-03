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
"""Script to show usage of TensorRT custom op & plugin."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from tensorflow.contrib import tensorrt
from tensorflow.contrib.tensorrt import custom_plugin_examples
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops


def get_plugin_graph_def():
  """Create a simple graph and return its graph_def."""
  g = ops.Graph()
  with g.as_default():
    a = array_ops.placeholder(
        dtype=dtypes.float32, shape=(None, 24, 24, 2), name="input")
    relu = nn.relu(a, "relu")
    v = nn_ops.max_pool(
        relu, [1, 2, 2, 1], [1, 2, 2, 1], "VALID", name="max_pool")

    # insert custom_op in the graph
    v = custom_plugin_examples.inc_op(v, inc=[16.5], name="plugin_test")

    v = v * 2.0
    v = nn.relu(v)
    v = nn.relu(v)
    array_ops.squeeze(v, name="output")
  return g.as_graph_def()


def run_graph(gdef, dumm_inp):
  """Run given graphdef once."""
  gpu_options = config_pb2.GPUOptions(per_process_gpu_memory_fraction=0.50)
  ops.reset_default_graph()
  g = ops.Graph()
  with g.as_default():
    inp, out = importer.import_graph_def(
        graph_def=gdef, return_elements=["input", "output"])
    inp = inp.outputs[0]
    out = out.outputs[0]

  with session.Session(
      config=config_pb2.ConfigProto(gpu_options=gpu_options), graph=g) as sess:
    val = sess.run(out, {inp: dumm_inp})
  return val


if "__main__" in __name__:
  inp_dims = (5, 24, 24, 2)
  dummy_input = numpy.ones(inp_dims).astype(numpy.float32)
  orig_graph = get_plugin_graph_def()  # graph with plugin node

  # trigger conversion.
  # plugin nodes have been registered during import, converter will be able to
  # create corresponding plugin layer during conversion.
  trt_graph = tensorrt.create_inference_graph(
      input_graph_def=orig_graph,
      outputs=["output"],
      max_batch_size=inp_dims[0],
      max_workspace_size_bytes=1 << 25,
      precision_mode="FP32",
      minimum_segment_size=2)
  o2 = run_graph(trt_graph, dummy_input)
  if o2.reshape([-1])[0] == 35:
    print("pass")
  else:
    raise RuntimeError("contrib/tensorrt/custom_plugin_examples wrong result")
