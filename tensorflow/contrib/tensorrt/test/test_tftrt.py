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

import argparse
import numpy as np

# normally we should do import tensorflow as tf and then
# tf.placeholder, tf.constant, tf.nn.conv2d etc but
# it looks like internal builds don't like it so
# importing every module individually

from tensorflow.contrib import tensorrt as trt
from tensorflow.core.protobuf import config_pb2 as cpb2
from tensorflow.core.protobuf import rewriter_config_pb2 as rwpb2
from tensorflow.python.client import session as csess
from tensorflow.python.framework import constant_op as cop
from tensorflow.python.framework import dtypes as dtypes
from tensorflow.python.framework import importer as importer
from tensorflow.python.framework import ops as ops
from tensorflow.python.ops import array_ops as aops
from tensorflow.python.ops import nn as nn
from tensorflow.python.ops import nn_ops as nn_ops


def get_simple_graph_def():
  """Create a simple graph and return its graph_def."""
  g = ops.Graph()
  with g.as_default():
    a = aops.placeholder(
        dtype=dtypes.float32, shape=(None, 24, 24, 2), name="input")
    e = cop.constant(
        [[[[1., 0.5, 4., 6., 0.5, 1.], [1., 0.5, 1., 1., 0.5, 1.]]]],
        name="weights",
        dtype=dtypes.float32)
    conv = nn.conv2d(
        input=a, filter=e, strides=[1, 2, 2, 1], padding="SAME", name="conv")
    b = cop.constant(
        [4., 1.5, 2., 3., 5., 7.], name="bias", dtype=dtypes.float32)
    t = nn.bias_add(conv, b, name="biasAdd")
    relu = nn.relu(t, "relu")
    idty = aops.identity(relu, "ID")
    v = nn_ops.max_pool(
        idty, [1, 2, 2, 1], [1, 2, 2, 1], "VALID", name="max_pool")
    aops.squeeze(v, name="output")
  return g.as_graph_def()


def execute_graph(gdef, dumm_inp):
  """Run given graphdef once."""
  print("executing")
  gpu_options = cpb2.GPUOptions(per_process_gpu_memory_fraction=0.50)
  sessconfig = cpb2.ConfigProto(gpu_options=gpu_options)
  ops.reset_default_graph()
  g = ops.Graph()
  with g.as_default():
    inp, out = importer.import_graph_def(
        graph_def=gdef, return_elements=["input", "output"])
    inp = inp.outputs[0]
    out = out.outputs[0]
  with csess.Session(config=sessconfig, graph=g) as sess:
    val = sess.run(out, {inp: dumm_inp})
  return val


# Use real data that is representative of the inference dataset
# for calibration. For this test script it is random data.
def execute_calibration(gdef, dumm_inp):
  """Run given calibration graph multiple times."""
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
    # run over real calibration data here, we are mimicking a calibration set of
    # 30 different batches. Use as much calibration data as you want
    for _ in range(30):
      val = sess.run(out, {inp: dumm_inp})
  return val


def user(run_graph=execute_graph, run_calibration=execute_calibration):
  """Example function that converts a graph to TFTRT graph."""

  inp_dims = (100, 24, 24, 2)
  dummy_input = np.random.random_sample(inp_dims)
  orig_graph = get_simple_graph_def()  # use a frozen graph for inference
  # Get optimized graph
  trt_graph = trt.create_inference_graph(
      input_graph_def=orig_graph,
      outputs=["output"],
      max_batch_size=inp_dims[0],
      max_workspace_size_bytes=1 << 25,
      precision_mode="FP32",  # TRT Engine precision "FP32","FP16" or "INT8"
      minimum_segment_size=2,  # minimum number of nodes in an engine
      is_dynamic_op=False,
      maximum_cached_engines=1,
      cached_engine_batches=[])
  o1 = run_graph(orig_graph, dummy_input)
  o2 = run_graph(trt_graph, dummy_input)
  o3 = run_graph(trt_graph, dummy_input)
  assert np.array_equal(o1, o2)
  assert np.array_equal(o3, o2)  # sanity check
  fp16_graph = trt.create_inference_graph(
      input_graph_def=orig_graph,
      outputs=["output"],
      max_batch_size=inp_dims[0],
      max_workspace_size_bytes=1 << 25,
      precision_mode="FP16",  # TRT Engine precision "FP32","FP16" or "INT8"
      minimum_segment_size=2,  # minimum number of nodes in an engine
      is_dynamic_op=False,
      maximum_cached_engines=1,
      cached_engine_batches=[])
  int8_calib_gdef = trt.create_inference_graph(
      input_graph_def=orig_graph,
      outputs=["output"],
      max_batch_size=inp_dims[0],
      max_workspace_size_bytes=1 << 25,
      precision_mode="INT8",  # TRT Engine precision "FP32","FP16" or "INT8"
      minimum_segment_size=2,  # minimum number of nodes in an engine
      is_dynamic_op=False,
      maximum_cached_engines=1,
      cached_engine_batches=[])
  o4 = run_graph(fp16_graph, dummy_input)
  _ = run_calibration(int8_calib_gdef, dummy_input)
  int8_graph = trt.calib_graph_to_infer_graph(int8_calib_gdef)
  o5 = run_graph(int8_graph, dummy_input)
  assert np.allclose(o1, o4)
  assert np.allclose(o1, o5)
  print("Pass")


def auto():
  """Run the conversion as an optimization pass."""
  inp_dims = (100, 24, 24, 2)
  dummy_input = np.random.random_sample(inp_dims)
  orig_graph = get_simple_graph_def()
  opt_config = rwpb2.RewriterConfig()
  opt_config.optimizers.extend(["constfold", "layout"])
  custom_op = opt_config.custom_optimizers.add()
  custom_op.name = "TensorRTOptimizer"
  custom_op.parameter_map["minimum_segment_size"].i = 3
  custom_op.parameter_map["precision_mode"].s = "FP32"
  custom_op.parameter_map["max_batch_size"].i = inp_dims[0]
  custom_op.parameter_map["max_workspace_size_bytes"].i = 1 << 25
  print(custom_op)
  gpu_options = cpb2.GPUOptions(per_process_gpu_memory_fraction=0.50)
  graph_options = cpb2.GraphOptions(rewrite_options=opt_config)
  sessconfig = cpb2.ConfigProto(
      gpu_options=gpu_options, graph_options=graph_options)
  print(sessconfig)
  g = ops.Graph()
  ops.reset_default_graph()
  with g.as_default():
    inp, out = importer.import_graph_def(
        graph_def=orig_graph, return_elements=["input", "output"])
    inp = inp.outputs[0]
    out = out.outputs[0]
    with csess.Session(config=sessconfig, graph=g) as sess:
      val = sess.run(out, {inp: dummy_input})
  print(val.shape)


if "__main__" in __name__:
  P = argparse.ArgumentParser(
      prog="tftrt_test",
      description="Example utilization of TensorFlow-TensorRT integration")
  P.add_argument(
      "--automatic",
      "-a",
      action="store_true",
      help="Do TRT conversion automatically",
      default=False)
  flags, unparsed = P.parse_known_args()
  if flags.automatic:
    auto()
  else:
    user()
