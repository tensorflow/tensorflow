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
"""script to convert and execute TF-TensorRT graph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import tensorrt as trt
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.training import training
from tensorflow.contrib.tensorrt.test.utilities import get_all_variables

OUTPUT_NODE = "output"
INPUT_NODE = "input"
CALIB_COUNT = 5  # calibration iteration


class RunTest:
  """base class to run TR-TRT conversion and execution"""

  def __init__(self):
    self.clean()

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.clean()

  def clean(self):
    self.tftrt = {}
    self.tftrt_conversion_flag = {}
    self.tftrt_nb_nodes = {}
    self.tftrt_result = {}
    self.tftrt_dynamic_conversion_flag = {}
    self.tftrt_dynamic_result = {}
    self.check_file = None
    self.native_network = None

  def run_test(self,
               network,
               static_mode_list,
               dynamic_mode_list,
               dummy_input,
               file_name=None):
    self.native_network = network()
    success = True
    initialization = False
    if file_name != None:
      initialization = True
      self.check_file = file_name
    self.native_result, self.native_nb_nodes = self.execute_graph(
        self.native_network, dummy_input, initialization)
    for mode in static_mode_list:
      try:
        self.run_static_convert_network(mode, dummy_input, initialization)
        self.tftrt_conversion_flag[mode] = True
      except Exception as inst:
        self.tftrt_conversion_flag[mode] = False
        success = False
    for mode in dynamic_mode_list:
      try:
        self.run_dynamic_convert_network(mode, dummy_input, initialization)
        self.tftrt_dynamic_conversion_flag[mode] = True
      except Exception as inst:
        self.tftrt_dynamic_conversion_flag[mode] = False
        success = False
    return success

  def run_dynamic_convert_network(self, mode, dummy_input, initialization=True):
    inp_dims = dummy_input.shape
    if mode == "FP32" or mode == "FP16":
      opt_config = rewriter_config_pb2.RewriterConfig()
      opt_config.optimizers.extend(["constfold", "layout"])
      custom_op = opt_config.custom_optimizers.add()
      custom_op.name = "TensorRTOptimizer"
      custom_op.parameter_map["minimum_segment_size"].i = 3
      custom_op.parameter_map["precision_mode"].s = mode
      custom_op.parameter_map["max_batch_size"].i = inp_dims[0]
      custom_op.parameter_map["max_workspace_size_bytes"].i = 1 << 25
      print(custom_op)
      gpu_options = config_pb2.GPUOptions(per_process_gpu_memory_fraction=0.50)
      graph_options = config_pb2.GraphOptions(rewrite_options=opt_config)
      sessconfig = config_pb2.ConfigProto(
          gpu_options=gpu_options, graph_options=graph_options)
      print(sessconfig)
      g = ops.Graph()
      ops.reset_default_graph()
      with g.as_default():
        inp, out = importer.import_graph_def(
            graph_def=self.native_network, return_elements=["input", "output"])
        inp = inp.outputs[0]
        out = out.outputs[0]
        with session.Session(config=sessconfig, graph=g) as sess:
          if (initialization):
            names_var_list = get_all_variables(sess)
            saver = training.Saver(names_var_list)
            saver.restore(sess, self.check_file)
          self.tftrt_dynamic_result[mode] = sess.run(out, {inp: dummy_input})
    else:
      raise Exception("dynamic op mode: " + mode + " not supported")

  def run_static_convert_network(self, mode, dummy_input, initialization=True):
    inp_dims = dummy_input.shape
    if mode == "FP32" or mode == "FP16" or mode == "INT8":
      trt_graph = trt.create_inference_graph(
          input_graph_def=self.native_network,
          outputs=[OUTPUT_NODE],
          max_batch_size=inp_dims[0],
          max_workspace_size_bytes=1 << 25,
          precision_mode=mode,  # TRT Engine precision "FP32","FP16" or "INT8"
          minimum_segment_size=2  # minimum number of nodes in an engine
      )
      if mode == "INT8":
        _ = self.execute_calibration(trt_graph, dummy_input, initialization)
        trt_graph = trt.calib_graph_to_infer_graph(trt_graph)
      trt_result, nb_nodes = self.execute_graph(trt_graph, dummy_input,
                                                initialization)
      self.tftrt[mode] = trt_graph
      self.tftrt_nb_nodes[mode] = nb_nodes
      self.tftrt_result[mode] = trt_result
    else:
      raise Exception("mode: " + mode + " not supported")

  def execute_graph(self, gdef, dummy_input, initialization=True):
    """Run given graphdef once."""
    gpu_options = config_pb2.GPUOptions()
    sessconfig = config_pb2.ConfigProto(gpu_options=gpu_options)
    ops.reset_default_graph()
    g = ops.Graph()
    nb_nodes = 0
    with g.as_default():
      inp, out = importer.import_graph_def(
          graph_def=gdef, return_elements=[INPUT_NODE, OUTPUT_NODE], name="")
      nb_nodes = len(g.get_operations())
      inp = inp.outputs[0]
      out = out.outputs[0]
    with session.Session(config=sessconfig, graph=g) as sess:
      if (initialization):
        names_var_list = get_all_variables(sess)
        saver = training.Saver(names_var_list)
        saver.restore(sess, self.check_file)
      val = sess.run(out, {inp: dummy_input})
    return val, nb_nodes

  # Use real data that is representative of the inference dataset
  # for calibration. For this test script it is random data.
  def execute_calibration(self, gdef, dummy_input, initialization=True):
    """Run given calibration graph multiple times."""
    gpu_options = config_pb2.GPUOptions()
    ops.reset_default_graph()
    g = ops.Graph()
    with g.as_default():
      inp, out = importer.import_graph_def(
          graph_def=gdef, return_elements=[INPUT_NODE, OUTPUT_NODE], name="")
      inp = inp.outputs[0]
      out = out.outputs[0]
    with session.Session(
        config=config_pb2.ConfigProto(gpu_options=gpu_options),
        graph=g) as sess:
      if (initialization):
        names_var_list = get_all_variables(sess)
        saver = training.Saver(names_var_list)
        saver.restore(sess, self.check_file)
      for _ in range(CALIB_COUNT):
        val = sess.run(out, {inp: dummy_input})
    return val
