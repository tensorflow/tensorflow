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
"""Utilities to test TF-TensorRT integration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# pylint: disable=unused-import
from tensorflow.compiler.tf2tensorrt.python.ops import trt_ops
# pylint: enable=unused-import
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.compiler.tensorrt import trt_convert
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.tools import saved_model_utils


class TrtConvertTest(test_util.TensorFlowTestCase):
  """Class to test Tensorflow-TensorRT integration python API."""

  def testGetTensorrtRewriterConfig(self):
    """Test case for trt_convert.get_tensorrt_rewriter_config()."""
    rewriter_cfg = trt_convert.get_tensorrt_rewriter_config(
        rewriter_config=None,
        max_batch_size=128,
        max_workspace_size_bytes=1234,
        precision_mode="INT8",
        minimum_segment_size=10,
        is_dynamic_op=True,
        maximum_cached_engines=2,
        cached_engine_batches=[1, 128])
    self.assertEqual(["constfold", "layout", "constfold"],
                     rewriter_cfg.optimizers)
    self.assertEqual(rewriter_config_pb2.RewriterConfig.ONE,
                     rewriter_cfg.meta_optimizer_iterations)
    trt_optimizer = None
    for optimizer in rewriter_cfg.custom_optimizers:
      if optimizer.name == "TensorRTOptimizer":
        self.assertTrue(trt_optimizer is None)
        trt_optimizer = optimizer
    self.assertTrue(trt_optimizer is not None)
    for key in [
        "minimum_segment_size", "max_batch_size", "is_dynamic_op",
        "max_workspace_size_bytes", "precision_mode", "maximum_cached_engines",
        "cached_engine_batches"
    ]:
      self.assertTrue(key in trt_optimizer.parameter_map)
    self.assertEqual(10, trt_optimizer.parameter_map["minimum_segment_size"].i)
    self.assertEqual(128, trt_optimizer.parameter_map["max_batch_size"].i)
    self.assertEqual(True, trt_optimizer.parameter_map["is_dynamic_op"].b)
    self.assertEqual(1234,
                     trt_optimizer.parameter_map["max_workspace_size_bytes"].i)
    self.assertEqual(
        trt_convert._to_bytes("INT8"),
        trt_optimizer.parameter_map["precision_mode"].s)
    self.assertEqual(2, trt_optimizer.parameter_map["maximum_cached_engines"].i)
    self.assertEqual(
        [1, 128], trt_optimizer.parameter_map["cached_engine_batches"].list.i)

  def _GetConfigProto(self):
    """Get ConfigProto for session creation."""
    config = config_pb2.ConfigProto(
        gpu_options=config_pb2.GPUOptions(allow_growth=True))
    return config

  def _GetGraph(self):
    """Get the graph for testing."""
    g = ops.Graph()
    with g.as_default():
      with g.device("/GPU:0"):
        inp = array_ops.placeholder(
            dtype=dtypes.float32, shape=[None, 1, 1], name="input")
        var = variables.VariableV1([[[1.0]]], dtype=dtypes.float32, name="v1")
        add = inp + var.value()
        mul = inp * add
        add = mul + add
        out = array_ops.identity(add, name="output")
    return g, var, inp, out

  def _GetGraphDef(self):
    """Get the graph def for testing."""
    g, var, _, _ = self._GetGraph()
    with self.session(graph=g, config=self._GetConfigProto()) as sess:
      sess.run(var.initializer)
      graph_def = graph_util.convert_variables_to_constants(
          sess, g.as_graph_def(add_shapes=True), ["output"])
    node_name_to_op = {node.name: node.op for node in graph_def.node}
    self.assertEqual({
        "v1": "Const",
        "v1/read": "Identity",
        "input": "Placeholder",
        "add": "Add",
        "mul": "Mul",
        "add_1": "Add",
        "output": "Identity"
    }, node_name_to_op)
    return graph_def

  def _WriteInputSavedModel(self, input_saved_model_dir):
    """Write the saved model as an input for testing."""
    g, var, inp, out = self._GetGraph()
    signature_def = signature_def_utils.build_signature_def(
        inputs={"myinput": utils.build_tensor_info(inp)},
        outputs={"myoutput": utils.build_tensor_info(out)},
        method_name=signature_constants.PREDICT_METHOD_NAME)
    saved_model_builder = builder.SavedModelBuilder(input_saved_model_dir)
    with self.session(graph=g, config=self._GetConfigProto()) as sess:
      sess.run(var.initializer)
      saved_model_builder.add_meta_graph_and_variables(
          sess, [tag_constants.SERVING],
          signature_def_map={"mypredict": signature_def})
    saved_model_builder.save()

  def _TestCreateInferenceGraph(self,
                                input_saved_model_dir=None,
                                output_saved_model_dir=None):
    """General method to test trt_convert.create_inference_graph()."""
    input_graph_def = None if input_saved_model_dir else self._GetGraphDef()
    output_graph_def = trt_convert.create_inference_graph(
        input_graph_def, ["output"],
        input_saved_model_dir=input_saved_model_dir,
        output_saved_model_dir=output_saved_model_dir,
        session_config=self._GetConfigProto())
    graph_defs_to_verify = [output_graph_def]
    if output_saved_model_dir is not None:
      saved_model_graph_def = saved_model_utils.get_meta_graph_def(
          output_saved_model_dir, tag_constants.SERVING).graph_def
      self.assertTrue(isinstance(saved_model_graph_def, graph_pb2.GraphDef))
      graph_defs_to_verify.append(saved_model_graph_def)

    for graph_def in graph_defs_to_verify:
      node_name_to_op = {node.name: node.op for node in graph_def.node}
      self.assertEqual({
          "input": "Placeholder",
          "TRTEngineOp_0": "TRTEngineOp",
          "output": "Identity"
      }, node_name_to_op)

  def testCreateInferenceGraph_BasicConversion(self):
    """Test case for trt_convert.create_inference_graph()."""
    if not trt_convert.is_tensorrt_enabled():
      return

    # Use GraphDef as input.
    self._TestCreateInferenceGraph()

    # Use SavedModel as input.
    tmp_dir = self.get_temp_dir()
    input_saved_model_dir = os.path.join(tmp_dir, "in_dir1")
    output_saved_model_dir = os.path.join(tmp_dir, "out_dir1")
    self._WriteInputSavedModel(input_saved_model_dir)
    self._TestCreateInferenceGraph(input_saved_model_dir,
                                   output_saved_model_dir)

  def _TestRun(self, sess, batch_size, expect_engine_is_run):
    trt_convert.clear_test_values("")
    result = sess.run("output:0", feed_dict={"input:0": [[[1.0]]] * batch_size})
    self.assertAllEqual([[[4.0]]] * batch_size, result)
    execute_engine_test_value = ("done" if expect_engine_is_run else "")
    execute_native_segment_test_value = ("" if expect_engine_is_run else "done")
    self.assertEqual(
        execute_engine_test_value,
        trt_convert.get_test_value("TRTEngineOp_0:ExecuteTrtEngine"))
    self.assertEqual(
        execute_native_segment_test_value,
        trt_convert.get_test_value("TRTEngineOp_0:ExecuteNativeSegment"))

  def testCreateInferenceGraph_MinimumSegmentSize(self):
    if not trt_convert.is_tensorrt_enabled():
      return
    output_graph_def = trt_convert.create_inference_graph(
        self._GetGraphDef(), ["output"],
        minimum_segment_size=5,
        is_dynamic_op=False)
    node_name_to_op = {node.name: node.op for node in output_graph_def.node}
    self.assertEqual({
        "v1/read": "Const",
        "input": "Placeholder",
        "add": "Add",
        "mul": "Mul",
        "add_1": "Add",
        "output": "Identity"
    }, node_name_to_op)

  def testCreateInferenceGraph_DynamicOp(self):
    if not trt_convert.is_tensorrt_enabled():
      return
    trt_convert.enable_test_value()

    tmp_dir = self.get_temp_dir()
    input_saved_model_dir = os.path.join(tmp_dir, "in_dir2")
    output_saved_model_dir = os.path.join(tmp_dir, "out_dir2")
    self._WriteInputSavedModel(input_saved_model_dir)
    output_graph_def = trt_convert.create_inference_graph(
        None,
        None,
        is_dynamic_op=True,
        maximum_cached_engines=2,
        input_saved_model_dir=input_saved_model_dir,
        output_saved_model_dir=output_saved_model_dir,
        session_config=self._GetConfigProto())

    # Test the output GraphDef.
    with ops.Graph().as_default():
      importer.import_graph_def(output_graph_def, name="")
      with self.test_session(config=self._GetConfigProto()) as sess:
        # Run with batch size 1, a new engine is created and cached.
        self._TestRun(sess, 1, True)
        # Run with batch size 2, a new engine is created and cached.
        self._TestRun(sess, 2, True)
        # Run with batch size 3, since the number of cached engines has reached
        # the max, it should evict an old engine and create a new one.
        self._TestRun(sess, 3, True)

    # Test the output SavedModel
    with ops.Graph().as_default():
      with self.test_session(config=self._GetConfigProto()) as sess:
        loader.load(sess, [tag_constants.SERVING], output_saved_model_dir)
        # Run with batch size 1, a new engine is created and cached.
        self._TestRun(sess, 1, True)
        # Run with batch size 2, a new engine is created and cached.
        self._TestRun(sess, 2, True)
        # Run with batch size 3, since the number of cached engines has reached
        # the max, it should evict an old engine and create a new one.
        self._TestRun(sess, 3, True)

  def testCreateInferenceGraph_StaticOp(self):
    if not trt_convert.is_tensorrt_enabled():
      return
    trt_convert.enable_test_value()

    tmp_dir = self.get_temp_dir()
    input_saved_model_dir = os.path.join(tmp_dir, "in_dir3")
    output_saved_model_dir = os.path.join(tmp_dir, "out_dir3")
    self._WriteInputSavedModel(input_saved_model_dir)
    output_graph_def = trt_convert.create_inference_graph(
        None,
        None,
        max_batch_size=1,
        is_dynamic_op=False,
        maximum_cached_engines=2,  # This is noop, added just for testing.
        input_saved_model_dir=input_saved_model_dir,
        output_saved_model_dir=output_saved_model_dir,
        session_config=self._GetConfigProto())

    # Test the output GraphDef.
    with ops.Graph().as_default():
      importer.import_graph_def(output_graph_def, name="")
      with self.test_session(config=self._GetConfigProto()) as sess:
        # Run with batch size 1, the default engine embedded in the graphdef
        # will be used.
        self._TestRun(sess, 1, True)
        # Run with batch size 2, which exceed the max_batch_size, it should fall
        # back to TF function.
        self._TestRun(sess, 2, False)

    # Test the output SavedModel
    with ops.Graph().as_default():
      with self.test_session(config=self._GetConfigProto()) as sess:
        loader.load(sess, [tag_constants.SERVING], output_saved_model_dir)
        # Run with batch size 1, the default engine embedded in the graphdef
        # will be used.
        self._TestRun(sess, 1, True)
        # Run with batch size 2, which exceed the max_batch_size, it should fall
        # back to TF function.
        self._TestRun(sess, 2, False)


if __name__ == "__main__":
  test.main()
