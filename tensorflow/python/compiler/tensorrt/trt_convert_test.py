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

from tensorflow.compiler.tf2tensorrt.wrap_py_utils import is_tensorrt_enabled
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.compiler.tensorrt import trt_convert
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.tools import saved_model_utils
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import save
from tensorflow.python.training.tracking import tracking

_SAVED_MODEL_SIGNATURE_KEY = "mypredict"


class TrtConvertTest(test_util.TensorFlowTestCase):
  """Class to test Tensorflow-TensorRT integration python API."""

  # Use a small max_workspace_size for tests so they don't consume too much GPU
  # memory.
  _TRT_MAX_WORKSPACE_SIZE_BYTES = 2 << 20

  def testGetTensorrtRewriterConfig(self):
    """Test case for TrtGraphConverter.get_tensorrt_rewriter_config()."""
    if not is_tensorrt_enabled():
      return
    rewriter_cfg = trt_convert.TrtGraphConverter.get_tensorrt_rewriter_config(
        rewriter_config_template=None,
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
    # The graph computes (input+1)^2, it looks like:
    #
    # input (Placeholder)  v1 (Variable)
    #               |   \ /
    #                \   +
    #                 \ / \
    #                  *   |
    #                   \ /
    #                    +
    #                    |
    #                 output (Identity)
    g = ops.Graph()
    with g.as_default():
      with g.device("/GPU:0"):
        inp = array_ops.placeholder(
            dtype=dtypes.float32, shape=[None, 1, 1], name="input")
        var = variables.VariableV1([[[1.0]]],
                                   dtype=dtypes.float32,
                                   name="v1",
                                   use_resource=False)
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
    self.assertEqual(
        {
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
          signature_def_map={_SAVED_MODEL_SIGNATURE_KEY: signature_def})
    saved_model_builder.save()

  def _ConvertGraph(self,
                    input_saved_model_dir=None,
                    output_saved_model_dir=None,
                    need_calibration=False,
                    max_batch_size=1,
                    minimum_segment_size=3,
                    is_dynamic_op=False,
                    maximum_cached_engines=1,
                    use_function_backup=False):
    """Helper method to convert a GraphDef or SavedModel using TF-TRT."""
    converter = trt_convert.TrtGraphConverter(
        input_saved_model_dir=input_saved_model_dir,
        input_saved_model_signature_key=_SAVED_MODEL_SIGNATURE_KEY,
        input_graph_def=None if input_saved_model_dir else self._GetGraphDef(),
        nodes_blacklist=["output"],
        session_config=self._GetConfigProto(),
        max_batch_size=max_batch_size,
        max_workspace_size_bytes=TrtConvertTest._TRT_MAX_WORKSPACE_SIZE_BYTES,
        precision_mode=(trt_convert.TrtPrecisionMode.INT8 if need_calibration
                        else trt_convert.TrtPrecisionMode.FP32),
        minimum_segment_size=minimum_segment_size,
        is_dynamic_op=is_dynamic_op,
        maximum_cached_engines=maximum_cached_engines,
        use_function_backup=use_function_backup)
    conversion_result = converter.convert()

    if context.executing_eagerly():
      output_graph_def = conversion_result.graph.as_graph_def()
    else:
      output_graph_def = conversion_result

      if need_calibration:

        class CalibrationData(object):

          def __init__(self):
            self._data = 0

          def next(self):
            self._data += 1
            return {"input:0": [[[self._data]]]}

        output_graph_def = converter.calibrate(
            fetch_names=["output:0"],
            num_runs=10,
            feed_dict_fn=CalibrationData().next)

    if output_saved_model_dir is not None:
      converter.save(output_saved_model_dir=output_saved_model_dir)
    return output_graph_def

  def _TestTrtGraphConverter(self,
                             input_saved_model_dir=None,
                             output_saved_model_dir=None,
                             need_calibration=False,
                             is_dynamic_op=False):
    """General method to test trt_convert.TrtGraphConverter()."""
    output_graph_def = self._ConvertGraph(
        input_saved_model_dir=input_saved_model_dir,
        output_saved_model_dir=output_saved_model_dir,
        need_calibration=need_calibration,
        is_dynamic_op=is_dynamic_op,
        use_function_backup=need_calibration)
    graph_defs_to_verify = [output_graph_def]

    if output_saved_model_dir:
      if context.executing_eagerly():
        root = load.load(output_saved_model_dir)
        saved_model_graph_def = root.signatures[
            _SAVED_MODEL_SIGNATURE_KEY].graph.as_graph_def()
      else:
        saved_model_graph_def = saved_model_utils.get_meta_graph_def(
            output_saved_model_dir, tag_constants.SERVING).graph_def
      self.assertTrue(isinstance(saved_model_graph_def, graph_pb2.GraphDef))
      graph_defs_to_verify.append(saved_model_graph_def)

    for graph_def in graph_defs_to_verify:
      node_name_to_op = {node.name: node.op for node in graph_def.node}
      if context.executing_eagerly():
        # In V2 the actual graph could be inside a function.
        for func in graph_def.library.function:
          node_name_to_op.update({node.name: node.op for node in func.node_def})
        self.assertIn("TRTEngineOp_0", node_name_to_op)
        self.assertEqual("TRTEngineOp", node_name_to_op["TRTEngineOp_0"])
      else:
        self.assertEqual(
            {
                "input": "Placeholder",
                "TRTEngineOp_0": "TRTEngineOp",
                "output": "Identity"
            }, node_name_to_op)

      if need_calibration:
        trt_engine_nodes = [
            node for node in graph_def.node if node.op == "TRTEngineOp"
        ]
        self.assertNotEmpty(trt_engine_nodes)
        for node in trt_engine_nodes:
          self.assertTrue(len(node.attr["calibration_data"].s))
        # Run the calibrated graph.
        # TODO(laigd): consider having some input where the answer is different.
        with ops.Graph().as_default():
          importer.import_graph_def(graph_def, name="")
          with self.session(config=self._GetConfigProto()) as sess:
            for test_data in range(10):
              self.assertEqual(
                  (test_data + 1.0)**2,
                  sess.run("output:0", feed_dict={"input:0": [[[test_data]]]}))

  @test_util.deprecated_graph_mode_only
  def testTrtGraphConverter_BasicConversion(self):
    """Test case for trt_convert.TrtGraphConverter()."""
    if not is_tensorrt_enabled():
      return

    tmp_dir = self.get_temp_dir()
    input_saved_model_dir = os.path.join(tmp_dir, "in_dir1")
    self._WriteInputSavedModel(input_saved_model_dir)

    for need_calibration in [False, True]:
      # Use GraphDef as input.
      self._TestTrtGraphConverter()

      # Use SavedModel as input.
      output_saved_model_dir = os.path.join(
          tmp_dir, "out_dir1%s" % ("_int8" if need_calibration else ""))
      self._TestTrtGraphConverter(
          input_saved_model_dir=input_saved_model_dir,
          output_saved_model_dir=output_saved_model_dir,
          need_calibration=need_calibration)

  @test_util.run_v2_only
  def testTrtGraphConverter_BasicConversion_v2(self):
    """Test case for trt_convert.TrtGraphConverter()."""
    if not is_tensorrt_enabled():
      return

    # TODO(laigd): we need to use ops like conv2d so Grappler can infer the
    # shapes (at least rank) of the tensors, so we're able to build an TRT
    # engine in dynamic mode. Currently shape information is not propagate from
    # ConcreteFunction to GraphDef, need to investigate and fix it.
    class SimpleModel(tracking.AutoTrackable):

      def __init__(self):
        self.v = None

      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(shape=[None, 24, 24, 2], dtype=dtypes.float32)
      ])
      def run(self, inp):
        if self.v is None:
          self.v = variables.Variable([[[[1., 0.5, 4., 6., 0.5, 1.],
                                         [1., 0.5, 1., 1., 0.5, 1.]]]])
        conv = gen_nn_ops.conv2d(
            input=inp, filter=self.v, strides=[1, 2, 2, 1], padding="SAME")
        identity = array_ops.identity(conv)
        return identity

    tmp_dir = self.get_temp_dir()
    input_saved_model_dir = os.path.join(tmp_dir, "in_dir1_v2")
    root = SimpleModel()
    save.save(root, input_saved_model_dir,
              {_SAVED_MODEL_SIGNATURE_KEY: root.run})

    # Convert the SavedModel and verify the result.
    output_saved_model_dir = os.path.join(tmp_dir, "out_dir1_v2")
    self._TestTrtGraphConverter(
        input_saved_model_dir=input_saved_model_dir,
        output_saved_model_dir=output_saved_model_dir,
        is_dynamic_op=True)

  def _TestRun(self,
               sess,
               batch_size,
               use_function_backup=False,
               expect_engine_is_run=True):
    try:
      result = sess.run(
          "output:0", feed_dict={"input:0": [[[1.0]]] * batch_size})
      self.assertAllEqual([[[4.0]]] * batch_size, result)
    except errors.OpError as e:
      # This should happen only when fallback path is disabled and TRT engine
      # fails to run.
      self.assertTrue(not use_function_backup and not expect_engine_is_run)
      self.assertIn("Fallback path is disabled, for TRTEngineOp_0", str(e))

  @test_util.deprecated_graph_mode_only
  def testTrtGraphConverter_MinimumSegmentSize(self):
    if not is_tensorrt_enabled():
      return
    output_graph_def = self._ConvertGraph(minimum_segment_size=5)
    node_name_to_op = {node.name: node.op for node in output_graph_def.node}
    self.assertEqual(
        {
            "v1/read": "Const",
            "input": "Placeholder",
            "add": "Add",
            "mul": "Mul",
            "add_1": "Add",
            "output": "Identity"
        }, node_name_to_op)

  @test_util.deprecated_graph_mode_only
  def testTrtGraphConverter_DynamicOp(self):
    if not is_tensorrt_enabled():
      return

    tmp_dir = self.get_temp_dir()
    input_saved_model_dir = os.path.join(tmp_dir, "in_dir2")
    output_saved_model_dir = os.path.join(tmp_dir, "out_dir2")
    self._WriteInputSavedModel(input_saved_model_dir)
    output_graph_def = self._ConvertGraph(
        input_saved_model_dir=input_saved_model_dir,
        output_saved_model_dir=output_saved_model_dir,
        is_dynamic_op=True,
        maximum_cached_engines=2,
        use_function_backup=False)  # Disallow fallback.

    # Test the output GraphDef.
    with ops.Graph().as_default():
      importer.import_graph_def(output_graph_def, name="")
      with self.session(config=self._GetConfigProto()) as sess:
        # Run with batch size 1, a new engine is created and cached.
        self._TestRun(sess, 1)
        # Run with batch size 2, a new engine is created and cached.
        self._TestRun(sess, 2)
        # Run with batch size 3, since the number of cached engines has reached
        # the max, it should evict an old engine and create a new one.
        self._TestRun(sess, 3)

    # Test the output SavedModel
    with ops.Graph().as_default():
      with self.session(config=self._GetConfigProto()) as sess:
        loader.load(sess, [tag_constants.SERVING], output_saved_model_dir)
        # Run with batch size 1, a new engine is created and cached.
        self._TestRun(sess, 1)
        # Run with batch size 2, a new engine is created and cached.
        self._TestRun(sess, 2)
        # Run with batch size 3, since the number of cached engines has reached
        # the max, it should evict an old engine and create a new one.
        self._TestRun(sess, 3)

  def _TestStaticOp(self, use_function_backup):
    if not is_tensorrt_enabled():
      return

    tmp_dir = self.get_temp_dir()
    input_saved_model_dir = os.path.join(tmp_dir, "in_dir3")
    output_saved_model_dir = os.path.join(tmp_dir, "out_dir3")
    self._WriteInputSavedModel(input_saved_model_dir)
    output_graph_def = self._ConvertGraph(
        input_saved_model_dir=input_saved_model_dir,
        output_saved_model_dir=output_saved_model_dir,
        maximum_cached_engines=2,  # This is noop, added just for testing.
        use_function_backup=use_function_backup)

    # Test the output GraphDef.
    with ops.Graph().as_default():
      importer.import_graph_def(output_graph_def, name="")
      with self.session(config=self._GetConfigProto()) as sess:
        # Run with batch size 1, the default engine embedded in the graphdef
        # will be used.
        self._TestRun(
            sess,
            1,
            use_function_backup=use_function_backup,
            expect_engine_is_run=True)
        # Run with batch size 2, which exceed the max_batch_size, it should try
        # to fall back to TF function.
        self._TestRun(
            sess,
            2,
            use_function_backup=use_function_backup,
            expect_engine_is_run=False)

    # Test the output SavedModel
    with ops.Graph().as_default():
      with self.session(config=self._GetConfigProto()) as sess:
        loader.load(sess, [tag_constants.SERVING], output_saved_model_dir)
        # Run with batch size 1, the default engine embedded in the graphdef
        # will be used.
        self._TestRun(
            sess,
            1,
            use_function_backup=use_function_backup,
            expect_engine_is_run=True)
        # Run with batch size 2, which exceed the max_batch_size, it should try
        # to fall back to TF function.
        self._TestRun(
            sess,
            2,
            use_function_backup=use_function_backup,
            expect_engine_is_run=False)

  @test_util.deprecated_graph_mode_only
  def testTrtGraphConverter_StaticOp_NoFallback(self):
    self._TestStaticOp(use_function_backup=False)

  @test_util.deprecated_graph_mode_only
  def testTrtGraphConverter_StaticOp_WithFallback(self):
    self._TestStaticOp(use_function_backup=True)


if __name__ == "__main__":
  test.main()
