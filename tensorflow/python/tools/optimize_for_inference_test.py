# pylint: disable=g-bad-file-header
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
"""Tests for tensorflow.python.client.graph_util."""

import numpy as np

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import math_ops  # pylint: disable=unused-import
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test
from tensorflow.python.tools import optimize_for_inference_lib


class OptimizeForInferenceTest(test.TestCase):

  def create_node_def(self, op, name, inputs):
    new_node = node_def_pb2.NodeDef()
    new_node.op = op
    new_node.name = name
    for input_name in inputs:
      new_node.input.extend([input_name])
    return new_node

  def create_constant_node_def(self, name, value, dtype, shape=None):
    node = self.create_node_def("Const", name, [])
    self.set_attr_dtype(node, "dtype", dtype)
    self.set_attr_tensor(node, "value", value, dtype, shape)
    return node

  def set_attr_dtype(self, node, key, value):
    node.attr[key].CopyFrom(
        attr_value_pb2.AttrValue(type=value.as_datatype_enum))

  def set_attr_tensor(self, node, key, value, dtype, shape=None):
    node.attr[key].CopyFrom(
        attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
            value, dtype=dtype, shape=shape)))

  def testOptimizeForInference(self):
    self.maxDiff = 1000
    unused_constant_name = "unused_constant"
    unconnected_add_name = "unconnected_add"
    a_constant_name = "a_constant"
    b_constant_name = "b_constant"
    a_check_name = "a_check"
    b_check_name = "b_check"
    a_identity_name = "a_identity"
    b_identity_name = "b_identity"
    add_name = "add"
    unused_output_add_name = "unused_output_add"
    graph_def = graph_pb2.GraphDef()
    unused_constant = self.create_constant_node_def(
        unused_constant_name, value=0, dtype=dtypes.float32, shape=[])
    graph_def.node.extend([unused_constant])
    unconnected_add_node = self.create_node_def(
        "Add", unconnected_add_name,
        [unused_constant_name, unused_constant_name])
    self.set_attr_dtype(unconnected_add_node, "T", dtypes.float32)
    graph_def.node.extend([unconnected_add_node])
    a_constant = self.create_constant_node_def(
        a_constant_name, value=1, dtype=dtypes.float32, shape=[])
    graph_def.node.extend([a_constant])
    a_check_node = self.create_node_def("CheckNumerics", a_check_name,
                                        [a_constant_name])
    graph_def.node.extend([a_check_node])
    a_identity_node = self.create_node_def(
        "Identity", a_identity_name, [a_constant_name, "^" + a_check_name])
    graph_def.node.extend([a_identity_node])
    b_constant = self.create_constant_node_def(
        b_constant_name, value=1, dtype=dtypes.float32, shape=[])
    graph_def.node.extend([b_constant])
    b_check_node = self.create_node_def("CheckNumerics", b_check_name,
                                        [b_constant_name])
    graph_def.node.extend([b_check_node])
    b_identity_node = self.create_node_def(
        "Identity", b_identity_name, [b_constant_name, "^" + b_check_name])
    graph_def.node.extend([b_identity_node])
    add_node = self.create_node_def("Add", add_name,
                                    [a_identity_name, b_identity_name])
    self.set_attr_dtype(add_node, "T", dtypes.float32)
    graph_def.node.extend([add_node])
    unused_output_add_node = self.create_node_def("Add", unused_output_add_name,
                                                  [add_name, b_constant_name])
    self.set_attr_dtype(unused_output_add_node, "T", dtypes.float32)
    graph_def.node.extend([unused_output_add_node])

    expected_output = graph_pb2.GraphDef()
    a_constant = self.create_constant_node_def(
        a_constant_name, value=1, dtype=dtypes.float32, shape=[])
    expected_output.node.extend([a_constant])
    b_constant = self.create_constant_node_def(
        b_constant_name, value=1, dtype=dtypes.float32, shape=[])
    expected_output.node.extend([b_constant])
    add_node = self.create_node_def("Add", add_name,
                                    [a_constant_name, b_constant_name])
    self.set_attr_dtype(add_node, "T", dtypes.float32)
    expected_output.node.extend([add_node])

    output = optimize_for_inference_lib.optimize_for_inference(
        graph_def, [], [add_name], dtypes.float32.as_datatype_enum)
    self.assertProtoEquals(expected_output, output)

  def testConvertPlaceholderToConstant(self):
    """Build the placeholder testing graph."""
    placeholder_name = "phase_train"
    relu_name = "r_relu"

    g_def = graph_pb2.GraphDef()

    ph_node = node_def_pb2.NodeDef()
    ph_node.op = "Placeholder"
    ph_node.name = placeholder_name

    self.set_attr_dtype(ph_node, "dtype", dtypes.bool)
    g_def.node.extend([ph_node])

    r_node = self.create_node_def("Relu", relu_name, [placeholder_name])
    g_def.node.extend([r_node])

    opt_graph_def = optimize_for_inference_lib.optimize_for_inference(
        g_def,
        [],
        [relu_name],
        dtypes.float32.as_datatype_enum,
        placeholder_to_const_names=["phase_train=False"],
    )
    for node in opt_graph_def.node:
      self.assertNotEqual("Placeholder", node.op)
      if node.name == "phase_train":
        self.assertEqual(node.op, "Const")
        const_value = optimize_for_inference_lib.values_from_const(node)
        self.assertEqual(const_value, False)

  def testConvertPlaceholderToConstant2(self):
    """Build the placeholder testing graph."""
    placeholder_name = "phase_train"
    relu_name = "r_relu"

    g_def = graph_pb2.GraphDef()

    ph_node = node_def_pb2.NodeDef()
    ph_node.op = "Placeholder"
    ph_node.name = placeholder_name

    self.set_attr_dtype(ph_node, "dtype", dtypes.bool)
    g_def.node.extend([ph_node])

    r_node = self.create_node_def("Relu", relu_name, [placeholder_name])
    g_def.node.extend([r_node])

    opt_graph_def = optimize_for_inference_lib.convert_placeholder_to_const(
        g_def, ["phase_train=True"]
    )
    for node in opt_graph_def.node:
      self.assertNotEqual("Placeholder", node.op)
      if node.name == "phase_train":
        self.assertEqual(node.op, "Const")
        const_value = optimize_for_inference_lib.values_from_const(node)
        self.assertEqual(const_value, True)

  def testConvertPlaceholderWithDefaultToConstant(self):
    """Build the placeholder_with_default testing graph."""
    placeholder_name = "keras_learning_phase"
    a_constant_name = "a_constant"
    relu_name = "r_relu"

    g_def = graph_pb2.GraphDef()
    const_node = self.create_constant_node_def(
        a_constant_name, value=True, dtype=dtypes.bool, shape=[]
    )
    g_def.node.extend([const_node])

    ph_node = self.create_node_def(
        "PlaceholderWithDefault", placeholder_name, [a_constant_name]
    )
    self.set_attr_dtype(ph_node, "dtype", dtypes.bool)
    g_def.node.extend([ph_node])

    r_node = self.create_node_def("Relu", relu_name, [placeholder_name])
    g_def.node.extend([r_node])

    opt_graph_def = optimize_for_inference_lib.convert_placeholder_to_const(
        g_def
    )
    for node in opt_graph_def.node:
      self.assertNotEqual("PlaceholderWithDefault", node.op)
      if node.name == "keras_learning_phase":
        self.assertEqual(node.op, "Const")
        const_value = optimize_for_inference_lib.values_from_const(node)
        # Notice optimize_for_inference rewrites keras_learning_phase to False
        self.assertEqual(const_value, False)

  def testConvertPlaceholderWithDefaultToConstant2(self):
    """Build the placeholder_with_default testing graph."""
    placeholder_name = "keras_learning_phase"
    a_constant_name = "a_constant"
    relu_name = "r_relu"

    g_def = graph_pb2.GraphDef()
    const_node = self.create_constant_node_def(
        a_constant_name, value=True, dtype=dtypes.bool, shape=[]
    )
    g_def.node.extend([const_node])

    ph_node = self.create_node_def(
        "PlaceholderWithDefault", placeholder_name, [a_constant_name]
    )
    self.set_attr_dtype(ph_node, "dtype", dtypes.bool)
    g_def.node.extend([ph_node])

    r_node = self.create_node_def("Relu", relu_name, [placeholder_name])
    g_def.node.extend([r_node])

    opt_graph_def = optimize_for_inference_lib.optimize_for_inference(
        g_def, [], [relu_name], dtypes.float32.as_datatype_enum
    )
    for node in opt_graph_def.node:
      self.assertNotEqual("PlaceholderWithDefault", node.op)
      if node.name == "keras_learning_phase":
        self.assertEqual(node.op, "Const")
        const_value = optimize_for_inference_lib.values_from_const(node)
        self.assertEqual(const_value, False)

  @test_util.run_deprecated_v1
  def testFoldBatchNorms(self):
    with self.cached_session() as sess:
      inputs = [1, 4, 2, 5, 3, 6, -1, -4, -2, -5, -3, -6]
      input_op = constant_op.constant(
          np.array(inputs), shape=[1, 1, 6, 2], dtype=dtypes.float32)
      weights = [1, 2, 3, 4, 0.1, 0.2, 0.3, 0.4]
      weights_op = constant_op.constant(
          np.array(weights), shape=[1, 2, 2, 2], dtype=dtypes.float32)
      conv_op = nn_ops.conv2d(
          input_op, weights_op, [1, 1, 1, 1], padding="SAME", name="conv_op")
      mean_op = constant_op.constant(
          np.array([10, 20]), shape=[2], dtype=dtypes.float32)
      variance_op = constant_op.constant(
          np.array([0.25, 0.5]), shape=[2], dtype=dtypes.float32)
      beta_op = constant_op.constant(
          np.array([0.1, 0.6]), shape=[2], dtype=dtypes.float32)
      gamma_op = constant_op.constant(
          np.array([1.0, 2.0]), shape=[2], dtype=dtypes.float32)
      test_util.set_producer_version(ops.get_default_graph(), 8)
      gen_nn_ops._batch_norm_with_global_normalization(
          conv_op,
          mean_op,
          variance_op,
          beta_op,
          gamma_op,
          0.00001,
          False,
          name="output")
      original_graph_def = sess.graph_def
      original_result = sess.run(["output:0"])
    optimized_graph_def = optimize_for_inference_lib.fold_batch_norms(
        original_graph_def)

    with self.cached_session() as sess:
      _ = importer.import_graph_def(
          optimized_graph_def, input_map={}, name="optimized")
      optimized_result = sess.run(["optimized/output:0"])

    self.assertAllClose(original_result, optimized_result)

    for node in optimized_graph_def.node:
      self.assertNotEqual("BatchNormWithGlobalNormalization", node.op)

  @test_util.run_deprecated_v1
  def testFoldFusedBatchNorms(self):
    for data_format, use_gpu, conv2d_func in [
        ("NHWC", False, nn_ops.conv2d), ("NCHW", True, nn_ops.conv2d),
        ("NHWC", False, nn_ops.depthwise_conv2d_native),
        ("NCHW", True, nn_ops.depthwise_conv2d_native)
    ]:
      with self.cached_session(use_gpu=use_gpu) as sess:
        inputs = [1, 4, 2, 5, 3, 6, -1, -4, -2, -5, -3, -6]
        input_op = constant_op.constant(
            np.array(inputs),
            shape=[1, 1, 6, 2] if data_format == "NHWC" else [1, 2, 1, 6],
            dtype=dtypes.float32)
        if conv2d_func == nn_ops.conv2d:
          weights = [1, 2, 3, 4, 0.1, 0.2, 0.3, 0.4]
          weights_op = constant_op.constant(
              np.array(weights), shape=[1, 2, 2, 2], dtype=dtypes.float32)
        else:
          weights = [1, 2, 0.3, 0.4]
          weights_op = constant_op.constant(
              np.array(weights), shape=[1, 2, 2, 1], dtype=dtypes.float32)
        conv_op = conv2d_func(
            input_op,
            weights_op, [1, 1, 1, 1],
            padding="SAME",
            data_format=data_format,
            name="conv_op")
        mean_op = constant_op.constant(
            np.array([10, 20]), shape=[2], dtype=dtypes.float32)
        variance_op = constant_op.constant(
            np.array([0.25, 0.5]), shape=[2], dtype=dtypes.float32)
        beta_op = constant_op.constant(
            np.array([0.1, 0.6]), shape=[2], dtype=dtypes.float32)
        gamma_op = constant_op.constant(
            np.array([1.0, 2.0]), shape=[2], dtype=dtypes.float32)
        ops.get_default_graph().graph_def_versions.producer = 9
        gen_nn_ops._fused_batch_norm(
            conv_op,
            gamma_op,
            beta_op,
            mean_op,
            variance_op,
            0.00001,
            is_training=False,
            data_format=data_format,
            name="output")
        original_graph_def = sess.graph_def
        original_result = sess.run(["output:0"])
      optimized_graph_def = optimize_for_inference_lib.fold_batch_norms(
          original_graph_def)

      _ = importer.import_graph_def(
          optimized_graph_def, input_map={}, name="optimized")
      optimized_result = sess.run(["optimized/output:0"])

      self.assertAllClose(
          original_result, optimized_result, rtol=1e-04, atol=1e-06)

      for node in optimized_graph_def.node:
        self.assertNotEqual("FusedBatchNorm", node.op)

  @test_util.run_deprecated_v1
  def testFoldFusedBatchNormsV3(self):
    for data_format, conv2d_func in [("NHWC", nn_ops.conv2d),
                                     ("NCHW", nn_ops.conv2d),
                                     ("NHWC", nn_ops.depthwise_conv2d_native),
                                     ("NCHW", nn_ops.depthwise_conv2d_native)]:
      with self.cached_session() as sess:
        inputs = [1, 4, 2, 5, 3, 6, -1, -4, -2, -5, -3, -6]
        input_op = constant_op.constant(
            np.array(inputs),
            shape=[1, 1, 6, 2] if data_format == "NHWC" else [1, 2, 1, 6],
            dtype=dtypes.float32)
        if conv2d_func == nn_ops.conv2d:
          weights = [1, 2, 3, 4, 0.1, 0.2, 0.3, 0.4]
          weights_op = constant_op.constant(
              np.array(weights), shape=[1, 2, 2, 2], dtype=dtypes.float32)
        else:
          weights = [1, 2, 0.3, 0.4]
          weights_op = constant_op.constant(
              np.array(weights), shape=[1, 2, 2, 1], dtype=dtypes.float32)
        mean_op = constant_op.constant(
            np.array([10, 20]), shape=[2], dtype=dtypes.float32)
        variance_op = constant_op.constant(
            np.array([0.25, 0.5]), shape=[2], dtype=dtypes.float32)
        beta_op = constant_op.constant(
            np.array([0.1, 0.6]), shape=[2], dtype=dtypes.float32)
        gamma_op = constant_op.constant(
            np.array([1.0, 2.0]), shape=[2], dtype=dtypes.float32)
        ops.get_default_graph().graph_def_versions.producer = 9
        conv_op = conv2d_func(
            input_op,
            weights_op, [1, 1, 1, 1],
            padding="SAME",
            data_format=data_format,
            name="conv_op")
        gen_nn_ops.fused_batch_norm_v3(
            conv_op,
            gamma_op,
            beta_op,
            mean_op,
            variance_op,
            0.00001,
            is_training=False,
            data_format=data_format,
            name="output")
        original_graph_def = sess.graph_def
        original_result = sess.run(["output:0"])
      optimized_graph_def = optimize_for_inference_lib.fold_batch_norms(
          original_graph_def)
    with self.cached_session() as sess:
      _ = importer.import_graph_def(
          optimized_graph_def, input_map={}, name="optimized")
      optimized_result = sess.run(["optimized/output:0"])

      self.assertAllClose(
          original_result, optimized_result, rtol=1e-04, atol=1e-06)

      for node in optimized_graph_def.node:
        self.assertNotEqual("FusedBatchNormV3", node.op)

  @test_util.run_deprecated_v1
  def testFuseResizePadAndConv(self):
    with self.cached_session() as sess:
      inputs = [1, 4, 2, 5, 3, 6, -1, -4, -2, -5, -3, -6]
      input_op = constant_op.constant(
          np.array(inputs), shape=[1, 2, 3, 2], dtype=dtypes.float32)
      resize_op = image_ops.resize_bilinear(
          input_op, [12, 4], align_corners=False)
      pad_op = array_ops.pad(resize_op, [[0, 0], [1, 1], [2, 2], [0, 0]],
                             mode="REFLECT")
      weights = [1, 2, 3, 4, 0.1, 0.2, 0.3, 0.4]
      weights_op = constant_op.constant(
          np.array(weights), shape=[1, 2, 2, 2], dtype=dtypes.float32)
      nn_ops.conv2d(
          pad_op, weights_op, [1, 1, 1, 1], padding="VALID", name="output")
      original_graph_def = sess.graph_def
      original_result = sess.run(["output:0"])
    optimized_graph_def = optimize_for_inference_lib.fuse_resize_and_conv(
        original_graph_def, ["output"])

    with self.cached_session() as sess:
      _ = importer.import_graph_def(
          optimized_graph_def, input_map={}, name="optimized")
      optimized_result = sess.run(["optimized/output:0"])

    self.assertAllClose(original_result, optimized_result)

    for node in optimized_graph_def.node:
      self.assertNotEqual("Conv2D", node.op)
      self.assertNotEqual("MirrorPad", node.op)
      self.assertNotEqual("ResizeBilinear", node.op)

  @test_util.run_deprecated_v1
  def testFuseResizeAndConv(self):
    with self.cached_session() as sess:
      inputs = [1, 4, 2, 5, 3, 6, -1, -4, -2, -5, -3, -6]
      input_op = constant_op.constant(
          np.array(inputs), shape=[1, 2, 3, 2], dtype=dtypes.float32)
      resize_op = image_ops.resize_bilinear(
          input_op, [12, 4], align_corners=False)
      weights = [1, 2, 3, 4, 0.1, 0.2, 0.3, 0.4]
      weights_op = constant_op.constant(
          np.array(weights), shape=[1, 2, 2, 2], dtype=dtypes.float32)
      nn_ops.conv2d(
          resize_op, weights_op, [1, 1, 1, 1], padding="VALID", name="output")
      original_graph_def = sess.graph_def
      original_result = sess.run(["output:0"])
    optimized_graph_def = optimize_for_inference_lib.fuse_resize_and_conv(
        original_graph_def, ["output"])

    with self.cached_session() as sess:
      _ = importer.import_graph_def(
          optimized_graph_def, input_map={}, name="optimized")
      optimized_result = sess.run(["optimized/output:0"])

    self.assertAllClose(original_result, optimized_result)

    for node in optimized_graph_def.node:
      self.assertNotEqual("Conv2D", node.op)
      self.assertNotEqual("MirrorPad", node.op)

  @test_util.run_deprecated_v1
  def testFusePadAndConv(self):
    with self.cached_session() as sess:
      inputs = [1, 4, 2, 5, 3, 6, -1, -4, -2, -5, -3, -6]
      input_op = constant_op.constant(
          np.array(inputs), shape=[1, 2, 3, 2], dtype=dtypes.float32)
      pad_op = array_ops.pad(input_op, [[0, 0], [1, 1], [2, 2], [0, 0]],
                             mode="REFLECT")
      weights = [1, 2, 3, 4, 0.1, 0.2, 0.3, 0.4]
      weights_op = constant_op.constant(
          np.array(weights), shape=[1, 2, 2, 2], dtype=dtypes.float32)
      nn_ops.conv2d(
          pad_op, weights_op, [1, 1, 1, 1], padding="VALID", name="output")
      original_graph_def = sess.graph_def
      original_result = sess.run(["output:0"])
    optimized_graph_def = optimize_for_inference_lib.fuse_resize_and_conv(
        original_graph_def, ["output"])

    with self.cached_session() as sess:
      _ = importer.import_graph_def(
          optimized_graph_def, input_map={}, name="optimized")
      optimized_result = sess.run(["optimized/output:0"])

    self.assertAllClose(original_result, optimized_result)

    for node in optimized_graph_def.node:
      self.assertNotEqual("Conv2D", node.op)
      self.assertNotEqual("ResizeBilinear", node.op)

  def count_batchnorm_relavant_ops(self, graph_def):
    """Return the count of FusedBatchNorm op and the count of primitive

    ops which may make up batchnorm computation in a given graph.
    """
    batchnorm_count = 0
    decompose_count = 0
    for node in graph_def.node:
      if node.op == "FusedBatchNorm":
        batchnorm_count += 1
      if node.op in ["Add", "Rsqrt", "Mul", "Sub"]:
        decompose_count += 1
    return batchnorm_count, decompose_count

  @test_util.run_deprecated_v1
  def create_base_for_fuse_batchnorm(self, pattern_match_mode="MATCH_ALL"):
    """Create testing graph and compute the result from original graph.

    Args:
      pattern_match_mode: A label string to indicate which batchnorm composition
        pattern to create in the resulting graph. "MATCH_ALL" - Create a graph
        matching the decomposed batchnorm pattern with full set of primitive
        ops. "MATCH_NO_GAMMA" - Create a graph matching the decomposed batchnorm
        pattern when gamma factor is 1 and multiplication with gamma is omitted.
        "MATCH_SWITCH_ORDER" - Create a graph matching the decomposed batchnorm
        pattern with a different order of inputs to the root Add node.
        "MISMATCH_PATTERN" - Create a graph with same set of primitive ops which
        makes up the decomposed batchnorm, but not matching the pattern.
        "MISMATCH_FORMAT" - Create a graph with NCHW format as input.

    Returns:
      A GraphDef as original graph to run the decomposed batchnorm test cases.
      Computation result from executing the original graph defined by GraphDef.
    """
    with self.cached_session() as sess:
      data_format = "NHWC"
      if pattern_match_mode == "MISMATCH_FORMAT":
        data_format = "NCHW"
      inputs = [1, 4, 2, 5, 3, 6, -1, -4, -2, -5, -3, -6]
      input_op = constant_op.constant(
          np.array(inputs),
          shape=[1, 1, 6, 2] if data_format == "NHWC" else [1, 2, 1, 6],
          dtype=dtypes.float32,
      )
      weights = [1, 2, 3, 4, 0.1, 0.2, 0.3, 0.4]
      weights_op = constant_op.constant(
          np.array(weights), shape=[1, 2, 2, 2], dtype=dtypes.float32
      )
      conv_op = nn_ops.conv2d(
          input_op,
          weights_op,
          [1, 1, 1, 1],
          data_format=data_format,
          padding="SAME",
          name="conv_op",
      )

      const_op_1 = None
      const_op_2 = constant_op.constant(0.00001, dtype=dtypes.float32)
      const_op_3 = None
      const_op_4 = None
      const_op_5 = None
      const_op_6 = None

      if data_format == "NHWC":
        const_op_1 = constant_op.constant(
            np.array([0.25, 0.5]), shape=[2], dtype=dtypes.float32
        )
        const_op_3 = constant_op.constant(
            np.array([10, 20]), shape=[2], dtype=dtypes.float32
        )
        const_op_4 = constant_op.constant(
            np.array([0.1, 0.6]), shape=[2], dtype=dtypes.float32
        )
        const_op_5 = constant_op.constant(
            np.array([1.0, 2.0]), shape=[2], dtype=dtypes.float32
        )
        const_op_6 = constant_op.constant(
            np.array([0.2, 0.5]), shape=[2], dtype=dtypes.float32
        )
      else:
        const_op_1 = constant_op.constant(
            np.array([0.25, 0.5, 0.6, 0.7, 0.8, 0.9]),
            shape=[6],
            dtype=dtypes.float32,
        )
        const_op_3 = constant_op.constant(
            np.array([10, 20, 30, 40, 50, 60]), shape=[6], dtype=dtypes.float32
        )
        const_op_4 = constant_op.constant(
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            shape=[6],
            dtype=dtypes.float32,
        )
        const_op_5 = constant_op.constant(
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            shape=[6],
            dtype=dtypes.float32,
        )
        const_op_6 = constant_op.constant(
            np.array([0.2, 0.4, 0.5, 0.6, 0.7, 0.8]),
            shape=[6],
            dtype=dtypes.float32,
        )

      add_op_1 = gen_math_ops.add(const_op_1, const_op_2)
      rsqrt_op = math_ops.rsqrt(add_op_1)

      variable_op = None
      if pattern_match_mode == "MATCH_NO_GAMMA":
        variable_op = rsqrt_op
      else:
        variable_op = math_ops.multiply(rsqrt_op, const_op_5)

      mul_op_1 = math_ops.multiply(conv_op, variable_op)

      mul_op_2 = None
      if pattern_match_mode == "MISMATCH_PATTERN":
        mul_op_2 = math_ops.multiply(const_op_3, const_op_6)
      else:
        mul_op_2 = math_ops.multiply(const_op_3, variable_op)

      sub_op = math_ops.subtract(const_op_4, mul_op_2)

      if pattern_match_mode == "MATCH_SWITCH_ORDER":
        gen_math_ops.add(sub_op, mul_op_1, name="output")
      else:
        gen_math_ops.add(mul_op_1, sub_op, name="output")

      test_util.set_producer_version(ops.get_default_graph(), 8)

      original_graph = sess.graph_def
      original_result = sess.run(["output:0"])

      return original_graph, original_result

  @test_util.run_deprecated_v1
  def testFuseDecomposedBatchNorm_MatchAll(self):
    original_graph_def, original_result = self.create_base_for_fuse_batchnorm(
        "MATCH_ALL"
    )

    # Test correctness of fusing individual ops to FusedBatchNorm.
    optimized_graph_def = optimize_for_inference_lib.fuse_decomposed_batch_norm(
        original_graph_def
    )

    batchnorm_count, decompose_count = self.count_batchnorm_relavant_ops(
        optimized_graph_def
    )
    self.assertEqual(batchnorm_count, 1)
    self.assertEqual(decompose_count, 0)

    with self.cached_session() as sess:
      _ = importer.import_graph_def(
          optimized_graph_def, input_map={}, name="optimized"
      )
      optimized_result = sess.run(["optimized/output:0"])

    self.assertAllClose(original_result, optimized_result)

    # Test correctness of fusing individual ops to FusedBatchNorm followed by
    # folding FusedBatchNorm.
    optimized_graph_def = optimize_for_inference_lib.fold_batch_norms(
        optimized_graph_def
    )
    for node in optimized_graph_def.node:
      self.assertNotEqual("FusedBatchNorm", node.op)

    with self.cached_session() as sess:
      _ = importer.import_graph_def(
          optimized_graph_def, input_map={}, name="optimized2"
      )
      optimized_result = sess.run(["optimized2/output:0"])

    self.assertAllClose(
        original_result, optimized_result, rtol=1e-04, atol=1e-06
    )

  @test_util.run_deprecated_v1
  def testFuseDecomposedBatchNorm_MatchNoGamma(self):
    original_graph_def, original_result = self.create_base_for_fuse_batchnorm(
        "MATCH_NO_GAMMA"
    )

    # Test correctness of fusing individual ops to FusedBatchNorm.
    optimized_graph_def = optimize_for_inference_lib.fuse_decomposed_batch_norm(
        original_graph_def
    )

    batchnorm_count, decompose_count = self.count_batchnorm_relavant_ops(
        optimized_graph_def
    )
    self.assertEqual(batchnorm_count, 1)
    self.assertEqual(decompose_count, 0)

    with self.cached_session() as sess:
      _ = importer.import_graph_def(
          optimized_graph_def, input_map={}, name="optimized"
      )
      optimized_result = sess.run(["optimized/output:0"])

    self.assertAllClose(original_result, optimized_result)

    # Test correctness of fusing individual ops to FusedBatchNorm followed by
    # folding FusedBatchNorm.
    optimized_graph_def = optimize_for_inference_lib.fold_batch_norms(
        optimized_graph_def
    )
    for node in optimized_graph_def.node:
      self.assertNotEqual("FusedBatchNorm", node.op)

    with self.cached_session() as sess:
      _ = importer.import_graph_def(
          optimized_graph_def, input_map={}, name="optimized2"
      )
      optimized_result = sess.run(["optimized2/output:0"])

    self.assertAllClose(
        original_result, optimized_result, rtol=1e-04, atol=1e-06
    )

  @test_util.run_deprecated_v1
  def testFuseDecomposedBatchNorm_MatchSwitchOrder(self):
    original_graph_def, original_result = self.create_base_for_fuse_batchnorm(
        "MATCH_SWITCH_ORDER"
    )

    # Test correctness of fusing individual ops to FusedBatchNorm.
    optimized_graph_def = optimize_for_inference_lib.fuse_decomposed_batch_norm(
        original_graph_def
    )

    batchnorm_count, decompose_count = self.count_batchnorm_relavant_ops(
        optimized_graph_def
    )
    self.assertEqual(batchnorm_count, 1)
    self.assertEqual(decompose_count, 0)

    with self.cached_session() as sess:
      _ = importer.import_graph_def(
          optimized_graph_def, input_map={}, name="optimized"
      )
      optimized_result = sess.run(["optimized/output:0"])

    self.assertAllClose(original_result, optimized_result)

    # Test correctness of fusing individual ops to FusedBatchNorm followed by
    # folding FusedBatchNorm.
    optimized_graph_def = optimize_for_inference_lib.fold_batch_norms(
        optimized_graph_def
    )
    for node in optimized_graph_def.node:
      self.assertNotEqual("FusedBatchNorm", node.op)

    with self.cached_session() as sess:
      _ = importer.import_graph_def(
          optimized_graph_def, input_map={}, name="optimized2"
      )
      optimized_result = sess.run(["optimized2/output:0"])

    self.assertAllClose(
        original_result, optimized_result, rtol=1e-04, atol=1e-06
    )

  @test_util.run_deprecated_v1
  def testFuseDecomposedBatchNorm_PatternMismatchCase(self):
    original_graph_def, original_result = self.create_base_for_fuse_batchnorm(
        "MISMATCH_PATTERN"
    )

    # Test for not to fuse ops if graph has same types of ops but pattern mismatch.
    optimized_graph_def = optimize_for_inference_lib.fuse_decomposed_batch_norm(
        original_graph_def
    )

    batchnorm_count, math_op_count = self.count_batchnorm_relavant_ops(
        optimized_graph_def
    )
    self.assertEqual(batchnorm_count, 0)
    self.assertEqual(math_op_count, 7)

    with self.cached_session() as sess:
      _ = importer.import_graph_def(
          optimized_graph_def, input_map={}, name="optimized"
      )
      optimized_result = sess.run(["optimized/output:0"])

    self.assertAllClose(original_result, optimized_result)

  @test_util.run_deprecated_v1
  def testFuseDecomposedBatchNorm_FormatUnsupportedCase(self):
    if not test_util.IsMklEnabled():
      # Non-Mkl build doesn't support NCHW format on CPU.
      self.skipTest("Skip test for non-Mkl build.")

    original_graph_def, original_result = self.create_base_for_fuse_batchnorm(
        "MISMATCH_FORMAT"
    )

    # Test for not to fuse ops if graph has same types of ops but pattern mismatch.
    optimized_graph_def = optimize_for_inference_lib.fuse_decomposed_batch_norm(
        original_graph_def
    )

    batchnorm_count, math_op_count = self.count_batchnorm_relavant_ops(
        optimized_graph_def
    )
    self.assertEqual(batchnorm_count, 0)
    self.assertEqual(math_op_count, 7)


if __name__ == "__main__":
  test.main()
