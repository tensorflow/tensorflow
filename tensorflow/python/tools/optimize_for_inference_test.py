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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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


if __name__ == "__main__":
  test.main()
