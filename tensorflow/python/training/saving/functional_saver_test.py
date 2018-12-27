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
# =============================================================================
"""Tests for the functional saver."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.eager import test
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training.saving import functional_saver
from tensorflow.python.training.saving import saveable_object_util


class SaverTest(test.TestCase):

  def test_resource_variable(self):
    v1 = resource_variable_ops.ResourceVariable(2.)
    saver = functional_saver.Saver(
        saveable_object_util.saveable_objects_for_op(v1, "x"))
    prefix = os.path.join(self.get_temp_dir(), "ckpt")
    save_path = saver.save(constant_op.constant(prefix))
    v1.assign(1.)
    saver.restore(save_path)
    self.assertEqual(2., self.evaluate(v1))

    v2 = resource_variable_ops.ResourceVariable(3.)
    second_saver = functional_saver.Saver(
        saveable_object_util.saveable_objects_for_op(v2, "x"))
    second_saver.restore(save_path)
    self.assertEqual(2., self.evaluate(v2))

  def test_to_proto(self):
    v1 = resource_variable_ops.ResourceVariable(2.)
    saver = functional_saver.Saver(
        saveable_object_util.saveable_objects_for_op(v1, "x"))
    prefix = os.path.join(self.get_temp_dir(), "ckpt")

    proto_accumulator = []
    wrapped = wrap_function.wrap_function(
        lambda: proto_accumulator.append(saver.to_proto()), signature=())
    self.assertEqual(1, len(proto_accumulator))
    proto = proto_accumulator[0]
    save = wrapped.prune(
        feeds=wrapped.graph.get_tensor_by_name(proto.filename_tensor_name),
        fetches=wrapped.graph.get_tensor_by_name(proto.save_tensor_name))
    restore = wrapped.prune(
        feeds=wrapped.graph.get_tensor_by_name(proto.filename_tensor_name),
        fetches=wrapped.graph.get_operation_by_name(proto.restore_op_name))
    save_path = save(constant_op.constant(prefix))
    v1.assign(1.)
    restore(constant_op.constant(save_path))
    self.assertEqual(2., self.evaluate(v1))

    v2 = resource_variable_ops.ResourceVariable(3.)
    second_saver = functional_saver.Saver(
        saveable_object_util.saveable_objects_for_op(v2, "x"))
    second_saver.restore(save_path)
    self.assertEqual(2., self.evaluate(v2))


if __name__ == "__main__":
  test.main()
