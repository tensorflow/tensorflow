# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tracing_utils."""

from tensorflow.python.eager import def_function

from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables

from tensorflow.python.saved_model import tracing_utils
from tensorflow.python.trackable import base


class MyTrackable(base.Trackable):

  def __init__(self):
    self.a = variables.Variable(0)
    self.b = variables.Variable(1)

  def _serialize_to_tensors(self):
    return {"a": self.a, "b": self.b}

  def _restore_from_tensors(self, restored_tensors):
    return control_flow_ops.group(
        self.a.assign(restored_tensors["a"]),
        self.b.assign(restored_tensors["b"]))


class TracingUtilsTest(test.TestCase):

  def test_trace_save_and_restore(self):
    t = MyTrackable()
    save_fn, restore_fn = tracing_utils.trace_save_and_restore(t)
    self.assertDictEqual({"a": 0, "b": 1}, self.evaluate(save_fn()))
    restore_fn({"a": constant_op.constant(2), "b": constant_op.constant(3)})
    self.assertDictEqual({"a": 2, "b": 3}, self.evaluate(save_fn()))

  def test_trace_save_and_restore_concrete(self):
    t = MyTrackable()
    t._serialize_to_tensors = (def_function.function(t._serialize_to_tensors)
                               .get_concrete_function())
    restored_tensor_spec = t._serialize_to_tensors.structured_outputs
    # The wrapped tf.function doesn't matter.
    t._restore_from_tensors = (def_function.function(lambda x: x)
                               .get_concrete_function(restored_tensor_spec))

    save_fn, restore_fn = tracing_utils.trace_save_and_restore(t)
    self.assertIs(t._serialize_to_tensors, save_fn)
    self.assertIs(t._restore_from_tensors, restore_fn)


if __name__ == "__main__":
  test.main()
