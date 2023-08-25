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
"""Tests for SaveableObject compatibility."""

import os

from tensorflow.python.checkpoint import checkpoint
from tensorflow.python.checkpoint import tensor_callable
from tensorflow.python.eager import test
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import save as saved_model_save
from tensorflow.python.trackable import base


class IncrementWhenSave(base.Trackable):

  def __init__(self):
    self.read_counter = variables.Variable(0)

  def _serialize_to_tensors(self):

    def _get_and_increment_counter():
      value = self.read_counter.read_value()
      self.read_counter.assign_add(1)
      return value

    return {
        "read_counter":
            tensor_callable.Callable(_get_and_increment_counter,
                                     self.read_counter.dtype,
                                     self.read_counter.device)
    }

  def _restore_from_tensors(self, restored_tensors):
    self.read_counter.assign(restored_tensors["read_counter"])


class CallableTest(test.TestCase):

  def test_callable(self):
    trackable = IncrementWhenSave()
    ckpt = checkpoint.Checkpoint(attr=trackable)
    prefix = os.path.join(self.get_temp_dir(), "ckpt")
    save_path = ckpt.save(prefix)
    self.assertEqual(1, self.evaluate(trackable.read_counter))
    ckpt.save(prefix)
    self.assertEqual(2, self.evaluate(trackable.read_counter))

    ckpt.restore(save_path)
    self.assertEqual(0, self.evaluate(trackable.read_counter))

  def test_callable_saved_model_compatibility(self):
    trackable = IncrementWhenSave()
    trackable.read_counter.assign(15)
    save_path = os.path.join(self.get_temp_dir(), "saved_model")
    with self.assertRaisesRegex(NotImplementedError, "returns a Callable"):
      saved_model_save.save(trackable, save_path)


if __name__ == "__main__":
  test.main()
