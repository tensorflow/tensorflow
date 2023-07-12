# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Tests serialization of Variable wrapper class using the Trackable API.

A similar implementation is used in Keras Variable, and this test exists to
ensure the use case continues to work.
"""

import os

from tensorflow.python.checkpoint import checkpoint
from tensorflow.python.eager import test
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import save
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import base


class VariableWrapper(base.Trackable):
  _should_act_as_resource_variable = True

  def __init__(self, value):
    self.value = variables.Variable(value)

  @property
  def _shared_name(self):
    return self.value._shared_name

  def _serialize_to_tensors(self):
    return self.value._serialize_to_tensors()

  def _restore_from_tensors(self, restored_tensors):
    return self.value._restore_from_tensors(restored_tensors)

  def _export_to_saved_model_graph(
      self, object_map, tensor_map, options, **kwargs
  ):
    resource_list = self.value._export_to_saved_model_graph(  # pylint:disable=protected-access
        object_map, tensor_map, options, **kwargs
    )
    object_map[self] = VariableWrapper(object_map[self.value])
    return resource_list

  def _write_object_proto(self, proto, options):
    return self.value._write_object_proto(proto, options)


class VariableWrapperTest(test.TestCase):

  def test_checkpoint(self):
    v = VariableWrapper(5.0)
    root = autotrackable.AutoTrackable()
    root.v = v

    save_prefix = os.path.join(self.get_temp_dir(), "checkpoint")
    ckpt = checkpoint.Checkpoint(v=v)
    save_path = ckpt.save(save_prefix)

    v.value.assign(100)
    ckpt.restore(save_path)
    self.assertEqual(5, v.value.numpy())

  def test_export(self):
    v = VariableWrapper(15.0)
    root = autotrackable.AutoTrackable()
    root.v = v

    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(root, save_dir)
    loaded = load.load(save_dir)
    self.assertTrue(resource_variable_ops.is_resource_variable(loaded.v))
    self.assertEqual(15, loaded.v.numpy())


if __name__ == "__main__":
  test.main()
