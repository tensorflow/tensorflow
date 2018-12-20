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
"""Tests for checkpointable object SavedModel loading."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import save
from tensorflow.python.training.checkpointable import tracking


class LoadTest(test.TestCase):

  def cycle(self, obj):
    path = tempfile.mkdtemp(prefix=self.get_temp_dir())
    save.save(obj, path, signatures={})
    return load.load(path)

  def test_structure_import(self):
    root = tracking.Checkpointable()
    root.f = def_function.function(
        lambda x: 2. * x,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
    root.dep_one = tracking.Checkpointable()
    root.dep_two = tracking.Checkpointable()
    root.dep_two.dep = tracking.Checkpointable()
    root.dep_three = root.dep_two.dep
    imported = self.cycle(root)
    self.assertIs(imported.dep_three, imported.dep_two.dep)
    self.assertIsNot(imported.dep_one, imported.dep_two)
    self.assertEqual(4., imported.f(constant_op.constant(2.)).numpy())

  def test_variables(self):
    root = tracking.Checkpointable()
    root.v1 = variables.Variable(1.)
    root.v2 = variables.Variable(2.)
    root.f = def_function.function(
        lambda x: root.v2 * x,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
    imported = self.cycle(root)
    self.assertEquals(imported.v1.numpy(), 1.0)
    self.assertEquals(imported.v2.numpy(), 2.0)
    self.assertEqual(4., imported.f(constant_op.constant(2.)).numpy())

  def _make_asset(self, contents):
    filename = tempfile.mktemp(prefix=self.get_temp_dir())
    with open(filename, "w") as f:
      f.write(contents)
    return filename

  def test_assets_import(self):
    file1 = self._make_asset("contents 1")
    file2 = self._make_asset("contents 2")

    root = tracking.Checkpointable()
    root.f = def_function.function(
        lambda x: 2. * x,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
    root.asset1 = tracking.TrackableAsset(file1)
    root.asset2 = tracking.TrackableAsset(file2)

    save_dir = os.path.join(self.get_temp_dir(), "save_dir")
    save.save(root, save_dir)

    file_io.delete_file(file1)
    file_io.delete_file(file2)
    load_dir = os.path.join(self.get_temp_dir(), "load_dir")
    file_io.rename(save_dir, load_dir)

    imported = load.load(load_dir)
    with open(imported.asset1.asset_path.numpy(), "r") as f:
      self.assertEquals("contents 1", f.read())
    with open(imported.asset2.asset_path.numpy(), "r") as f:
      self.assertEquals("contents 2", f.read())

  def test_capture_assets(self):
    root = tracking.Checkpointable()
    root.vocab = tracking.TrackableAsset(self._make_asset("contents"))
    root.f = def_function.function(
        lambda: root.vocab.asset_path,
        input_signature=[])
    imported = self.cycle(root)
    origin_output = root.f().numpy()
    imported_output = imported.f().numpy()
    self.assertNotEqual(origin_output, imported_output)
    with open(imported_output, "r") as f:
      self.assertEquals("contents", f.read())

  def test_assets_dedup(self):
    vocab = self._make_asset("contents")
    root = tracking.Checkpointable()
    root.f = def_function.function(
        lambda x: 2. * x,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])

    root.asset1 = tracking.TrackableAsset(vocab)
    root.asset2 = tracking.TrackableAsset(vocab)

    imported = self.cycle(root)

    self.assertEqual(imported.asset1.asset_path.numpy(),
                     imported.asset2.asset_path.numpy())

  def test_implicit_input_signature(self):
    @def_function.function
    def func(x):
      return 2 * x

    root = tracking.Checkpointable()
    root.f = func

    # Add two traces.
    root.f(constant_op.constant(1.))
    root.f(constant_op.constant(1))

    imported = self.cycle(root)

    self.assertEqual(4., imported.f(constant_op.constant(2.)).numpy())
    self.assertEqual(14, imported.f(constant_op.constant(7)).numpy())

  def test_explicit_input_signature(self):
    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
    def func(x):
      return 2 * x

    root = tracking.Checkpointable()
    root.f = func

    imported = self.cycle(root)
    self.assertEqual(4., imported.f(constant_op.constant(2.0)).numpy())

  def test_function_with_default_bool_input(self):

    def func(x, training=False):
      if training:
        return 2 * x
      else:
        return 7

    root = tracking.Checkpointable()
    root.f = def_function.function(func)

    self.assertEqual(20, root.f(constant_op.constant(10), True).numpy())
    self.assertEqual(7, root.f(constant_op.constant(1)).numpy())
    self.assertEqual(2, root.f(constant_op.constant(1), True).numpy())

    imported = self.cycle(root)

    self.assertEqual(4, imported.f(constant_op.constant(2), True).numpy())
    self.assertEqual(7, imported.f(constant_op.constant(2)).numpy())

  def test_positional_arguments(self):
    def func(x, training=False, abc=7.1, defg=7.7):
      del abc
      if training:
        return 2 * x
      if defg == 7:
        return 6
      else:
        return 7

    root = tracking.Checkpointable()
    root.f = def_function.function(func)

    self.assertEqual(20, root.f(constant_op.constant(10), True).numpy())
    self.assertEqual(7, root.f(constant_op.constant(1)).numpy())
    self.assertEqual(2, root.f(constant_op.constant(1), True).numpy())
    self.assertEqual(6, root.f(constant_op.constant(1), defg=7.0).numpy())

    imported = self.cycle(root)

    self.assertEqual(4, imported.f(constant_op.constant(2), True).numpy())
    self.assertEqual(7, imported.f(constant_op.constant(2)).numpy())
    self.assertEqual(6, imported.f(constant_op.constant(1), defg=7.0).numpy())

  def test_member_function(self):
    class CheckpointableWithMember(tracking.Checkpointable):

      def __init__(self):
        super(CheckpointableWithMember, self).__init__()
        self._some_value = 20

      @def_function.function
      def f(self, x, training=False):
        if training:
          return 2 * x
        else:
          return 7 + self._some_value

    root = CheckpointableWithMember()

    self.assertEqual(20, root.f(constant_op.constant(10), True).numpy())
    self.assertEqual(27, root.f(constant_op.constant(1)).numpy())
    self.assertEqual(2, root.f(constant_op.constant(1), True).numpy())

    imported = self.cycle(root)

    self.assertEqual(4, imported.f(constant_op.constant(2), True).numpy())
    self.assertEqual(27, imported.f(constant_op.constant(2)).numpy())


if __name__ == "__main__":
  test.main()
