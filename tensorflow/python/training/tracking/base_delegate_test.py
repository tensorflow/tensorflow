# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for base_delegate."""
import os

from tensorflow.python.eager import test
from tensorflow.python.framework import test_util
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import save
from tensorflow.python.training.tracking import base
from tensorflow.python.training.tracking import base_delegate
from tensorflow.python.training.tracking import util


class Inner(base.Trackable):

  def __init__(self, v):
    self.v = v
    self._track_trackable(v, "v")


class Wrapper(base_delegate.DelegatingTrackableMixin, base.Trackable):

  def __init__(self, inner):
    self.inner = inner
    super(Wrapper, self).__init__(inner)

  @property
  def v(self):
    return self.inner.v


@test_util.run_all_in_graph_and_eager_modes
class BaseDelegateTest(test.TestCase):

  def test_checkpoint(self):
    a = Wrapper(Inner(variables_lib.Variable(15.0)))
    b = Wrapper(Inner(variables_lib.Variable(-15.0)))
    self.evaluate([a.v.initializer, b.v.initializer])

    test_dir = self.get_temp_dir()
    prefix = os.path.join(test_dir, "ckpt")
    ckpt = util.Checkpoint(a=a, b=b)
    prefix_tensor = ckpt.save(prefix)

    self.assertEqual([15, -15], self.evaluate([a.v, b.v]))
    self.evaluate(a.v.assign(-3))
    self.evaluate(b.v.assign(12))
    self.assertEqual([-3, 12], self.evaluate([a.v, b.v]))

    # Test that the model can be saved with the wrapper and loaded without it.
    ckpt2 = util.Checkpoint(a=a.inner, b=b.inner)
    ckpt2.restore(prefix_tensor).assert_consumed().run_restore_ops()
    self.assertEqual([15, -15], self.evaluate([a.v, b.v]))

  def test_saved_model(self):
    a = Wrapper(Inner(variables_lib.Variable(-15.0)))
    self.evaluate([a.v.initializer])
    self.assertEqual([-15], self.evaluate([a.v]))

    test_dir = self.get_temp_dir()
    saved_model_path = os.path.join(test_dir, "saved_model")
    save.save(a, saved_model_path)

    loaded = load.load(saved_model_path)
    self.evaluate([loaded.v.initializer])
    self.assertEqual([-15], self.evaluate([loaded.v]))


if __name__ == "__main__":
  test.main()
