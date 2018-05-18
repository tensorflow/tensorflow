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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

from tensorflow.contrib.checkpoint.python import visualize

from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.keras._impl.keras.engine import training
from tensorflow.python.keras._impl.keras.layers import core
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import adam
from tensorflow.python.training.checkpointable import util as checkpointable_utils

try:
  import pydot  # pylint: disable=g-import-not-at-top
except ImportError:
  pydot = None


class MyModel(training.Model):
  """A concrete Model for testing."""

  def __init__(self):
    super(MyModel, self).__init__()
    self._named_dense = core.Dense(1, use_bias=True)
    self._second = core.Dense(1, use_bias=False)

  def call(self, values):
    ret = self._second(self._named_dense(values))
    return ret


class DotGraphTests(test.TestCase):

  def testMakeDotGraph(self):
    with context.eager_mode():
      input_value = constant_op.constant([[3.]])
      model = MyModel()
      optimizer = adam.AdamOptimizer(0.001)
      optimizer_step = resource_variable_ops.ResourceVariable(12)
      save_checkpoint = checkpointable_utils.Checkpoint(
          optimizer=optimizer, model=model, optimizer_step=optimizer_step)
      optimizer.minimize(functools.partial(model, input_value))
      checkpoint_directory = self.get_temp_dir()
      checkpoint_prefix = os.path.join(checkpoint_directory, 'ckpt')
      save_path = save_checkpoint.save(checkpoint_prefix)
      prefix = save_checkpoint.save(save_path)

    dot_graph_string = visualize.dot_graph_from_checkpoint(prefix)

    # The remainder of this test is more-or-less optional since it's so
    # dependent on pydot/platform/Python versions.
    if pydot is None:
      self.skipTest('pydot is required for the remainder of this test.')
    try:
      parsed, = pydot.graph_from_dot_data(dot_graph_string)
    except NameError as e:
      if "name 'dot_parser' is not defined" in str(e):
        self.skipTest("pydot isn't working")
      else:
        raise
    # Check that the graph isn't completely trivial
    self.assertEqual(
        '"model"',
        parsed.obj_dict['edges'][('N_0', 'N_1')][0]['attributes']['label'])
    image_path = os.path.join(self.get_temp_dir(), 'saved.svg')
    try:
      parsed.write_svg(image_path)
    except Exception as e:  # pylint: disable=broad-except
      # For some reason PyDot's "dot not available" error is an Exception, not
      # something more specific.
      if '"dot" not found in path' in str(e):
        self.skipTest("pydot won't save SVGs (dot not available)")
      else:
        raise

if __name__ == '__main__':
  test.main()
