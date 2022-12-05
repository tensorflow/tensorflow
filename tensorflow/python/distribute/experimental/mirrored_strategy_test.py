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
"""Test for MirroredStrategy backed by DTensor API."""

from absl.testing import parameterized
import numpy as np

from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import layout
from tensorflow.dtensor.python.tests import test_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute.experimental import mirrored_strategy
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import variables


class VariableCreationTest(test_util.DTensorBaseTest):

  def setUp(self):
    super().setUp()
    global_ids = test_util.create_device_ids_array((1, 2))
    local_ids = np.ravel(global_ids).tolist()
    mesh_dict = {
        device: layout.Mesh(['x', 'y'], global_ids, local_ids,
                            test_util.create_device_list((1, 2), device))
        for device in ['TPU', 'GPU', 'CPU']
    }
    self.mesh = self.configTestMesh(mesh_dict)

  @parameterized.named_parameters([
      ('py_floats', lambda: [1.0, 2.0], True),
      ('np_floats', lambda: np.array([1.0, 2.0]), True),
      ('tf_const', lambda: constant_op.constant([1.0, 2.0]), True),
      ('py_floats_callable', lambda: [1.0, 2.0], False),
      ('np_floats_callable', lambda: np.array([1.0, 2.0]), False),
      ('tf_const_callable', lambda: constant_op.constant([1.0, 2.0]), False),
  ])
  def test_variable_creation(self, init_value, convert_callable):
    if convert_callable:
      init_value = init_value()
    strategy = mirrored_strategy.MirroredStrategy(self.mesh)
    with strategy.scope():
      v = variables.Variable(init_value)

    self.assertIsInstance(v, d_variable.DVariable)
    self.assertIsNotNone(v.layout)
    self.assertEqual(v.layout, layout.Layout.replicated(self.mesh, rank=1))

  def test_mesh(self):
    strategy = mirrored_strategy.MirroredStrategy(self.mesh)
    self.assertEqual(strategy._mesh, self.mesh)

  def test_strategy_extension(self):
    strategy = mirrored_strategy.MirroredStrategy(self.mesh)
    self.assertIsInstance(strategy.extended, distribute_lib.StrategyExtendedV2)

  # TODO(scottzhu): Add more test coverage for all the strategy extension method
  # or raise ValueError if certain method is not supported.


if __name__ == '__main__':
  test.main()
