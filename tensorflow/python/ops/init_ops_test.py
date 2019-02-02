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
"""Tests for initializers in init_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape as tensor_shape_lib
from tensorflow.python.framework import test_util
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class InitializersTest(test.TestCase):

  def _runner(self,
              init,
              shape,
              target_mean=None,
              target_std=None,
              target_max=None,
              target_min=None):
    output = self.evaluate(init(shape))
    self.assertEqual(output.shape, shape)
    lim = 3e-2
    if target_std is not None:
      self.assertGreater(lim, abs(output.std() - target_std))
    if target_mean is not None:
      self.assertGreater(lim, abs(output.mean() - target_mean))
    if target_max is not None:
      self.assertGreater(lim, abs(output.max() - target_max))
    if target_min is not None:
      self.assertGreater(lim, abs(output.min() - target_min))

  def test_uniform(self):
    shape = (9, 6, 99)
    with self.cached_session():
      for tensor_shape in [shape, tensor_shape_lib.TensorShape(shape)]:
        self._runner(
            init_ops.RandomUniform(minval=-1, maxval=1, seed=124),
            tensor_shape,
            target_mean=0.,
            target_max=1,
            target_min=-1)

  def test_normal(self):
    shape = (8, 12, 99)
    with self.cached_session():
      for tensor_shape in [shape, tensor_shape_lib.TensorShape(shape)]:
        self._runner(
            init_ops.RandomNormal(mean=0, stddev=1, seed=153),
            tensor_shape,
            target_mean=0.,
            target_std=1)

  def test_truncated_normal(self):
    shape = (12, 99, 7)
    with self.cached_session():
      for tensor_shape in [shape, tensor_shape_lib.TensorShape(shape)]:
        self._runner(
            init_ops.TruncatedNormal(mean=0, stddev=1, seed=126),
            tensor_shape,
            target_mean=0.,
            target_max=2,
            target_min=-2)

  def test_constant(self):
    shape = (5, 6, 4)
    with self.cached_session():
      for tensor_shape in [shape, tensor_shape_lib.TensorShape(shape)]:
        self._runner(
            init_ops.Constant(2),
            tensor_shape,
            target_mean=2,
            target_max=2,
            target_min=2)

  def test_lecun_uniform(self):
    shape = (5, 6, 4, 2)
    with self.cached_session():
      for tensor_shape in [shape, tensor_shape_lib.TensorShape(shape)]:
        fan_in, _ = init_ops._compute_fans(tensor_shape)
        std = np.sqrt(1. / fan_in)
        self._runner(
            init_ops.lecun_uniform(seed=123),
            tensor_shape,
            target_mean=0.,
            target_std=std)

  def test_glorot_uniform_initializer(self):
    shape = (5, 6, 4, 2)
    with self.cached_session():
      for tensor_shape in [shape, tensor_shape_lib.TensorShape(shape)]:
        fan_in, fan_out = init_ops._compute_fans(tensor_shape)
        std = np.sqrt(2. / (fan_in + fan_out))
        self._runner(
            init_ops.glorot_uniform_initializer(seed=123),
            tensor_shape,
            target_mean=0.,
            target_std=std)

  def test_he_uniform(self):
    shape = (5, 6, 4, 2)
    with self.cached_session():
      for tensor_shape in [shape, tensor_shape_lib.TensorShape(shape)]:
        fan_in, _ = init_ops._compute_fans(tensor_shape)
        std = np.sqrt(2. / fan_in)
        self._runner(
            init_ops.he_uniform(seed=123),
            tensor_shape,
            target_mean=0.,
            target_std=std)

  def test_lecun_normal(self):
    shape = (5, 6, 4, 2)
    with self.cached_session():
      for tensor_shape in [shape, tensor_shape_lib.TensorShape(shape)]:
        fan_in, _ = init_ops._compute_fans(tensor_shape)
        std = np.sqrt(1. / fan_in)
        self._runner(
            init_ops.lecun_normal(seed=123),
            tensor_shape,
            target_mean=0.,
            target_std=std)

  def test_glorot_normal_initializer(self):
    shape = (5, 6, 4, 2)
    with self.cached_session():
      for tensor_shape in [shape, tensor_shape_lib.TensorShape(shape)]:
        fan_in, fan_out = init_ops._compute_fans(tensor_shape)
        std = np.sqrt(2. / (fan_in + fan_out))
        self._runner(
            init_ops.glorot_normal_initializer(seed=123),
            tensor_shape,
            target_mean=0.,
            target_std=std)

  def test_he_normal(self):
    shape = (5, 6, 4, 2)
    with self.cached_session():
      for tensor_shape in [shape, tensor_shape_lib.TensorShape(shape)]:
        fan_in, _ = init_ops._compute_fans(tensor_shape)
        std = np.sqrt(2. / fan_in)
        self._runner(
            init_ops.he_normal(seed=123),
            tensor_shape,
            target_mean=0.,
            target_std=std)

  def test_Orthogonal(self):
    shape = (20, 20)
    with self.cached_session():
      for tensor_shape in [shape, tensor_shape_lib.TensorShape(shape)]:
        self._runner(init_ops.Orthogonal(seed=123),
                     tensor_shape,
                     target_mean=0.)

  def testVariablePlacementWithOrthogonalInitializer(self):
    if not context.context().num_gpus():
      self.skipTest('No devices other than CPUs found')
    with ops.Graph().as_default() as g:
      with ops.device('gpu:0'):
        variable_scope.get_variable(
            name='v', shape=[8, 2], initializer=init_ops.Orthogonal)
        variable_scope.get_variable(
            name='w', shape=[8, 2], initializer=init_ops.RandomNormal)
      run_metadata = config_pb2.RunMetadata()
      run_options = config_pb2.RunOptions(
          trace_level=config_pb2.RunOptions.FULL_TRACE)
      config = config_pb2.ConfigProto(
          allow_soft_placement=False, log_device_placement=True)

      # Note: allow_soft_placement=False will fail whenever we cannot satisfy
      # the colocation constraints.
      with session.Session(config=config, graph=g) as sess:
        sess.run(
            variables.global_variables_initializer(),
            options=run_options,
            run_metadata=run_metadata)

  def test_eager_orthogonal_gpu(self):
    if not context.context().num_gpus():
      self.skipTest('No devices other than CPUs found')
    with context.eager_mode():
      v = variable_scope.get_variable(
          name='v', shape=[8, 2], initializer=init_ops.Orthogonal)
      w = variable_scope.get_variable(
          name='w', shape=[8, 2], initializer=init_ops.RandomNormal)
      self.assertTrue('GPU' in v.handle.device)
      self.assertTrue('GPU' in w.handle.device)

  def test_Identity(self):
    with self.cached_session():
      shape = (3, 4, 5)
      for tensor_shape in [shape, tensor_shape_lib.TensorShape(shape)]:
        with self.assertRaises(ValueError):
          self._runner(
              init_ops.Identity(),
              tensor_shape,
              target_mean=1. / int(tensor_shape[0]),
              target_max=1.)

      shape = (3, 3)
      for tensor_shape in [shape, tensor_shape_lib.TensorShape(shape)]:
        self._runner(
            init_ops.Identity(),
            tensor_shape,
            target_mean=1. / int(tensor_shape[0]),
            target_max=1.)

  def test_Zeros(self):
    shape = (4, 5)
    with self.cached_session():
      for tensor_shape in [shape, tensor_shape_lib.TensorShape(shape)]:
        self._runner(
            init_ops.Zeros(), tensor_shape, target_mean=0., target_max=0.)

  def test_Ones(self):
    shape = (4, 5)
    with self.cached_session():
      for tensor_shape in [shape, tensor_shape_lib.TensorShape(shape)]:
        self._runner(init_ops.Ones(), tensor_shape,
                     target_mean=1., target_max=1.)


if __name__ == '__main__':
  test.main()
