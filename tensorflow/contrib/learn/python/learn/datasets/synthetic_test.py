# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

import six

import numpy as np
from tensorflow.python.platform import test
from tensorflow.contrib.learn.python.learn import datasets
from tensorflow.contrib.learn.python.learn.datasets import synthetic


class SyntheticTest(test.TestCase):
  """Test synthetic dataset generation"""

  def test_make_dataset(self):
    """Test if the synthetic routine wrapper complains about the name"""
    self.assertRaises(
        ValueError, datasets.make_dataset, name='_non_existing_name')

  def test_all_datasets_callable(self):
    """Test if all methods inside the `SYNTHETIC` are callable"""
    self.assertIsInstance(datasets.SYNTHETIC, dict)
    if len(datasets.SYNTHETIC) > 0:
      for name, method in six.iteritems(datasets.SYNTHETIC):
        self.assertTrue(callable(method))

  def test_circles(self):
    """Test if the circles are generated correctly

    Tests:
      - return type is `Dataset`
      - returned `data` shape is (n_samples, n_features)
      - returned `target` shape is (n_samples,)
      - set of unique classes range is [0, n_classes)

    TODO:
      - all points have the same radius, if no `noise` specified
    """
    n_samples = 100
    n_classes = 2
    circ = synthetic.circles(
        n_samples=n_samples, noise=None, n_classes=n_classes)
    self.assertIsInstance(circ, datasets.base.Dataset)
    self.assertTupleEqual(circ.data.shape, (n_samples, 2))
    self.assertTupleEqual(circ.target.shape, (n_samples,))
    self.assertSetEqual(set(circ.target), set(range(n_classes)))

  def test_circles_replicable(self):
    """Test if the data generation is replicable with a specified `seed`

    Tests:
      - return the same value if raised with the same seed
      - return different values if noise or seed is different
    """
    seed = 42
    noise = 0.1
    circ0 = synthetic.circles(
        n_samples=100, noise=noise, n_classes=2, seed=seed)
    circ1 = synthetic.circles(
        n_samples=100, noise=noise, n_classes=2, seed=seed)
    np.testing.assert_array_equal(circ0.data, circ1.data)
    np.testing.assert_array_equal(circ0.target, circ1.target)

    circ1 = synthetic.circles(
        n_samples=100, noise=noise, n_classes=2, seed=seed + 1)
    self.assertRaises(AssertionError, np.testing.assert_array_equal, circ0.data,
                      circ1.data)
    self.assertRaises(AssertionError, np.testing.assert_array_equal,
                      circ0.target, circ1.target)

    circ1 = synthetic.circles(
        n_samples=100, noise=noise / 2., n_classes=2, seed=seed)
    self.assertRaises(AssertionError, np.testing.assert_array_equal, circ0.data,
                      circ1.data)

  def test_spirals(self):
    """Test if the circles are generated correctly

    Tests:
      - if mode is unknown, ValueError is raised
      - return type is `Dataset`
      - returned `data` shape is (n_samples, n_features)
      - returned `target` shape is (n_samples,)
      - set of unique classes range is [0, n_classes)
    """
    self.assertRaises(
        ValueError, synthetic.spirals, mode='_unknown_mode_spiral_')
    n_samples = 100
    modes = ('archimedes', 'bernoulli', 'fermat')
    for mode in modes:
      spir = synthetic.spirals(n_samples=n_samples, noise=None, mode=mode)
      self.assertIsInstance(spir, datasets.base.Dataset)
      self.assertTupleEqual(spir.data.shape, (n_samples, 2))
      self.assertTupleEqual(spir.target.shape, (n_samples,))
      self.assertSetEqual(set(spir.target), set(range(2)))

  def test_spirals_replicable(self):
    """Test if the data generation is replicable with a specified `seed`

    Tests:
      - return the same value if raised with the same seed
      - return different values if noise or seed is different
    """
    seed = 42
    noise = 0.1
    modes = ('archimedes', 'bernoulli', 'fermat')
    for mode in modes:
      spir0 = synthetic.spirals(n_samples=1000, noise=noise, seed=seed)
      spir1 = synthetic.spirals(n_samples=1000, noise=noise, seed=seed)
      np.testing.assert_array_equal(spir0.data, spir1.data)
      np.testing.assert_array_equal(spir0.target, spir1.target)

      spir1 = synthetic.spirals(n_samples=1000, noise=noise, seed=seed + 1)
      self.assertRaises(AssertionError, np.testing.assert_array_equal,
                        spir0.data, spir1.data)
      self.assertRaises(AssertionError, np.testing.assert_array_equal,
                        spir0.target, spir1.target)

      spir1 = synthetic.spirals(n_samples=1000, noise=noise / 2., seed=seed)
      self.assertRaises(AssertionError, np.testing.assert_array_equal,
                        spir0.data, spir1.data)

  def test_spirals_synthetic(self):
    synthetic.spirals(3)


if __name__ == '__main__':
  test.main()
