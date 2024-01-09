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
"""Tests for tensorflow.python.ops.special_math_ops on WeakTensor."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import ops

from tensorflow.python.framework import test_util
from tensorflow.python.framework.weak_tensor import WeakTensor
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import weak_tensor_ops  # pylint: disable=unused-import
from tensorflow.python.ops import weak_tensor_test_util
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging

_get_weak_tensor = weak_tensor_test_util.get_weak_tensor


@test_util.run_all_in_graph_and_eager_modes
class DawsnTest(test.TestCase, parameterized.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_dawsn_boundary(self):
    self.assertAllClose(0., special_math_ops.dawsn(0.))
    self.assertTrue(np.isnan(self.evaluate(special_math_ops.dawsn(np.nan))))

  @parameterized.parameters(np.float32, np.float64)
  def test_dawsn_odd(self, dtype):
    x = _get_weak_tensor(
        np.random.uniform(-100.0, 100.0, size=int(1e4)).astype(dtype)
    )
    y = special_math_ops.dawsn(x)
    neg_y = -special_math_ops.dawsn(-x)

    self.assertIsInstance(y, WeakTensor)
    self.assertIsInstance(neg_y, WeakTensor)
    self.assertAllClose(self.evaluate(y), self.evaluate(neg_y))

  @parameterized.parameters(np.float32, np.float64)
  def test_dawsn_small(self, dtype):
    x = np.random.uniform(-1., 1., size=int(1e4)).astype(dtype)
    x_wt = _get_weak_tensor(x)
    y_wt = special_math_ops.dawsn(x_wt)
    self.assertIsInstance(y_wt, WeakTensor)

    try:
      from scipy import special  # pylint: disable=g-import-not-at-top

      self.assertAllClose(special.dawsn(x), self.evaluate(y_wt))
    except ImportError as e:
      tf_logging.warn('Cannot test special functions: %s' % str(e))

  @parameterized.parameters(np.float32, np.float64)
  def test_dawsn_larger(self, dtype):
    x = np.random.uniform(1., 100., size=int(1e4)).astype(dtype)
    x_wt = _get_weak_tensor(x)
    y_wt = special_math_ops.dawsn(x_wt)
    self.assertIsInstance(y_wt, WeakTensor)

    try:
      from scipy import special  # pylint: disable=g-import-not-at-top

      self.assertAllClose(special.dawsn(x), y_wt)
    except ImportError as e:
      tf_logging.warn('Cannot test special functions: %s' % str(e))

  def test_dawsn_gradient(self):
    inputs = [_get_weak_tensor(np.random.uniform(-50.0, 50.0, size=int(1e2)))]
    analytical, numerical = gradient_checker_v2.compute_gradient(
        special_math_ops.dawsn, inputs)
    self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 1e-4)


@test_util.run_all_in_graph_and_eager_modes
class ExpintTest(test.TestCase, parameterized.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_expint_boundary(self):
    self.assertAllClose(-np.inf, special_math_ops.expint(0.))
    self.assertTrue(np.isnan(self.evaluate(special_math_ops.expint(np.nan))))
    # Check that the domain of definition is [0, inf)
    self.assertTrue(
        np.all(
            np.isnan(
                self.evaluate(
                    special_math_ops.expint(
                        np.random.uniform(-20., -1., size=int(1e3)))))))

  @parameterized.parameters(np.float32, np.float64)
  def test_expint_small(self, dtype):
    x = np.random.uniform(0., 1., size=int(1e4)).astype(dtype)
    x_wt = _get_weak_tensor(x)
    y_wt = special_math_ops.expint(x_wt)
    self.assertIsInstance(y_wt, WeakTensor)

    try:
      from scipy import special  # pylint: disable=g-import-not-at-top

      self.assertAllClose(
          special.expi(x), self.evaluate(special_math_ops.expint(x_wt))
      )
    except ImportError as e:
      tf_logging.warn('Cannot test special functions: %s' % str(e))

  @parameterized.parameters(np.float32, np.float64)
  def test_expint_larger(self, dtype):
    x = np.random.uniform(1., 50., size=int(1e4)).astype(dtype)
    x_wt = _get_weak_tensor(x)
    try:
      from scipy import special  # pylint: disable=g-import-not-at-top

      self.assertAllClose(
          special.expi(x), self.evaluate(special_math_ops.expint(x_wt))
      )
    except ImportError as e:
      tf_logging.warn('Cannot test special functions: %s' % str(e))

  def test_expint_gradient(self):
    inputs = [_get_weak_tensor(np.random.uniform(1.0, 10.0, size=int(1e2)))]
    analytical, numerical = gradient_checker_v2.compute_gradient(
        special_math_ops.expint, inputs)
    self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 5e-3)


@test_util.run_all_in_graph_and_eager_modes
class FresnelCosTest(test.TestCase, parameterized.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_fresnel_cos_boundary(self):
    self.assertAllClose(0., special_math_ops.fresnel_cos(0.))
    self.assertTrue(
        np.isnan(self.evaluate(special_math_ops.fresnel_cos(np.nan))))

  @parameterized.parameters(np.float32, np.float64)
  def test_fresnel_cos_odd(self, dtype):
    x = _get_weak_tensor(
        np.random.uniform(-100.0, 100.0, size=int(1e4)).astype(dtype)
    )
    y = special_math_ops.fresnel_cos(x)
    neg_y = -special_math_ops.fresnel_cos(-x)

    self.assertIsInstance(y, WeakTensor)
    self.assertIsInstance(neg_y, WeakTensor)
    self.assertAllClose(self.evaluate(y), self.evaluate(neg_y))

  @parameterized.parameters(np.float32, np.float64)
  def test_fresnel_cos_small(self, dtype):
    x = np.random.uniform(0., 1., size=int(1e4)).astype(dtype)
    x_wt = _get_weak_tensor(x)
    y_wt = special_math_ops.fresnel_cos(x_wt)
    self.assertIsInstance(y_wt, WeakTensor)

    try:
      from scipy import special  # pylint: disable=g-import-not-at-top

      self.assertAllClose(special.fresnel(x)[1], self.evaluate(y_wt))
    except ImportError as e:
      tf_logging.warn('Cannot test special functions: %s' % str(e))

  @parameterized.parameters(np.float32, np.float64)
  def test_fresnel_cos_larger(self, dtype):
    x = np.random.uniform(1., 100., size=int(1e4)).astype(dtype)
    x_wt = _get_weak_tensor(x)
    try:
      from scipy import special  # pylint: disable=g-import-not-at-top

      self.assertAllClose(
          special.fresnel(x)[1],
          self.evaluate(special_math_ops.fresnel_cos(x_wt)),
          rtol=1e-5,
      )
    except ImportError as e:
      tf_logging.warn('Cannot test special functions: %s' % str(e))

  def test_fresnel_cos_gradient(self):
    inputs = [_get_weak_tensor(np.random.uniform(1.0, 50.0, size=int(1e2)))]
    analytical, numerical = gradient_checker_v2.compute_gradient(
        special_math_ops.fresnel_cos, inputs)
    self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 5e-3)


@test_util.run_all_in_graph_and_eager_modes
class FresnelSinTest(test.TestCase, parameterized.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_fresnel_sin_boundary(self):
    self.assertAllClose(0., special_math_ops.fresnel_sin(0.))
    self.assertTrue(
        np.isnan(self.evaluate(special_math_ops.fresnel_sin(np.nan))))

  @parameterized.parameters(np.float32, np.float64)
  def test_fresnel_sin_odd(self, dtype):
    x = _get_weak_tensor(
        np.random.uniform(-100.0, 100.0, size=int(1e4)).astype(dtype)
    )
    y = special_math_ops.fresnel_sin(x)
    neg_y = -special_math_ops.fresnel_sin(-x)

    self.assertIsInstance(y, WeakTensor)
    self.assertIsInstance(neg_y, WeakTensor)
    self.assertAllClose(y, neg_y)

  @parameterized.parameters(np.float32, np.float64)
  def test_fresnel_sin_small(self, dtype):
    x = np.random.uniform(0., 1., size=int(1e4)).astype(dtype)
    x_wt = _get_weak_tensor(x)
    try:
      from scipy import special  # pylint: disable=g-import-not-at-top

      self.assertAllClose(
          special.fresnel(x)[0],
          self.evaluate(special_math_ops.fresnel_sin(x_wt)),
      )
    except ImportError as e:
      tf_logging.warn('Cannot test special functions: %s' % str(e))

  @parameterized.parameters(np.float32, np.float64)
  def test_fresnel_sin_larger(self, dtype):
    x = np.random.uniform(1., 100., size=int(1e4)).astype(dtype)
    x_wt = _get_weak_tensor(x)
    try:
      from scipy import special  # pylint: disable=g-import-not-at-top

      self.assertAllClose(
          special.fresnel(x)[0],
          self.evaluate(special_math_ops.fresnel_sin(x_wt)),
          rtol=1e-5,
      )
    except ImportError as e:
      tf_logging.warn('Cannot test special functions: %s' % str(e))

  def test_fresnel_sin_gradient(self):
    inputs = [_get_weak_tensor(np.random.uniform(1.0, 50.0, size=int(1e2)))]
    analytical, numerical = gradient_checker_v2.compute_gradient(
        special_math_ops.fresnel_sin, inputs)
    self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 5e-3)


@test_util.run_all_in_graph_and_eager_modes
class SpenceTest(test.TestCase, parameterized.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_spence_boundary(self):
    self.assertAllClose(np.pi**2 / 6., special_math_ops.spence(0.))
    self.assertAllClose(0., special_math_ops.spence(1.))
    self.assertTrue(np.isnan(self.evaluate(special_math_ops.spence(np.nan))))
    # Check that the domain of definition is [0, inf)
    self.assertTrue(
        np.all(
            np.isnan(
                self.evaluate(
                    special_math_ops.spence(
                        np.random.uniform(-20., -1., size=int(1e3)))))))

  @parameterized.parameters(np.float32, np.float64)
  def test_spence_small(self, dtype):
    x = np.random.uniform(0., 1., size=int(1e4)).astype(dtype)
    x_wt = _get_weak_tensor(x)
    y_wt = special_math_ops.spence(x_wt)
    self.assertIsInstance(y_wt, WeakTensor)

    try:
      from scipy import special  # pylint: disable=g-import-not-at-top

      self.assertAllClose(special.spence(x), self.evaluate(y_wt))
    except ImportError as e:
      tf_logging.warn('Cannot test special functions: %s' % str(e))

  @parameterized.parameters(np.float32, np.float64)
  def test_spence_larger(self, dtype):
    x = np.random.uniform(1., 100., size=int(1e4)).astype(dtype)
    x_wt = _get_weak_tensor(x)
    try:
      from scipy import special  # pylint: disable=g-import-not-at-top

      self.assertAllClose(
          special.spence(x), self.evaluate(special_math_ops.spence(x_wt))
      )
    except ImportError as e:
      tf_logging.warn('Cannot test special functions: %s' % str(e))

  def test_spence_gradient(self):
    inputs = [_get_weak_tensor(np.random.uniform(1.0, 50.0, size=int(1e2)))]
    analytical, numerical = gradient_checker_v2.compute_gradient(
        special_math_ops.spence, inputs)
    self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 1e-4)

  def test_spence_gradient_at_one(self):
    analytical, _ = gradient_checker_v2.compute_gradient(
        special_math_ops.spence, [1.])
    self.assertAllClose([[[-1.]]], analytical)


@test_util.run_all_in_graph_and_eager_modes
class BesselTest(test.TestCase, parameterized.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_besseli_boundary(self):
    self.assertAllClose(1., special_math_ops.bessel_i0(0.))
    self.assertAllClose(1., special_math_ops.bessel_i0e(0.))
    self.assertAllClose(0., special_math_ops.bessel_i1(0.))
    self.assertAllClose(0., special_math_ops.bessel_i1e(0.))
    self.assertTrue(np.isnan(self.evaluate(special_math_ops.bessel_i0(np.nan))))
    self.assertTrue(
        np.isnan(self.evaluate(special_math_ops.bessel_i0e(np.nan))))
    self.assertTrue(np.isnan(self.evaluate(special_math_ops.bessel_i1(np.nan))))
    self.assertTrue(
        np.isnan(self.evaluate(special_math_ops.bessel_i1e(np.nan))))

  @test_util.run_in_graph_and_eager_modes
  def test_besselj_boundary(self):
    self.assertAllClose(1., special_math_ops.bessel_j0(0.))
    self.assertAllClose(0., special_math_ops.bessel_j1(0.))
    self.assertTrue(np.isnan(self.evaluate(special_math_ops.bessel_j0(np.nan))))
    self.assertTrue(np.isnan(self.evaluate(special_math_ops.bessel_j1(np.nan))))

  @test_util.run_in_graph_and_eager_modes
  def test_besselk_boundary(self):
    self.assertTrue(np.isinf(self.evaluate(special_math_ops.bessel_k0(0.))))
    self.assertTrue(np.isinf(self.evaluate(special_math_ops.bessel_k0e(0.))))
    self.assertTrue(np.isinf(self.evaluate(special_math_ops.bessel_k1(0.))))
    self.assertTrue(np.isinf(self.evaluate(special_math_ops.bessel_k1e(0.))))
    self.assertTrue(np.isnan(self.evaluate(special_math_ops.bessel_k0(np.nan))))
    self.assertTrue(
        np.isnan(self.evaluate(special_math_ops.bessel_k0e(np.nan))))
    self.assertTrue(np.isnan(self.evaluate(special_math_ops.bessel_k1(np.nan))))
    self.assertTrue(
        np.isnan(self.evaluate(special_math_ops.bessel_k1e(np.nan))))

  @parameterized.parameters(np.float32, np.float64)
  def test_i0j0_even(self, dtype):
    x = _get_weak_tensor(
        np.random.uniform(-100.0, 100.0, size=int(1e4)).astype(dtype)
    )
    self.assertAllClose(
        self.evaluate(special_math_ops.bessel_i0(x)),
        self.evaluate(special_math_ops.bessel_i0(-x)))

    self.assertAllClose(
        self.evaluate(special_math_ops.bessel_i0e(x)),
        self.evaluate(special_math_ops.bessel_i0e(-x)))

    self.assertAllClose(
        self.evaluate(special_math_ops.bessel_j0(x)),
        self.evaluate(special_math_ops.bessel_j0(-x)))

  @parameterized.parameters(np.float32, np.float64)
  def test_i1j1_odd(self, dtype):
    x = _get_weak_tensor(
        np.random.uniform(-100.0, 100.0, size=int(1e4)).astype(dtype)
    )
    self.assertAllClose(
        self.evaluate(special_math_ops.bessel_i1(x)),
        self.evaluate(-special_math_ops.bessel_i1(-x)))

    self.assertAllClose(
        self.evaluate(special_math_ops.bessel_i1e(x)),
        self.evaluate(-special_math_ops.bessel_i1e(-x)))

    self.assertAllClose(
        self.evaluate(special_math_ops.bessel_j1(x)),
        self.evaluate(-special_math_ops.bessel_j1(-x)))

  @parameterized.parameters(np.float32, np.float64)
  def test_besseli_small(self, dtype):
    x = np.random.uniform(-1.0, 1.0, size=int(1e4)).astype(dtype)
    x_wt = _get_weak_tensor(x)
    try:
      from scipy import special  # pylint: disable=g-import-not-at-top

      self.assertAllClose(
          special.i0(x), self.evaluate(special_math_ops.bessel_i0(x_wt))
      )
      self.assertAllClose(
          special.i1(x), self.evaluate(special_math_ops.bessel_i1(x_wt))
      )
      self.assertAllClose(
          special.i0e(x), self.evaluate(special_math_ops.bessel_i0e(x_wt))
      )
      self.assertAllClose(
          special.i1e(x), self.evaluate(special_math_ops.bessel_i1e(x_wt))
      )
    except ImportError as e:
      tf_logging.warn('Cannot test special functions: %s' % str(e))

  @parameterized.parameters(np.float32, np.float64)
  def test_besselj_small(self, dtype):
    x = np.random.uniform(-1.0, 1.0, size=int(1e4)).astype(dtype)
    x_wt = _get_weak_tensor(x)

    try:
      from scipy import special  # pylint: disable=g-import-not-at-top

      self.assertAllClose(
          special.j0(x), self.evaluate(special_math_ops.bessel_j0(x_wt))
      )
      self.assertAllClose(
          special.j1(x), self.evaluate(special_math_ops.bessel_j1(x_wt))
      )
    except ImportError as e:
      tf_logging.warn('Cannot test special functions: %s' % str(e))

  @parameterized.parameters(np.float32, np.float64)
  def test_besselk_small(self, dtype):
    x = np.random.uniform(np.finfo(dtype).eps, 1.0, size=int(1e4)).astype(dtype)
    x_wt = _get_weak_tensor(x)

    try:
      from scipy import special  # pylint: disable=g-import-not-at-top

      self.assertAllClose(
          special.k0(x), self.evaluate(special_math_ops.bessel_k0(x_wt))
      )
      self.assertAllClose(
          special.k0e(x), self.evaluate(special_math_ops.bessel_k0e(x_wt))
      )
      self.assertAllClose(
          special.k1(x), self.evaluate(special_math_ops.bessel_k1(x_wt))
      )
      self.assertAllClose(
          special.k1e(x), self.evaluate(special_math_ops.bessel_k1e(x_wt))
      )
    except ImportError as e:
      tf_logging.warn('Cannot test special functions: %s' % str(e))

  @parameterized.parameters(np.float32, np.float64)
  def test_bessely_small(self, dtype):
    x = np.random.uniform(np.finfo(dtype).eps, 1.0, size=int(1e4)).astype(dtype)
    x_wt = _get_weak_tensor(x)

    try:
      from scipy import special  # pylint: disable=g-import-not-at-top

      self.assertAllClose(
          special.y0(x), self.evaluate(special_math_ops.bessel_y0(x_wt))
      )
      self.assertAllClose(
          special.y1(x), self.evaluate(special_math_ops.bessel_y1(x_wt))
      )
    except ImportError as e:
      tf_logging.warn('Cannot test special functions: %s' % str(e))

  @parameterized.parameters(np.float32, np.float64)
  def test_besseli_larger(self, dtype):
    x = np.random.uniform(1.0, 20.0, size=int(1e4)).astype(dtype)
    x_wt = _get_weak_tensor(x)

    try:
      from scipy import special  # pylint: disable=g-import-not-at-top

      self.assertAllClose(
          special.i0e(x), self.evaluate(special_math_ops.bessel_i0e(x_wt))
      )
      self.assertAllClose(
          special.i1e(x), self.evaluate(special_math_ops.bessel_i1e(x_wt))
      )
    except ImportError as e:
      tf_logging.warn('Cannot test special functions: %s' % str(e))

  @parameterized.parameters(np.float32, np.float64)
  def test_besselj_larger(self, dtype):
    x = np.random.uniform(1.0, 30.0, size=int(1e4)).astype(dtype)
    x_wt = _get_weak_tensor(x)

    try:
      from scipy import special  # pylint: disable=g-import-not-at-top

      self.assertAllClose(
          special.j0(x), self.evaluate(special_math_ops.bessel_j0(x_wt))
      )
      self.assertAllClose(
          special.j1(x), self.evaluate(special_math_ops.bessel_j1(x_wt))
      )
    except ImportError as e:
      tf_logging.warn('Cannot test special functions: %s' % str(e))

  @parameterized.parameters(np.float32, np.float64)
  def test_besselk_larger(self, dtype):
    x = np.random.uniform(1.0, 30.0, size=int(1e4)).astype(dtype)
    x_wt = _get_weak_tensor(x)
    try:
      from scipy import special  # pylint: disable=g-import-not-at-top

      self.assertAllClose(
          special.k0(x), self.evaluate(special_math_ops.bessel_k0(x_wt))
      )
      self.assertAllClose(
          special.k0e(x), self.evaluate(special_math_ops.bessel_k0e(x_wt))
      )
      self.assertAllClose(
          special.k1(x), self.evaluate(special_math_ops.bessel_k1(x_wt))
      )
      self.assertAllClose(
          special.k1e(x), self.evaluate(special_math_ops.bessel_k1e(x_wt))
      )
    except ImportError as e:
      tf_logging.warn('Cannot test special functions: %s' % str(e))

  @parameterized.parameters(np.float32, np.float64)
  def test_bessely_larger(self, dtype):
    x = np.random.uniform(1.0, 30.0, size=int(1e4)).astype(dtype)
    x_wt = _get_weak_tensor(x)
    try:
      from scipy import special  # pylint: disable=g-import-not-at-top

      self.assertAllClose(
          special.y0(x), self.evaluate(special_math_ops.bessel_y0(x_wt))
      )
      self.assertAllClose(
          special.y1(x), self.evaluate(special_math_ops.bessel_y1(x_wt))
      )
    except ImportError as e:
      tf_logging.warn('Cannot test special functions: %s' % str(e))

  def test_besseli_gradient(self):
    inputs = [_get_weak_tensor(np.random.uniform(-10.0, 10.0, size=int(1e2)))]
    analytical, numerical = gradient_checker_v2.compute_gradient(
        special_math_ops.bessel_i0, inputs)
    self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 1e-3)

    analytical, numerical = gradient_checker_v2.compute_gradient(
        special_math_ops.bessel_i0e, inputs)
    self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 1e-4)

    analytical, numerical = gradient_checker_v2.compute_gradient(
        special_math_ops.bessel_i1, inputs)
    self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 1e-3)

    analytical, numerical = gradient_checker_v2.compute_gradient(
        special_math_ops.bessel_i1e, inputs)
    self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 1e-4)

  def test_besselj_gradient(self):
    inputs = [_get_weak_tensor(np.random.uniform(-50.0, 50.0, size=int(1e2)))]
    analytical, numerical = gradient_checker_v2.compute_gradient(
        special_math_ops.bessel_j0, inputs)
    self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 1e-4)

    analytical, numerical = gradient_checker_v2.compute_gradient(
        special_math_ops.bessel_j1, inputs)
    self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 1e-4)

  def test_besselk_gradient(self):
    inputs = [_get_weak_tensor(np.random.uniform(1.0, 50.0, size=int(1e2)))]
    analytical, numerical = gradient_checker_v2.compute_gradient(
        special_math_ops.bessel_k0, inputs)
    self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 1e-4)

    analytical, numerical = gradient_checker_v2.compute_gradient(
        special_math_ops.bessel_k0e, inputs)
    self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 1e-4)

    analytical, numerical = gradient_checker_v2.compute_gradient(
        special_math_ops.bessel_k1, inputs)
    self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 1e-4)

    analytical, numerical = gradient_checker_v2.compute_gradient(
        special_math_ops.bessel_k1e, inputs)
    self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 1e-4)

  def test_bessely_gradient(self):
    inputs = [_get_weak_tensor(np.random.uniform(1.0, 50.0, size=int(1e2)))]
    analytical, numerical = gradient_checker_v2.compute_gradient(
        special_math_ops.bessel_y0, inputs)
    self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 1e-4)

    analytical, numerical = gradient_checker_v2.compute_gradient(
        special_math_ops.bessel_y1, inputs)
    self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 1e-4)

# TODO(b/291943949): Add WeakTensor support for Einsum.

if __name__ == '__main__':
  ops.set_dtype_conversion_mode('all')
  test.main()
