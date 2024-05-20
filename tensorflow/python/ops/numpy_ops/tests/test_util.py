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
"""NumPy test utilities."""
from contextlib import contextmanager
from distutils.util import strtobool
import functools
from functools import partial
import re
import itertools as it
import os
from typing import Dict, Sequence, Union
import unittest
import warnings
import zlib

from absl.testing import absltest
from absl.testing import parameterized

import numpy as onp
import numpy.random as npr

from tensorflow.python.util import nest
from tensorflow.python.framework import tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gradient_checker_v2

from tensorflow.python.ops.numpy_ops.tests.config import flags
import tensorflow.python.ops.numpy_ops.tests.extensions as nje
from tensorflow.python.ops.numpy_ops import np_utils
from tensorflow.python.ops.numpy_ops import np_array_ops


tree_map = nest.map_structure
tree_multimap = nest.map_structure


FLAGS = flags.FLAGS


# TODO(wangpeng): Remove this flag after broken tests are fixed
flags.DEFINE_bool('enable_x64',
                  strtobool('False'),
                  'Enable 64-bit types to be used.')


flags.DEFINE_enum(
    'test_dut', '',
    enum_values=['', 'cpu', 'gpu', 'tpu'],
    help=
    'Describes the device under test in case special consideration is required.'
)


flags.DEFINE_integer(
  'num_generated_cases',
  10,
  help='Number of generated cases to test')


EPS = 1e-4


# Default dtypes corresponding to Python scalars.
python_scalar_dtypes = {
  bool: onp.dtype(onp.bool_),
  int: onp.dtype(onp.int_),
  float: onp.dtype(onp.float64),
  complex: onp.dtype(onp.complex128),
}


def _dtype(x):
  if isinstance(x, tensor.Tensor):
    return x.dtype.as_numpy_dtype
  return (getattr(x, 'dtype', None) or
          onp.dtype(python_scalar_dtypes.get(type(x), None)) or
          onp.asarray(x).dtype)


def is_sequence(x):
  try:
    iter(x)
  except TypeError:
    return False
  else:
    return True

_default_tolerance = {
  onp.dtype(onp.bool_): 0,
  onp.dtype(onp.int8): 0,
  onp.dtype(onp.int16): 0,
  onp.dtype(onp.int32): 0,
  onp.dtype(onp.int64): 0,
  onp.dtype(onp.uint8): 0,
  onp.dtype(onp.uint16): 0,
  onp.dtype(onp.uint32): 0,
  onp.dtype(onp.uint64): 0,
  # TODO(b/154768983): onp.dtype(dtypes.bfloat16): 1e-2,
  onp.dtype(onp.float16): 1e-3,
  onp.dtype(onp.float32): 1e-6,
  onp.dtype(onp.float64): 1e-15,
  onp.dtype(onp.complex64): 1e-6,
  onp.dtype(onp.complex128): 1e-15,
}

def default_tolerance():
  return _default_tolerance

default_gradient_tolerance = {
  # TODO(b/154768983): onp.dtype(dtypes.bfloat16): 1e-1,
  onp.dtype(onp.float16): 1e-2,
  onp.dtype(onp.float32): 2e-3,
  onp.dtype(onp.float64): 1e-5,
  onp.dtype(onp.complex64): 1e-3,
  onp.dtype(onp.complex128): 1e-5,
}

def _assert_numpy_allclose(a, b, atol=None, rtol=None):
  # TODO(b/154768983):
  #   a = a.astype(onp.float32) if a.dtype == dtypes.bfloat16 else a
  #   b = b.astype(onp.float32) if b.dtype == dtypes.bfloat16 else b
  kw = {}
  if atol: kw["atol"] = atol
  if rtol: kw["rtol"] = rtol
  onp.testing.assert_allclose(a, b, **kw)

def tolerance(dtype, tol=None):
  tol = {} if tol is None else tol
  if not isinstance(tol, dict):
    return tol
  tol = {onp.dtype(key): value for key, value in tol.items()}
  dtype = onp.dtype(dtype)
  return tol.get(dtype, default_tolerance()[dtype])

def _normalize_tolerance(tol):
  tol = tol or 0
  if isinstance(tol, dict):
    return {onp.dtype(k): v for k, v in tol.items()}
  else:
    return {k: tol for k in _default_tolerance}

def join_tolerance(tol1, tol2):
  tol1 = _normalize_tolerance(tol1)
  tol2 = _normalize_tolerance(tol2)
  out = tol1
  for k, v in tol2.items():
    out[k] = max(v, tol1.get(k, 0))
  return out

def _assert_numpy_close(a, b, atol=None, rtol=None):
  assert a.shape == b.shape
  atol = max(tolerance(a.dtype, atol), tolerance(b.dtype, atol))
  rtol = max(tolerance(a.dtype, rtol), tolerance(b.dtype, rtol))
  _assert_numpy_allclose(a, b, atol=atol * a.size, rtol=rtol * b.size)


def check_eq(xs, ys):
  tree_all(tree_multimap(_assert_numpy_allclose, xs, ys))


def check_close(xs, ys, atol=None, rtol=None):
  assert_close = partial(_assert_numpy_close, atol=atol, rtol=rtol)
  tree_all(tree_multimap(assert_close, xs, ys))


def inner_prod(xs, ys):
  def contract(x, y):
    return onp.real(onp.dot(onp.conj(x).reshape(-1), y.reshape(-1)))
  return tree_reduce(onp.add, tree_multimap(contract, xs, ys))


add = partial(tree_multimap, lambda x, y: onp.add(x, y, dtype=_dtype(x)))
sub = partial(tree_multimap, lambda x, y: onp.subtract(x, y, dtype=_dtype(x)))
conj = partial(tree_map, lambda x: onp.conj(x, dtype=_dtype(x)))

def scalar_mul(xs, a):
  return tree_map(lambda x: onp.multiply(x, a, dtype=_dtype(x)), xs)


def rand_like(rng, x):
  shape = onp.shape(x)
  dtype = _dtype(x)
  randn = lambda: onp.asarray(rng.randn(*shape), dtype=dtype)
  if onp.issubdtype(dtype, onp.complexfloating):
    return randn() + dtype.type(1.0j) * randn()
  else:
    return randn()


def numerical_jvp(f, primals, tangents, eps=EPS):
  delta = scalar_mul(tangents, eps)
  f_pos = f(*add(primals, delta))
  f_neg = f(*sub(primals, delta))
  return scalar_mul(sub(f_pos, f_neg), 0.5 / eps)


def _merge_tolerance(tol, default):
  if tol is None:
    return default
  if not isinstance(tol, dict):
    return tol
  out = default.copy()
  for k, v in tol.items():
    out[onp.dtype(k)] = v
  return out

def check_jvp(f, f_jvp, args, atol=None, rtol=None, eps=EPS):
  atol = _merge_tolerance(atol, default_gradient_tolerance)
  rtol = _merge_tolerance(rtol, default_gradient_tolerance)
  rng = onp.random.RandomState(0)
  tangent = tree_map(partial(rand_like, rng), args)
  v_out, t_out = f_jvp(args, tangent)
  v_out_expected = f(*args)
  t_out_expected = numerical_jvp(f, args, tangent, eps=eps)
  # In principle we should expect exact equality of v_out and v_out_expected,
  # but due to nondeterminism especially on GPU (e.g., due to convolution
  # autotuning) we only require "close".
  check_close(v_out, v_out_expected, atol=atol, rtol=rtol)
  check_close(t_out, t_out_expected, atol=atol, rtol=rtol)


def check_vjp(f, f_vjp, args, atol=None, rtol=None, eps=EPS):
  atol = _merge_tolerance(atol, default_gradient_tolerance)
  rtol = _merge_tolerance(rtol, default_gradient_tolerance)
  _rand_like = partial(rand_like, onp.random.RandomState(0))
  v_out, vjpfun = f_vjp(*args)
  v_out_expected = f(*args)
  check_close(v_out, v_out_expected, atol=atol, rtol=rtol)
  tangent = tree_map(_rand_like, args)
  tangent_out = numerical_jvp(f, args, tangent, eps=eps)
  cotangent = tree_map(_rand_like, v_out)
  cotangent_out = conj(vjpfun(conj(cotangent)))
  ip = inner_prod(tangent, cotangent_out)
  ip_expected = inner_prod(tangent_out, cotangent)
  check_close(ip, ip_expected, atol=atol, rtol=rtol)


def device_under_test():
  return FLAGS.test_dut

def if_device_under_test(device_type: Union[str, Sequence[str]],
                         if_true, if_false):
  """Chooses `if_true` of `if_false` based on device_under_test."""
  if device_under_test() in ([device_type] if isinstance(device_type, str)
                             else device_type):
    return if_true
  else:
    return if_false

def supported_dtypes():
  if device_under_test() == "tpu":
    return {onp.bool_, onp.int32, onp.uint32, dtypes.bfloat16, onp.float32,
            onp.complex64}
  else:
    return {onp.bool_, onp.int8, onp.int16, onp.int32, onp.int64,
            onp.uint8, onp.uint16, onp.uint32, onp.uint64,
            dtypes.bfloat16, onp.float16, onp.float32, onp.float64,
            onp.complex64, onp.complex128}

def skip_if_unsupported_type(dtype):
  if dtype not in supported_dtypes():
    raise unittest.SkipTest(
      f"Type {dtype} not supported on {device_under_test()}")

def skip_on_devices(*disabled_devices):
  """A decorator for test methods to skip the test on certain devices."""
  def skip(test_method):
    @functools.wraps(test_method)
    def test_method_wrapper(self, *args, **kwargs):
      device = device_under_test()
      if device in disabled_devices:
        test_name = getattr(test_method, '__name__', '[unknown test]')
        raise unittest.SkipTest(
          f"{test_name} not supported on {device.upper()}.")
      return test_method(self, *args, **kwargs)
    return test_method_wrapper
  return skip


def skip_on_flag(flag_name, skip_value):
  """A decorator for test methods to skip the test when flags are set."""
  def skip(test_method):        # pylint: disable=missing-docstring
    @functools.wraps(test_method)
    def test_method_wrapper(self, *args, **kwargs):
      flag_value = getattr(FLAGS, flag_name)
      if flag_value == skip_value:
        test_name = getattr(test_method, '__name__', '[unknown test]')
        raise unittest.SkipTest(
          f"{test_name} not supported when FLAGS.{flag_name} is {flag_value}")
      return test_method(self, *args, **kwargs)
    return test_method_wrapper
  return skip


def format_test_name_suffix(opname, shapes, dt):
  arg_descriptions = (format_shape_dtype_string(shape, dtype)
                      for shape, dtype in zip(shapes, dt))
  return '{}_{}'.format(opname.capitalize(), '_'.join(arg_descriptions))


# We use special symbols, represented as singleton objects, to distinguish
# between NumPy scalars, Python scalars, and 0-D arrays.
class ScalarShape:
  def __len__(self): return 0
  def __getitem__(self, i):
    raise IndexError(f'index {i} out of range.')
class _NumpyScalar(ScalarShape): pass
class _PythonScalar(ScalarShape): pass
NUMPY_SCALAR_SHAPE = _NumpyScalar()
PYTHON_SCALAR_SHAPE = _PythonScalar()


def _dims_of_shape(shape):
  """Converts `shape` to a tuple of dimensions."""
  if type(shape) in (list, tuple):
    return shape
  elif isinstance(shape, ScalarShape):
    return ()
  else:
    raise TypeError(type(shape))


def _cast_to_shape(value, shape, dtype):
  """Casts `value` to the correct Python type for `shape` and `dtype`."""
  if shape is NUMPY_SCALAR_SHAPE:
    # explicitly cast to NumPy scalar in case `value` is a Python scalar.
    return onp.dtype(dtype).type(value)
  elif shape is PYTHON_SCALAR_SHAPE:
    # explicitly cast to Python scalar via https://stackoverflow.com/a/11389998
    return onp.asarray(value).item()
  elif type(shape) in (list, tuple):
    assert onp.shape(value) == tuple(shape)
    return value
  else:
    raise TypeError(type(shape))


def dtype_str(dtype):
  return onp.dtype(dtype).name


def format_shape_dtype_string(shape, dtype):
  if shape is NUMPY_SCALAR_SHAPE:
    return dtype_str(dtype)
  elif shape is PYTHON_SCALAR_SHAPE:
    return 'py' + dtype_str(dtype)
  elif type(shape) in (list, tuple):
    shapestr = ','.join(str(dim) for dim in shape)
    return '{}[{}]'.format(dtype_str(dtype), shapestr)
  elif type(shape) is int:
    return '{}[{},]'.format(dtype_str(dtype), shape)
  elif isinstance(shape, onp.ndarray):
    return '{}[{}]'.format(dtype_str(dtype), shape)
  else:
    raise TypeError(type(shape))


def _rand_dtype(rand, shape, dtype, scale=1., post=lambda x: x):
  """Produce random values given shape, dtype, scale, and post-processor.

  Args:
    rand: a function for producing random values of a given shape, e.g. a
      bound version of either onp.RandomState.randn or onp.RandomState.rand.
    shape: a shape value as a tuple of positive integers.
    dtype: a numpy dtype.
    scale: optional, a multiplicative scale for the random values (default 1).
    post: optional, a callable for post-processing the random values (default
      identity).

  Returns:
    An ndarray of the given shape and dtype using random values based on a call
    to rand but scaled, converted to the appropriate dtype, and post-processed.
  """
  r = lambda: onp.asarray(scale * rand(*_dims_of_shape(shape)), dtype)
  if onp.issubdtype(dtype, onp.complexfloating):
    vals = r() + 1.0j * r()
  else:
    vals = r()
  return _cast_to_shape(onp.asarray(post(vals), dtype), shape, dtype)


def rand_default(scale=3):
  randn = npr.RandomState(0).randn
  return partial(_rand_dtype, randn, scale=scale)


def rand_nonzero():
  post = lambda x: onp.where(x == 0, onp.array(1, dtype=x.dtype), x)
  randn = npr.RandomState(0).randn
  return partial(_rand_dtype, randn, scale=3, post=post)


def rand_positive():
  post = lambda x: x + 1
  rand = npr.RandomState(0).rand
  return partial(_rand_dtype, rand, scale=2, post=post)


def rand_small():
  randn = npr.RandomState(0).randn
  return partial(_rand_dtype, randn, scale=1e-3)


def rand_not_small(offset=10.):
  post = lambda x: x + onp.where(x > 0, offset, -offset)
  randn = npr.RandomState(0).randn
  return partial(_rand_dtype, randn, scale=3., post=post)


def rand_small_positive():
  rand = npr.RandomState(0).rand
  return partial(_rand_dtype, rand, scale=2e-5)

def rand_uniform(low=0.0, high=1.0):
  assert low < high
  rand = npr.RandomState(0).rand
  post = lambda x: x * (high - low) + low
  return partial(_rand_dtype, rand, post=post)


def rand_some_equal():
  randn = npr.RandomState(0).randn
  rng = npr.RandomState(0)

  def post(x):
    x_ravel = x.ravel()
    if len(x_ravel) == 0:
      return x
    flips = rng.rand(*onp.shape(x)) < 0.5
    return onp.where(flips, x_ravel[0], x)

  return partial(_rand_dtype, randn, scale=100., post=post)


def rand_some_inf():
  """Return a random sampler that produces infinities in floating types."""
  rng = npr.RandomState(1)
  base_rand = rand_default()

  """
  TODO: Complex numbers are not correctly tested
  If blocks should be switched in order, and relevant tests should be fixed
  """
  def rand(shape, dtype):
    """The random sampler function."""
    if not onp.issubdtype(dtype, onp.floating):
      # only float types have inf
      return base_rand(shape, dtype)

    if onp.issubdtype(dtype, onp.complexfloating):
      base_dtype = onp.real(onp.array(0, dtype=dtype)).dtype
      out = (rand(shape, base_dtype) +
             onp.array(1j, dtype) * rand(shape, base_dtype))
      return _cast_to_shape(out, shape, dtype)

    dims = _dims_of_shape(shape)
    posinf_flips = rng.rand(*dims) < 0.1
    neginf_flips = rng.rand(*dims) < 0.1

    vals = base_rand(shape, dtype)
    vals = onp.where(posinf_flips, onp.array(onp.inf, dtype=dtype), vals)
    vals = onp.where(neginf_flips, onp.array(-onp.inf, dtype=dtype), vals)

    return _cast_to_shape(onp.asarray(vals, dtype=dtype), shape, dtype)

  return rand

def rand_some_nan():
  """Return a random sampler that produces nans in floating types."""
  rng = npr.RandomState(1)
  base_rand = rand_default()

  def rand(shape, dtype):
    """The random sampler function."""
    if onp.issubdtype(dtype, onp.complexfloating):
      base_dtype = onp.real(onp.array(0, dtype=dtype)).dtype
      out = (rand(shape, base_dtype) +
             onp.array(1j, dtype) * rand(shape, base_dtype))
      return _cast_to_shape(out, shape, dtype)

    if not onp.issubdtype(dtype, onp.floating):
      # only float types have inf
      return base_rand(shape, dtype)

    dims = _dims_of_shape(shape)
    nan_flips = rng.rand(*dims) < 0.1

    vals = base_rand(shape, dtype)
    vals = onp.where(nan_flips, onp.array(onp.nan, dtype=dtype), vals)

    return _cast_to_shape(onp.asarray(vals, dtype=dtype), shape, dtype)

  return rand

def rand_some_inf_and_nan():
  """Return a random sampler that produces infinities in floating types."""
  rng = npr.RandomState(1)
  base_rand = rand_default()

  """
  TODO: Complex numbers are not correctly tested
  If blocks should be switched in order, and relevant tests should be fixed
  """
  def rand(shape, dtype):
    """The random sampler function."""
    if not onp.issubdtype(dtype, onp.floating):
      # only float types have inf
      return base_rand(shape, dtype)

    if onp.issubdtype(dtype, onp.complexfloating):
      base_dtype = onp.real(onp.array(0, dtype=dtype)).dtype
      out = (rand(shape, base_dtype) +
             onp.array(1j, dtype) * rand(shape, base_dtype))
      return _cast_to_shape(out, shape, dtype)

    dims = _dims_of_shape(shape)
    posinf_flips = rng.rand(*dims) < 0.1
    neginf_flips = rng.rand(*dims) < 0.1
    nan_flips = rng.rand(*dims) < 0.1

    vals = base_rand(shape, dtype)
    vals = onp.where(posinf_flips, onp.array(onp.inf, dtype=dtype), vals)
    vals = onp.where(neginf_flips, onp.array(-onp.inf, dtype=dtype), vals)
    vals = onp.where(nan_flips, onp.array(onp.nan, dtype=dtype), vals)

    return _cast_to_shape(onp.asarray(vals, dtype=dtype), shape, dtype)

  return rand

# TODO(mattjj): doesn't handle complex types
def rand_some_zero():
  """Return a random sampler that produces some zeros."""
  rng = npr.RandomState(1)
  base_rand = rand_default()

  def rand(shape, dtype):
    """The random sampler function."""
    dims = _dims_of_shape(shape)
    zeros = rng.rand(*dims) < 0.5

    vals = base_rand(shape, dtype)
    vals = onp.where(zeros, onp.array(0, dtype=dtype), vals)

    return _cast_to_shape(onp.asarray(vals, dtype=dtype), shape, dtype)

  return rand


def rand_int(low, high=None):
  randint = npr.RandomState(0).randint
  def fn(shape, dtype):
    return randint(low, high=high, size=shape, dtype=dtype)
  return fn

def rand_unique_int():
  randchoice = npr.RandomState(0).choice
  def fn(shape, dtype):
    return randchoice(onp.arange(onp.prod(shape), dtype=dtype),
                      size=shape, replace=False)
  return fn

def rand_bool():
  rng = npr.RandomState(0)
  def generator(shape, dtype):
    return _cast_to_shape(rng.rand(*_dims_of_shape(shape)) < 0.5, shape, dtype)
  return generator

def check_raises(thunk, err_type, msg):
  try:
    thunk()
    assert False
  except err_type as e:
    assert str(e).startswith(msg), "\n{}\n\n{}\n".format(e, msg)

def check_raises_regexp(thunk, err_type, pattern):
  try:
    thunk()
    assert False
  except err_type as e:
    assert re.match(pattern, str(e)), "{}\n\n{}\n".format(e, pattern)


def _iter_eqns(jaxpr):
  # TODO(necula): why doesn't this search in params?
  for eqn in jaxpr.eqns:
    yield eqn
  for subjaxpr in core.subjaxprs(jaxpr):
    yield from _iter_eqns(subjaxpr)

def assert_dot_precision(expected_precision, fun, *args):
  jaxpr = api.make_jaxpr(fun)(*args)
  precisions = [eqn.params['precision'] for eqn in _iter_eqns(jaxpr.jaxpr)
                if eqn.primitive == lax.dot_general_p]
  for precision in precisions:
    msg = "Unexpected precision: {} != {}".format(expected_precision, precision)
    assert precision == expected_precision, msg


_CACHED_INDICES: Dict[int, Sequence[int]] = {}

def cases_from_list(xs):
  xs = list(xs)
  n = len(xs)
  k = min(n, FLAGS.num_generated_cases)
  # Random sampling for every parameterized test is expensive. Do it once and
  # cache the result.
  indices = _CACHED_INDICES.get(n)
  if indices is None:
    rng = npr.RandomState(42)
    _CACHED_INDICES[n] = indices = rng.permutation(n)
  return [xs[i] for i in indices[:k]]

def cases_from_gens(*gens):
  sizes = [1, 3, 10]
  cases_per_size = int(FLAGS.num_generated_cases / len(sizes)) + 1
  for size in sizes:
    for i in range(cases_per_size):
      yield ('_{}_{}'.format(size, i),) + tuple(gen(size) for gen in gens)


def to_np(a):
  return nest.map_structure(np_array_ops.asarray, a)


def to_tf_fn(f):
  return lambda *args: f(*to_np(args))


class TestCase(parameterized.TestCase):
  """Base class for tests including numerical checks and boilerplate."""

  # copied from jax.test_util
  def setUp(self):
    super().setUp()
    self._rng = npr.RandomState(zlib.adler32(self._testMethodName.encode()))

  # copied from jax.test_util
  def rng(self):
    return self._rng

  # TODO(mattjj): this obscures the error messages from failures, figure out how
  # to re-enable it
  # def tearDown(self) -> None:
  #   assert core.reset_trace_state()

  def assertArraysAllClose(self, x, y, check_dtypes, atol=None, rtol=None):
    """Assert that x and y are close (up to numerical tolerances)."""
    self.assertEqual(x.shape, y.shape)
    atol = max(tolerance(_dtype(x), atol), tolerance(_dtype(y), atol))
    rtol = max(tolerance(_dtype(x), rtol), tolerance(_dtype(y), rtol))

    _assert_numpy_allclose(x, y, atol=atol, rtol=rtol)

    if check_dtypes:
      self.assertDtypesMatch(x, y)

  def assertDtypesMatch(self, x, y):
    if FLAGS.enable_x64:
      self.assertEqual(_dtype(x), _dtype(y))

  def assertAllClose(self, x, y, check_dtypes, atol=None, rtol=None):
    """Assert that x and y, either arrays or nested tuples/lists, are close."""
    if isinstance(x, dict):
      self.assertIsInstance(y, dict)
      self.assertEqual(set(x.keys()), set(y.keys()))
      for k in x:
        self.assertAllClose(x[k], y[k], check_dtypes, atol=atol, rtol=rtol)
    elif is_sequence(x) and not hasattr(x, '__array__'):
      self.assertTrue(is_sequence(y) and not hasattr(y, '__array__'))
      self.assertEqual(len(x), len(y))
      for x_elt, y_elt in zip(x, y):
        self.assertAllClose(x_elt, y_elt, check_dtypes, atol=atol, rtol=rtol)
    elif hasattr(x, '__array__') or onp.isscalar(x):
      self.assertTrue(hasattr(y, '__array__') or onp.isscalar(y))
      if check_dtypes:
        self.assertDtypesMatch(x, y)
      x = onp.asarray(x)
      y = onp.asarray(y)
      self.assertArraysAllClose(x, y, check_dtypes=False, atol=atol, rtol=rtol)
    elif x == y:
      return
    else:
      raise TypeError((type(x), type(y)))

  def assertMultiLineStrippedEqual(self, expected, what):
    """Asserts two strings are equal, after stripping each line."""
    ignore_space_re = re.compile(r'\s*\n\s*')
    expected_clean = re.sub(ignore_space_re, '\n', expected.strip())
    what_clean = re.sub(ignore_space_re, '\n', what.strip())
    self.assertMultiLineEqual(expected_clean, what_clean,
                              msg="Found\n{}\nExpecting\n{}".format(what, expected))

  def _CheckAgainstNumpy(self, numpy_reference_op, lax_op, args_maker,
                         check_dtypes=True, tol=None):
    args = args_maker()
    lax_ans = lax_op(*args)
    numpy_ans = numpy_reference_op(*args)
    self.assertAllClose(numpy_ans, lax_ans, check_dtypes=check_dtypes,
                        atol=tol, rtol=tol)

  def _CompileAndCheck(self,
                       fun,
                       args_maker,
                       check_dtypes=True,
                       rtol=None,
                       atol=None,
                       check_eval_on_shapes=True,
                       check_incomplete_shape=True,
                       check_unknown_rank=True,
                       static_argnums=(),
                       check_experimental_compile=True,
                       check_xla_forced_compile=True):
    """Compiles the function and checks the results.

    Args:
      fun: the function to be checked.
      args_maker: a callable that returns a tuple which will be used as the
        positional arguments.
      check_dtypes: whether to check that the result dtypes from non-compiled
        and compiled runs agree.
      rtol: relative tolerance for allclose assertions.
      atol: absolute tolerance for allclose assertions.
      check_eval_on_shapes: whether to run `eval_on_shapes` on the function and
        check that the result shapes and dtypes are correct.
      check_incomplete_shape: whether to check that the function can handle
        incomplete shapes (including those with and without a known rank).
      check_unknown_rank: (only has effect when check_incomplete_shape is True)
        whether to check that the function can handle unknown ranks.
      static_argnums: indices of arguments to be treated as static arguments for
        `jit` and `eval_on_shapes`.
      check_experimental_compile: whether to check compilation with
        experimental_compile=True (in addition to compilation without the flag).
      check_xla_forced_compile: whether to check compilation with
        forced_compile=True (in addition to compilation without the flag). This
        flag is different from experimental_compile because it enforces
        whole-function compilation while the latter doesn't. TPU requires
        whole-function compilation.
    """
    args = args_maker()

    for x in args:
      if not hasattr(x, 'dtype'):
        # If there is a input that doesn't have dtype info, jit and
        # eval_on_shapes may pick a different dtype for it than numpy, so we
        # skip the dtype check.
        check_dtypes = False

    python_ans = fun(*args)

    python_shapes = nest.map_structure(onp.shape, python_ans)
    onp_shapes = nest.map_structure(
        lambda x: onp.shape(onp.asarray(x)), python_ans
    )
    self.assertEqual(python_shapes, onp_shapes)

    def check_compile(**kwargs):
      # `wrapped_fun` and `python_should_be_executing` are used to check that
      # when the jitted function is called the second time, the original Python
      # function won't be executed.
      def wrapped_fun(*args):
        self.assertTrue(python_should_be_executing)
        return fun(*args)

      cfun = nje.jit(wrapped_fun, static_argnums=static_argnums, **kwargs)
      python_should_be_executing = True
      monitored_ans = cfun(*args)

      python_should_be_executing = False
      compiled_ans = cfun(*args)

      self.assertAllClose(python_ans, monitored_ans, check_dtypes, atol, rtol)
      self.assertAllClose(python_ans, compiled_ans, check_dtypes, atol, rtol)

      # Run `cfun` with a different set of arguments to check that changing
      # arguments won't cause recompilation.

      new_args = args_maker()

      skip_retracing_test = False
      for old, new in zip(nest.flatten(args), nest.flatten(new_args)):
        if nje.most_precise_int_dtype(old) != nje.most_precise_int_dtype(new):
          # If the old and new arguments result in different dtypes (because
          # they fall into different value ranges), tf-numpy will retrace, so we
          # skip the no-retrace test.
          skip_retracing_test = True

      if not skip_retracing_test:
        python_should_be_executing = True
        new_python_ans = fun(*new_args)
        python_should_be_executing = False
        compiled_ans = cfun(*new_args)
        self.assertAllClose(new_python_ans, compiled_ans, check_dtypes, atol,
                            rtol)

    check_compile()
    if check_experimental_compile:
      check_compile(experimental_compile=True)
    if check_xla_forced_compile:
      check_compile(xla_forced_compile=True)

    if check_eval_on_shapes:
      # Check that nje.eval_on_shapes can get complete output shapes given
      # complete input shapes.
      cfun = nje.eval_on_shapes(fun, static_argnums=static_argnums)
      compiled_ans = cfun(*args)
      flat_python_ans = nest.flatten(python_ans)
      flat_compiled_ans = nest.flatten(compiled_ans)
      self.assertEqual(len(flat_python_ans), len(flat_compiled_ans))
      for a, b in zip(flat_python_ans, flat_compiled_ans):
        if hasattr(a, 'shape'):
          self.assertEqual(a.shape, b.shape)
        if check_dtypes and hasattr(a, 'dtype'):
          self.assertEqual(dtypes.as_dtype(a.dtype), b.dtype)

    # If some argument doesn't have a `dtype` attr (e.g. a Python scalar), we
    # skip incomplete-shape checks, since shape specs need dtype. It's OK to
    # skip since the same incomplete-shape checks will run for []-shaped arrays.
    if check_incomplete_shape and all(hasattr(x, 'dtype') for x in args):
      # Check partial shapes with known ranks.
      # Numpy scalars (created by e.g. np.int32(5)) have `dtype` but not
      # `shape`.
      if all(hasattr(x, 'shape') for x in args):
        specs = [
            tensor.TensorSpec([None] * len(x.shape), x.dtype) for x in args
        ]
        cfun = nje.jit(
            fun, static_argnums=static_argnums, input_signature=specs
        )
        compiled_ans = cfun(*args)
        self.assertAllClose(python_ans, compiled_ans, check_dtypes, atol, rtol)

      if check_unknown_rank:
        # Check unknown ranks.
        specs = [tensor.TensorSpec(None, x.dtype) for x in args]
        cfun = nje.jit(
            fun, static_argnums=static_argnums, input_signature=specs)
        compiled_ans = cfun(*args)
        self.assertAllClose(python_ans, compiled_ans, check_dtypes, atol, rtol)

  def check_grads(self, f, args, atol=None, rtol=None, delta=None):
    """Check gradients against finite differences.

    Args:
      f: function to check at ``f(*args)``.
      args: a list or tuple of argument values.
      atol: absolute tolerance for gradient equality.
      rtol: relative tolerance for gradient equality.
      delta: step size used for finite differences.
    """
    if delta is None:
      # Optimal stepsize for central difference is O(epsilon^{1/3}).
      dtype = np_utils.result_type(*args)
      epsilon = onp.finfo(dtype).eps
      delta = epsilon ** (1.0 / 3.0)
    theoretical, numerical = gradient_checker_v2.compute_gradient(
        to_tf_fn(f), args, delta=delta)
    self.assertAllClose(theoretical, numerical, check_dtypes=False, atol=atol,
                        rtol=rtol)


@contextmanager
def ignore_warning(**kw):
  with warnings.catch_warnings():
    warnings.filterwarnings("ignore", **kw)
    yield


def disable(_):

  def wrapper(self, *args, **kwargs):
    self.skipTest('Test is disabled')

  return wrapper
