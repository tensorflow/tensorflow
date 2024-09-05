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


import collections
import enum
from functools import partial
import itertools
import unittest

from absl.testing import absltest
from absl.testing import parameterized
import numpy as onp

from tensorflow.python.framework import config as tf_config
from tensorflow.python.ops.numpy_ops.tests.config import config
import tensorflow.python.ops.numpy_ops.tests.extensions as nje
import tensorflow.python.ops.numpy_ops.tests.np_wrapper as tnp
import tensorflow.python.ops.numpy_ops.tests.test_util as jtu
from tensorflow.python.util import nest

config.parse_flags_with_absl()


# We disable the whitespace continuation check in this file because otherwise it
# makes the test name formatting unwieldy.
# pylint: disable=bad-continuation
# We also disable undefined-variable till we start enabling tests.
# pylint: disable=undefined-variable


def subvals(lst, replace):
  lst = list(lst)
  for i, v in replace:
    lst[i] = v
  return tuple(lst)


float_dtypes = [onp.float32, onp.float64]
int_dtypes = [onp.int32, onp.int64]
bool_types = [onp.bool_]
default_dtypes = float_dtypes + int_dtypes
all_dtypes = float_dtypes + int_dtypes + bool_types

IndexSpec = collections.namedtuple("IndexTest", ["shape", "indexer"])


suppress_deprecated_indexing_warnings = partial(
  jtu.ignore_warning, category=FutureWarning,
  message='Using a non-tuple sequence.*')


STATIC_INDEXING_TESTS = [
    ("OneIntIndex", [
        IndexSpec(shape=(3,), indexer=1),
        IndexSpec(shape=(3, 3), indexer=0),
        IndexSpec(shape=(3, 4, 5), indexer=2),
        IndexSpec(shape=(3,), indexer=-1),
        IndexSpec(shape=(3,), indexer=-2),
    ]),
    ("TwoIntIndices", [
        IndexSpec(shape=(3, 3), indexer=(2, 1)),
        IndexSpec(shape=(3, 4, 5), indexer=(1, 2)),
        IndexSpec(shape=(3, 4, 5), indexer=(-1, 2)),
    ]),
    ("ThreeIntIndices", [IndexSpec((3, 4, 5), indexer=(1, 2, 3))]),
    ("OneSliceIndex", [
        IndexSpec(shape=(10,), indexer=slice(1, 3)),
        IndexSpec(shape=(10,), indexer=slice(1, -1)),
        IndexSpec(shape=(10,), indexer=slice(None, -1)),
        IndexSpec(shape=(10,), indexer=slice(None, None, None)),
        IndexSpec(shape=(10, 8), indexer=slice(1, 3)),
        IndexSpec(shape=(10, 8), indexer=slice(1, None)),
        IndexSpec(shape=(10, 8), indexer=slice(None, 3)),
        IndexSpec(shape=(10, 8), indexer=slice(-3, None)),
    ]),
    ("OneSliceIndexNegativeStride", [
        IndexSpec(shape=(10,), indexer=slice(3, 1, -1)),
        IndexSpec(shape=(10,), indexer=slice(1, 8, -1)),  # empty result
        IndexSpec(shape=(10,), indexer=slice(None, 1, -2)),
        IndexSpec(shape=(10,), indexer=slice(None, None, -1)),
        IndexSpec(shape=(10, 8), indexer=slice(3, 1, -1)),
        IndexSpec(shape=(10, 8), indexer=slice(0, 8, -1)),  # empty result
        IndexSpec(shape=(10, 8), indexer=slice(None, None, -1)),
    ]),
    ("OneSliceIndexNonUnitStride", [
        IndexSpec(shape=(10,), indexer=slice(0, 8, 2)),
        IndexSpec(shape=(10,), indexer=slice(0, 8, 3)),
        IndexSpec(shape=(10,), indexer=slice(1, 3, 2)),
        IndexSpec(shape=(10,), indexer=slice(1, None, 2)),
        IndexSpec(shape=(10,), indexer=slice(None, 1, -2)),
        IndexSpec(shape=(10, 8), indexer=slice(1, 8, 3)),
        IndexSpec(shape=(10, 8), indexer=slice(None, None, 2)),
        IndexSpec(shape=(10, 8), indexer=slice(None, 1, -2)),
        IndexSpec(shape=(10, 8), indexer=slice(None, None, -2)),
    ]),
    ("TwoSliceIndices", [
        IndexSpec(shape=(10, 8), indexer=(slice(1, 3), slice(0, 2))),
        IndexSpec(shape=(10, 8), indexer=(slice(1, None), slice(None, 2))),
        IndexSpec(
            shape=(10, 8), indexer=(slice(None, None, -1), slice(None, 2))),
        IndexSpec(shape=(10, 8, 3), indexer=(slice(1, 3), slice(0, 2))),
        IndexSpec(shape=(10, 8, 3), indexer=(slice(1, 3), slice(0, None))),
        IndexSpec(shape=(10, 8, 3), indexer=(slice(1, None), slice(0, 2))),
    ]),
    ("OneColonIndex", [
        IndexSpec(shape=(3,), indexer=slice(None)),
        IndexSpec(shape=(3, 4), indexer=slice(None)),
    ]),
    ("MultipleColonIndices", [
        IndexSpec(shape=(3, 4), indexer=(slice(None), slice(None))),
        IndexSpec(shape=(3, 4, 5), indexer=(slice(None), slice(None))),
    ]),
    ("MixedSliceIndices", [
        IndexSpec(shape=(10, 4), indexer=(slice(None), slice(0, 2))),
        IndexSpec(shape=(10, 4), indexer=(1, slice(None))),
    ]),
    ("EllipsisIndex", [
        IndexSpec(shape=(3,), indexer=Ellipsis),
        IndexSpec(shape=(3, 4), indexer=Ellipsis),
        IndexSpec(shape=(3, 4, 5), indexer=(0, Ellipsis)),
        IndexSpec(shape=(3, 4, 5), indexer=(Ellipsis, 2, 3)),
    ]),
    ("NoneIndex", [
        IndexSpec(shape=(), indexer=None),
        IndexSpec(shape=(), indexer=(None, None)),
        IndexSpec(shape=(), indexer=(Ellipsis, None)),
        IndexSpec(shape=(3,), indexer=None),
        IndexSpec(shape=(3, 4), indexer=None),
        IndexSpec(shape=(3, 4), indexer=(Ellipsis, None)),
        IndexSpec(shape=(3, 4), indexer=(0, None, Ellipsis)),
        IndexSpec(shape=(3, 4, 5), indexer=(1, None, Ellipsis)),
    ]),
    ("EmptyIndex", [
        IndexSpec(shape=(), indexer=()),
        IndexSpec(shape=(3,), indexer=()),
        IndexSpec(shape=(3, 4), indexer=()),
    ]),
]

STATIC_INDEXING_GRAD_TESTS = [
    ("OneIntIndex", [
        IndexSpec(shape=(3,), indexer=1),
        IndexSpec(shape=(3, 3), indexer=0),
        IndexSpec(shape=(3, 4, 5), indexer=2),
        IndexSpec(shape=(3,), indexer=-1),
        IndexSpec(shape=(3,), indexer=-2),
    ]),
    ("TwoIntIndices", [
        IndexSpec(shape=(3, 3), indexer=(2, 1)),
        IndexSpec(shape=(3, 4, 5), indexer=(1, 2)),
        IndexSpec(shape=(3, 4, 5), indexer=(-1, 2)),
    ]),
    ("ThreeIntIndices", [IndexSpec((3, 4, 5), indexer=(1, 2, 3))]),
    ("OneSliceIndex", [
        IndexSpec(shape=(5,), indexer=slice(1, 3)),
        IndexSpec(shape=(5,), indexer=slice(1, -1)),
        IndexSpec(shape=(5,), indexer=slice(None, -1)),
        IndexSpec(shape=(5,), indexer=slice(None, None, None)),
        IndexSpec(shape=(5, 4), indexer=slice(1, 3)),
        IndexSpec(shape=(5, 4), indexer=slice(1, None)),
        IndexSpec(shape=(5, 4), indexer=slice(None, 3)),
        IndexSpec(shape=(5, 4), indexer=slice(-3, None)),
    ]),
    ("TwoSliceIndices", [
        IndexSpec(shape=(5, 4), indexer=(slice(1, 3), slice(0, 2))),
        IndexSpec(shape=(5, 4), indexer=(slice(1, None), slice(None, 2))),
        IndexSpec(shape=(5, 4, 3), indexer=(slice(1, 3), slice(0, 2))),
        IndexSpec(shape=(5, 4, 3), indexer=(slice(1, 3), slice(0, None))),
        IndexSpec(shape=(5, 4, 3), indexer=(slice(1, None), slice(0, 2))),
    ]),
    ("OneColonIndex", [
        IndexSpec(shape=(3,), indexer=slice(None)),
        IndexSpec(shape=(3, 4), indexer=slice(None)),
    ]),
    ("MultipleColonIndices", [
        IndexSpec(shape=(3, 4), indexer=(slice(None), slice(None))),
        IndexSpec(shape=(3, 4, 5), indexer=(slice(None), slice(None))),
    ]),
    ("MixedSliceIndices", [
        IndexSpec(shape=(5, 4), indexer=(slice(None), slice(0, 2))),
        IndexSpec(shape=(5, 4), indexer=(1, slice(None))),
    ]),
    ("EllipsisIndex", [
        IndexSpec(shape=(3,), indexer=Ellipsis),
        IndexSpec(shape=(3, 4), indexer=Ellipsis),
        IndexSpec(shape=(3, 4, 5), indexer=(0, Ellipsis)),
        IndexSpec(shape=(3, 4, 5), indexer=(Ellipsis, 2, 3)),
    ]),
    ("NoneIndex", [
        IndexSpec(shape=(), indexer=None),
        IndexSpec(shape=(), indexer=(None, None)),
        IndexSpec(shape=(), indexer=(Ellipsis, None)),
        IndexSpec(shape=(3,), indexer=None),
        IndexSpec(shape=(3, 4), indexer=None),
        IndexSpec(shape=(3, 4), indexer=(Ellipsis, None)),
        IndexSpec(shape=(3, 4), indexer=(0, None, Ellipsis)),
        IndexSpec(shape=(3, 4, 5), indexer=(1, None, Ellipsis)),
    ]),
    # TODO(mattjj): these fail for uninteresting dtype reasons
    # ("EmptyIndex",
    #  [IndexSpec(shape=(), indexer=()),
    #   IndexSpec(shape=(3,), indexer=()),
    #   IndexSpec(shape=(3, 4), indexer=()),
    #   ]),
]

ADVANCED_INDEXING_TESTS = [
    ("One1DIntArrayIndex",
     [IndexSpec(shape=(3,), indexer=onp.array([0, 1])),
     IndexSpec(shape=(3, 3), indexer=onp.array([1, 2, 1])),
     IndexSpec(shape=(3, 4, 5), indexer=onp.array([0, 2, 0, 1])),
     IndexSpec(shape=(3,), indexer=onp.array([-1, 1])),
     IndexSpec(shape=(3,), indexer=onp.array([-2, -1])),
     IndexSpec(shape=(0,), indexer=onp.array([], dtype=onp.int32)),
     ]),
    ("One2DIntArrayIndex",
     [IndexSpec(shape=(3,), indexer=onp.array([[0, 0]])),
     IndexSpec(shape=(3, 3), indexer=onp.array([[1, 2, 1],
                                                [0, 1, -1]])),
     IndexSpec(shape=(3, 4, 5), indexer=onp.array([[0, 2, 0, 1],
                                                   [-1, -2, 1, 0]])),
     ]),
    ("Two1DIntArrayIndicesNoBroadcasting",
     [IndexSpec(shape=(3, 3), indexer=(onp.array([0, 1]),
                                       onp.array([1, 2]))),
     IndexSpec(shape=(3, 4, 5), indexer=(onp.array([0, 2, 0, 1]),
                                         onp.array([-1, 0, -1, 2]))),
     ]),
    ("Two1DIntArrayIndicesWithBroadcasting",
     [IndexSpec(shape=(3, 3), indexer=(onp.array([[0, 1]]),
                                       onp.array([1, 2]))),
     IndexSpec(shape=(3, 4, 5), indexer=(onp.array([[0, 2, 0, 1]]),
                                         onp.array([-1, 0, -1, 2]))),
     ]),
    ("TupleOfListsOfPythonInts",
     [IndexSpec(shape=(3, 4, 5), indexer=([0, 1])),
     IndexSpec(shape=(3, 4, 5), indexer=([[0], [-1]], [[2, 3, 0, 3]])),
     ]),
    ("TupleOfPythonIntsAndIntArrays",
     [IndexSpec(shape=(3, 4, 5), indexer=(0, onp.array([0, 1]))),
     IndexSpec(shape=(3, 4, 5), indexer=(0, 1,
                                         onp.array([[2, 3, 0, 3]]))),
     ]),
    ("TupleOfListsOfPythonIntsAndIntArrays",
     [IndexSpec(shape=(3, 4, 5), indexer=([0, 1], onp.array([0]))),
     IndexSpec(shape=(3, 4, 5), indexer=([[0], [-1]],
                                         onp.array([[2, 3, 0, 3]]))),
     ]),
]

ADVANCED_INDEXING_TESTS_NO_REPEATS = [
    ("One1DIntArrayIndex",
     [IndexSpec(shape=(3,), indexer=onp.array([0, 1])),
      IndexSpec(shape=(3, 3), indexer=onp.array([1, 2, 0])),
      IndexSpec(shape=(3, 4, 5), indexer=onp.array([0, 2, 1])),
      IndexSpec(shape=(3,), indexer=onp.array([-1, 1])),
      IndexSpec(shape=(3,), indexer=onp.array([-2, -1])),
      # Fails with a TF/XLA error.
      # IndexSpec(shape=(0,), indexer=onp.array([], dtype=onp.int32)),
     ]),
    ("One2DIntArrayIndex",
     [IndexSpec(shape=(3,), indexer=onp.array([[0, 1]])),
      IndexSpec(shape=(6, 6), indexer=onp.array([[1, 2, 0],
                                                 [3, 4, -1]])),
     ]),
    ("Two1DIntArrayIndicesNoBroadcasting",
     [IndexSpec(shape=(3, 3), indexer=(onp.array([0, 1]),
                                       onp.array([1, 2]))),
      IndexSpec(shape=(4, 5, 6), indexer=(onp.array([0, 2, 1, 3]),
                                          onp.array([-1, 0, -2, 1]))),
     ]),
    ("Two1DIntArrayIndicesWithBroadcasting",
     [IndexSpec(shape=(3, 3), indexer=(onp.array([[0, 1]]),
                                       onp.array([1, 2]))),
      IndexSpec(shape=(4, 5, 6), indexer=(onp.array([[0, 2, -1, 1]]),
                                          onp.array([-1, 0, -2, 2]))),
     ]),
    ("TupleOfListsOfPythonInts",
     [IndexSpec(shape=(3, 4, 5), indexer=([0, 1])),
      IndexSpec(shape=(3, 4, 5), indexer=([[0], [-1]], [[2, 3, 0]])),
     ]),
    ("TupleOfPythonIntsAndIntArrays",
     [IndexSpec(shape=(3, 4, 5), indexer=(0, onp.array([0, 1]))),
      IndexSpec(shape=(3, 4, 5), indexer=(0, 1,
                                          onp.array([[2, 3, 0]]))),
     ]),
    ("TupleOfListsOfPythonIntsAndIntArrays",
     [IndexSpec(shape=(3, 4, 5), indexer=([0, 1], onp.array([0]))),
      IndexSpec(shape=(3, 4, 5), indexer=([[0], [-1]],
                                          onp.array([[2, 3, 0]]))),
     ]),
]

MIXED_ADVANCED_INDEXING_TESTS_NO_REPEATS = [
    ("SlicesAndOneIntArrayIndex",
     [IndexSpec(shape=(2, 3), indexer=(onp.array([0, 1]), slice(1, 2))),
     IndexSpec(shape=(2, 3), indexer=(slice(0, 2),
                                      onp.array([0, 2]))),
     IndexSpec(shape=(3, 4, 5), indexer=(Ellipsis,
                                         onp.array([0, 2]),
                                         slice(None))),
     IndexSpec(shape=(3, 4, 5), indexer=(Ellipsis,
                                         onp.array([[0, 2], [1, 3]]),
                                         slice(None))),
     ]),
    ("SlicesAndTwoIntArrayIndices",
     [IndexSpec(shape=(3, 4, 5), indexer=(Ellipsis,
                                          onp.array([0, 2]),
                                          onp.array([-1, 2]))),
     IndexSpec(shape=(3, 4, 5), indexer=(onp.array([0, 2]),
                                         Ellipsis,
                                         onp.array([-1, 2]))),
     IndexSpec(shape=(3, 4, 5), indexer=(onp.array([0, 2]),
                                         onp.array([-1, 2]),
                                         Ellipsis)),
     IndexSpec(shape=(3, 4, 5), indexer=(onp.array([0, 2]),
                                         onp.array([-1, 2]),
                                         slice(1, 3))),
     IndexSpec(shape=(3, 4, 5), indexer=(onp.array([0, 2]),
                                         slice(1, 3),
                                         onp.array([-1, 2]))),
     IndexSpec(shape=(3, 4, 5), indexer=(onp.array([0, 2, -2]),
                                         slice(None, None, 2),
                                         onp.array([-1, 2, 1]))),
     ]),
    ("NonesAndIntArrayIndices",
     [IndexSpec(shape=(3, 4, 5), indexer=(onp.array([0, 2]),
                                          None,
                                          onp.array([-1, 2]))),
     IndexSpec(shape=(3, 4, 5), indexer=(onp.array([0, 2]),
                                         None,
                                         None,
                                         onp.array([-1, 2]))),
     IndexSpec(shape=(3, 4, 5), indexer=(Ellipsis,
                                         onp.array([0, 2]),
                                         None,
                                         None,
                                         onp.array([-1, 2]))),
     ]),
    ("IntArrayWithInt32Type",
     [IndexSpec(shape=(3, 4), indexer=(Ellipsis, onp.array(1, dtype=onp.int32)))
     ]),
]

MIXED_ADVANCED_INDEXING_TESTS = MIXED_ADVANCED_INDEXING_TESTS_NO_REPEATS + [
    ("SlicesAndOneIntArrayIndex",
     [
     IndexSpec(shape=(3, 4, 5), indexer=(Ellipsis,
                                         onp.array([[0, 2], [1, 1]]),
                                         slice(None))),
     ]),
    ("SlicesAndTwoIntArrayIndices",
     [IndexSpec(shape=(3, 4, 5), indexer=(onp.array([0, 2, -2]),
                                         slice(None, None, 2),
                                         onp.array([-1, 2, -1]))),
      IndexSpec(shape=(3, 4, 5), indexer=(onp.array([[0, 2], [2, 0]]),
                                          Ellipsis,
                                          onp.array([[1, 0], [1, 0]]))),
     ]),]


def dynamic_slice_reference(operand, start_indices, slice_sizes):
  out = onp.zeros(slice_sizes, dtype=operand.dtype)
  idx = tuple(slice(start, start+size)
              for start, size in zip(start_indices, slice_sizes))
  section = operand[idx]
  out[tuple(slice(None, stop) for stop in section.shape)] = section
  return out


def dynamic_update_slice_reference(operand, update, start_indices):
  slices = tuple(map(
      slice, start_indices, onp.add(start_indices, update.shape)))
  updated_operand = onp.copy(operand)
  updated_operand[slices] = update
  return updated_operand


class IndexingTest(jtu.TestCase):
  """Tests for Numpy indexing translation rules."""

  @parameterized.named_parameters(jtu.cases_from_list({
      "testcase_name": "{}_inshape={}_indexer={}".format(
          name, jtu.format_shape_dtype_string( shape, dtype), indexer),
       "shape": shape, "dtype": dtype, "rng_factory": rng_factory, "indexer": indexer
  } for name, index_specs in STATIC_INDEXING_TESTS
    for shape, indexer in index_specs
    for dtype in all_dtypes
    for rng_factory in [jtu.rand_default]))
  def testStaticIndexing(self, shape, dtype, rng_factory, indexer):
    # TODO(rohanj): Revisit passing in self.rng() to this to customize further.
    # This would need updating lax_numpy_test as well.
    rng = rng_factory()
    args_maker = lambda: [rng(shape, dtype)]
    onp_fun = lambda x: x[indexer]
    jnp_fun = lambda x: tnp.asarray(x)[indexer]
    self._CheckAgainstNumpy(onp_fun, jnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(jnp_fun, args_maker, check_dtypes=True,
                          check_incomplete_shape=True)

  def _ReplaceSlicesWithTuples(self, idx):
    """Helper method to replace slices with tuples for dynamic indexing args."""
    if isinstance(idx, slice):
      triple = idx.start, idx.stop, idx.step
      isnone = [i for i, elt in enumerate(triple) if elt is None]
      zeros = itertools.repeat(0)
      nones = itertools.repeat(None)
      out = subvals(triple, zip(isnone, zeros))
      return out, lambda out: slice(*subvals(out, zip(isnone, nones)))
    elif isinstance(idx, (tuple, list)) and idx:
      t = type(idx)
      elts, packs = zip(*map(self._ReplaceSlicesWithTuples, idx))
      return elts, lambda elts: t((pack(i) for pack, i in zip(packs, elts)))
    else:
      return idx, lambda x: x

  @parameterized.named_parameters(
      {"testcase_name": "{}_inshape={}_indexer={}"
       .format(name, jtu.format_shape_dtype_string(shape, dtype), indexer),
       "shape": shape, "dtype": dtype, "rng_factory": rng_factory, "indexer": indexer}
      for name, index_specs in [
          ("OneSliceIndex",
           [IndexSpec(shape=(5,), indexer=slice(1, 3)),
            IndexSpec(shape=(5, 4), indexer=slice(1, 3))]),
          ("TwoSliceIndices",
           [IndexSpec(shape=(5, 4), indexer=(slice(1, 3), slice(0, 2))),
            IndexSpec(shape=(5, 4, 3), indexer=(slice(1, 3), slice(0, 2)))]),
          ("NonUnitStrides", [
              IndexSpec(shape=(3,), indexer=slice(None, None, -1)),
              IndexSpec(shape=(3, 3), indexer=slice(0, 3, -2)),
              IndexSpec(shape=(3, 4, 5), indexer=slice(0, 4, 2))
          ]),
          ("OnlyStartOrStopDynamic", [
              IndexSpec(shape=(5, 4), indexer=(slice(None, 3), slice(0, 2))),
              IndexSpec(shape=(5, 4, 3), indexer=(slice(1, 3), slice(0, None)))
          ]),
      ]
      for shape, indexer in index_specs
      for dtype in all_dtypes
      for rng_factory in [jtu.rand_default])
  def testDynamicIndexingWithSlices(self, shape, dtype, rng_factory, indexer):
    rng = rng_factory()
    unpacked_indexer, pack_indexer = self._ReplaceSlicesWithTuples(indexer)

    def onp_fun(x, unpacked_indexer):
      indexer = pack_indexer(unpacked_indexer)
      return x[indexer]

    jnp_fun = lambda x, idx: onp_fun(tnp.asarray(x), idx)

    args_maker = lambda: [rng(shape, dtype), unpacked_indexer]
    self._CheckAgainstNumpy(onp_fun, jnp_fun, args_maker, check_dtypes=True)
    # TODO(wangpeng): check_xla_forced_compile is turned off because some
    # compile-time-constant requirements are violated. Investigate and turn it
    # on.
    self._CompileAndCheck(jnp_fun, args_maker, check_dtypes=True,
                          check_eval_on_shapes=False,
                          check_incomplete_shape=True,
                          check_xla_forced_compile=False)

  @parameterized.named_parameters(
      {"testcase_name": "{}_inshape={}_indexer={}"
       .format(name, jtu.format_shape_dtype_string(shape, dtype), indexer),
       "shape": shape, "dtype": dtype, "rng_factory": rng_factory, "indexer": indexer}
      for name, index_specs in [
          ("OneIntIndex",
           [IndexSpec(shape=(3,), indexer=1),
            IndexSpec(shape=(3, 3), indexer=0),
            IndexSpec(shape=(3, 4, 5), indexer=2),
            IndexSpec(shape=(3,), indexer=-1),
            IndexSpec(shape=(3,), indexer=-2)]),
          ("TwoIntIndices",
           [IndexSpec(shape=(3, 3), indexer=(2, 1)),
            IndexSpec(shape=(3, 4, 5), indexer=(1, 2)),
            IndexSpec(shape=(3, 4, 5), indexer=(-1, 2))]),
          ("ThreeIntIndices",
           [IndexSpec((3, 4, 5), indexer=(1, 2, 3))]),
      ]
      for shape, indexer in index_specs
      for dtype in all_dtypes
      for rng_factory in [jtu.rand_default])
  def testDynamicIndexingWithIntegers(self, shape, dtype, rng_factory, indexer):
    # TODO(rohanj): Revisit passing in self.rng() to this to customize further.
    # This would need updating lax_numpy_test as well.
    rng = rng_factory()
    unpacked_indexer, pack_indexer = self._ReplaceSlicesWithTuples(indexer)

    def onp_fun(x, unpacked_indexer):
      indexer = pack_indexer(unpacked_indexer)
      return x[indexer]

    jnp_fun = lambda x, idx: onp_fun(tnp.asarray(x), idx)

    args_maker = lambda: [rng(shape, dtype), unpacked_indexer]
    self._CheckAgainstNumpy(onp_fun, jnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(jnp_fun, args_maker, check_dtypes=True,
                          check_incomplete_shape=True)

  @parameterized.named_parameters(
      {"testcase_name": "_{}_inshape={}_indexer={}"  # pylint: disable=g-complex-comprehension
       .format(name, jtu.format_shape_dtype_string(shape, dtype), indexer),
       "name": name, "shape": shape, "dtype": dtype, "rng_factory": rng_factory,
       "indexer": indexer}
      for name, index_specs in ADVANCED_INDEXING_TESTS
      for shape, indexer in index_specs
      for dtype in all_dtypes
      for rng_factory in [jtu.rand_default])
  def testAdvancedIntegerIndexing(self, name, shape, dtype, rng_factory,
                                  indexer):
    rng = rng_factory()
    args_maker = lambda: [rng(shape, dtype), indexer]
    onp_fun = lambda x, idx: x[idx]
    jnp_fun = lambda x, idx: onp_fun(tnp.asarray(x), idx)

    self._CheckAgainstNumpy(onp_fun, jnp_fun, args_maker, check_dtypes=True)
    # TODO(wangpeng): check_xla_forced_compile is turned off for
    # ListOfPythonIntsAndIntArrays because it throws "The number of output
    # elements has to equal to number of input elements that are sliced when
    # input indices are not constant". Investigate and turn it on.
    check_xla = (name != "ListOfPythonIntsAndIntArrays")
    self._CompileAndCheck(jnp_fun, args_maker, check_dtypes=True,
                          check_incomplete_shape=True,
                          check_xla_forced_compile=check_xla)

  @parameterized.named_parameters(
      {"testcase_name": "_{}_inshape={}_indexer={}"  # pylint: disable=g-complex-comprehension
       .format(name, jtu.format_shape_dtype_string(shape, dtype), indexer),
       "name": name, "shape": shape, "dtype": dtype, "rng_factory": rng_factory,
       "indexer": indexer}
      for name, index_specs in MIXED_ADVANCED_INDEXING_TESTS
      for shape, indexer in index_specs
      for dtype in all_dtypes
      for rng_factory in [jtu.rand_default])
  def testMixedAdvancedIntegerIndexing(self, name, shape, dtype, rng_factory,
                                       indexer):
    rng = rng_factory()
    indexer_with_dummies = [e if isinstance(e, onp.ndarray) else ()
                            for e in indexer]
    substitutes = [(i, e) for i, e in enumerate(indexer)
                   if not isinstance(e, onp.ndarray)]
    args_maker = lambda: [rng(shape, dtype), indexer_with_dummies]

    def np_fun(x, indexer_with_dummies):
      idx = type(indexer)(subvals(indexer_with_dummies, substitutes))
      return x[idx]

    jnp_fun = lambda x, idx: np_fun(tnp.asarray(x), idx)

    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=True)
    # TODO(wangpeng): check_xla_forced_compile is turned off for
    # IntArrayWithInt32Type because it throws "The number of output elements has
    # to equal to number of input elements that are sliced when input indices
    # are not constant". Investigate and turn it on.
    check_xla = (name != "IntArrayWithInt32Type")
    self._CompileAndCheck(jnp_fun, args_maker, check_dtypes=True,
                          check_incomplete_shape=True,
                          check_xla_forced_compile=check_xla)

  def testAdvancedIndexingManually(self):
    x = onp.random.RandomState(0).randn(3, 4, 5)
    index_array = onp.array([0, 2, -1, 0])

    op = lambda x, index_array: x[..., index_array, :]
    cop = nje.jit(op)

    a1 = op(x, index_array)
    a2 = cop(x, index_array)

    self.assertAllClose(a1, a2, check_dtypes=True)

    op = lambda x, index_array: x[..., index_array, :, index_array, None]
    cop = nje.jit(op)

    a1 = op(x, index_array)
    a2 = cop(x, index_array)

    self.assertAllClose(a1, a2, check_dtypes=True)

    op = lambda x, index_array: x[index_array, ..., index_array[:, None], None]
    cop = nje.jit(op)

    a1 = op(x, index_array)
    a2 = cop(x, index_array)

    self.assertAllClose(a1, a2, check_dtypes=True)

  # Note that we don't currently allow __iter__ in graph mode. So this test only
  # iterates over eager tensor.
  def testUnpacking(self):

    def foo(x):
      a, b, c = x
      return a + b + c

    a1 = foo(onp.arange(3))
    a2 = foo(tnp.arange(3))

    self.assertAllClose(a1, a2, check_dtypes=True)

  def testBooleanIndexingArray1D(self):
    idx = onp.array([True, True, False])
    x = tnp.asarray(onp.arange(3))
    ans = x[idx]
    expected = onp.arange(3)[idx]
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testBooleanIndexingList1D(self):
    idx = [True, True, False]
    x = tnp.asarray(onp.arange(3))
    ans = x[idx]
    expected = onp.arange(3)[idx]
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testBooleanIndexingArray2DBroadcast(self):
    idx = onp.array([True, True, False, True])
    x = onp.arange(8).reshape(4, 2)
    ans = tnp.asarray(x)[idx]
    expected = x[idx]
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testBooleanIndexingList2DBroadcast(self):
    idx = [True, True, False, True]
    x = onp.arange(8).reshape(4, 2)
    ans = tnp.asarray(x)[idx]
    expected = x[idx]
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testBooleanIndexingArray2D(self):
    idx = onp.array([[True, False],
                     [False, True],
                     [False, False],
                     [True, True]])
    x = onp.arange(8).reshape(4, 2)
    ans = tnp.asarray(x)[idx]
    expected = x[idx]
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testBooleanIndexingDynamicShape(self):
    x = onp.zeros(3)
    i = onp.array([True, True, False])
    ans = x[i]
    expected = tnp.asarray(x)[i]
    self.assertAllClose(ans, expected, check_dtypes=True)

  def testIssue187(self):
    x = tnp.ones((5, 5))
    x[[0, 2, 4], [0, 2, 4]]  # doesn't crash

    x = onp.arange(25).reshape((5, 5))
    ans = nje.jit(lambda x: x[[0, 2, 4], [0, 2, 4]])(x)
    expected = x[[0, 2, 4], [0, 2, 4]]
    self.assertAllClose(ans, expected, check_dtypes=False)

  # TODO(agarwal): Fix this use case.
  @jtu.disable
  def testIndexingEmptyDimension(self):
    # Issue 2671: XLA error when indexing into dimension of size 0
    x = tnp.ones((2, 0))
    # The following work, even on axis 1 of size 0
    _ = x[0, :] + x[0, None] + x[0, 1:] + x[0, 1:3:2]

    with self.assertRaisesRegex(IndexError,
                                "index .* is out of bounds for axis .* with size 0"):
      _ = onp.ones((2, 0))[0, 0]  # The numpy error
    with self.assertRaisesRegex(IndexError,
                                "index is out of bounds for axis .* with size 0"):
      _ = x[0, 0]  # JAX indexing
    with self.assertRaisesRegex(IndexError,
                                "index is out of bounds for axis .* with size 0"):
      nje.jit(lambda i: x[0, i])(0)  # JAX indexing under jit

  def testBooleanIndexingWithEmptyResult(self):
    # based on a TensorFlow Probability test that started failing after #1623
    x = tnp.array([-1])
    mask = tnp.array([False])
    ans = x[mask]  # doesn't crash

    expected =  onp.array([-1])[onp.array([False])]
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testFloatIndexingError(self):
    error_regex = "only integers, slices.*are valid indices"
    # Verify onp behavior
    with self.assertRaisesRegex(IndexError, error_regex):
      _ = onp.zeros((2, 2))[(0, 0.)]
    # Test tnp
    with self.assertRaisesRegex(IndexError, error_regex):
      tnp.zeros(2)[0.]  # pylint: disable=expression-not-assigned
    with self.assertRaisesRegex(IndexError, error_regex):
      tnp.zeros((2, 2))[(0, 0.)]  # pylint: disable=expression-not-assigned
    # Test with jit
    with self.assertRaisesRegex(IndexError, error_regex):
      nje.jit(lambda idx: tnp.zeros((2, 2))[idx])((0, 0.0))

  def testIndexOutOfBounds(self):  # https://github.com/google/jax/issues/2245
    array = tnp.ones(5)
    self.assertAllClose(array, array[:10], check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_start_indices={}_size_indices={}".format(  # pylint: disable=g-complex-comprehension
          jtu.format_shape_dtype_string(shape, dtype),
          start_indices, size_indices),
       "shape": shape, "dtype": dtype, "start_indices": start_indices,
       "size_indices": size_indices, "rng_factory": rng_factory}
      for shape, start_indices, size_indices in [
        [(3,), onp.array((1,)), (1,)],
        [(5, 3), (1, 1), (3, 1)],
        [(5, 3), (1, -2), (3, 1)],
        [(5, 3), onp.array((1, 1)), (3, 1)],
        [(7, 5, 3), onp.array((4, 1, 0)), (2, 0, 1)],
        [(), (), ()],
      ]
      for dtype in default_dtypes
      for rng_factory in [jtu.rand_default]))
  def testDynamicSlice(self, shape, dtype, start_indices, size_indices,
                       rng_factory):
    rng = rng_factory()
    args_maker = lambda: [rng(shape, dtype), onp.array(start_indices)]
    op = lambda x, starts: nje.dynamic_slice(x, starts, size_indices)
    self._CompileAndCheck(op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_start_indices={}_size_indices={}".format(  # pylint: disable=g-complex-comprehension
          jtu.format_shape_dtype_string(shape, dtype),
          start_indices, size_indices),
       "shape": shape, "dtype": dtype, "start_indices": start_indices,
       "size_indices": size_indices, "rng_factory": rng_factory}
      for shape, start_indices, size_indices in [
        [(3,), (1,), (1,)],
        [(5, 3), (1, 1), (3, 1)],
        [(5, 3), (1, -2), (3, 1)],
        [(7, 5, 3), (4, 1, 0), (2, 0, 1)],
        [(), (), ()],
      ]
      for dtype in default_dtypes
      for rng_factory in [jtu.rand_default]))
  def testDynamicSliceAgainstNumpy(self, shape, dtype, start_indices,
                                   size_indices, rng_factory):
    rng = rng_factory()
    args_maker = lambda: [rng(shape, dtype), onp.array(start_indices)]
    op = lambda x, s: nje.dynamic_slice(x, s, size_indices)
    numpy_op = lambda x, s: dynamic_slice_reference(x, s, size_indices)
    self._CheckAgainstNumpy(numpy_op, op, args_maker)

  def testDynamicSliceInDim(self):
    rng = jtu.rand_default()
    x = rng((6, 7), onp.int32)
    self.assertAllClose(
        nje.dynamic_slice_in_dim(x, 2, 3), x[2:5], check_dtypes=True
    )


def _broadcastable_shapes(shape):
  """Returns all shapes that broadcast to `shape`."""
  def f(rshape):
    yield []
    if rshape:
      for s in f(rshape[1:]):
        yield rshape[0:1] + s
      if rshape[0] != 1:
        for s in f(rshape[1:]):
          yield [1] + s
  for x in f(list(reversed(shape))):
    yield list(reversed(x))


def _update_shape(shape, indexer):
  return onp.zeros(shape)[indexer].shape


class UpdateOps(enum.Enum):
  UPDATE = 0
  ADD = 1
  # MUL = 2
  MIN = 3
  MAX = 4

  def np_fn(op, indexer, x, y):  # pylint: disable=no-self-argument
    x = x.copy()
    x[indexer] = {
      UpdateOps.UPDATE: lambda: y,
      UpdateOps.ADD: lambda: x[indexer] + y,
      # UpdateOps.MUL: lambda: x[indexer] * y,
      UpdateOps.MIN: lambda: onp.minimum(x[indexer], y),
      UpdateOps.MAX: lambda: onp.maximum(x[indexer], y),
    }[op]()
    return x

  def tfnp_fn(op, indexer, x, y):  # pylint: disable=no-self-argument
    return {
        UpdateOps.UPDATE: nje.index_update,
        UpdateOps.ADD: nje.index_add,
        # UpdateOps.MUL: nje.index_mul,
        UpdateOps.MIN: nje.index_min,
        UpdateOps.MAX: nje.index_max,
    }[op](x, indexer, y)


# a test to workaround b/123559667
def has_non_trivial_stride(indexer):
  def has(idx):
    return isinstance(idx, slice) and idx.step not in (1, -1, None)
  return any(has(idx) for idx in nest.flatten(indexer))


class IndexedUpdateTest(jtu.TestCase):

  @parameterized.named_parameters(jtu.cases_from_list({  # pylint: disable=g-complex-comprehension
      "testcase_name": "_{}_{}_{}_{}".format(
          jtu.format_shape_dtype_string(shape, dtype), indexer,
          jtu.format_shape_dtype_string(update_shape, update_dtype), op.name),
      "shape": shape, "dtype": dtype, "rng_factory": rng_factory,
      "indexer": indexer, "update_shape": update_shape,
      "update_dtype": update_dtype, "op": op
  } for name, index_specs in STATIC_INDEXING_TESTS
    for shape, indexer in index_specs
    for op in UpdateOps
    for dtype in (all_dtypes if op == UpdateOps.UPDATE else default_dtypes)
    for update_shape in _broadcastable_shapes(_update_shape(shape, indexer))
    for update_dtype in all_dtypes
    for rng_factory in [jtu.rand_default]))
  def testStaticIndexing(self, shape, dtype, update_shape, update_dtype,
                         rng_factory, indexer, op):
    rng = rng_factory()
    args_maker = lambda: [rng(shape, dtype), rng(update_shape, update_dtype)]
    np_fn = lambda x, y: UpdateOps.np_fn(op, indexer, x, y)
    tfnp_fn = lambda x, y: UpdateOps.tfnp_fn(op, indexer, x, y)
    self._CheckAgainstNumpy(np_fn, tfnp_fn, args_maker)
    # TODO(wangpeng): When indexer is slice(_, 8, -1), XLA throws error "Missing
    # xla_context 0-th output from". Investigate.
    check_xla = (not has_non_trivial_stride(indexer) and  # b/123559667
                 not (isinstance(indexer, slice) and indexer.stop == 8 and
                      indexer.step == -1))
    self._CompileAndCheck(tfnp_fn, args_maker, check_incomplete_shape=True,
                          check_experimental_compile=check_xla,
                          check_xla_forced_compile=check_xla)

  @parameterized.named_parameters(jtu.cases_from_list({  # pylint: disable=g-complex-comprehension
      "testcase_name": "_{}_{}_{}_{}".format(
          jtu.format_shape_dtype_string(shape, dtype), indexer,
          jtu.format_shape_dtype_string(update_shape, update_dtype), op.name),
      "shape": shape, "dtype": dtype, "rng_factory": rng_factory,
      "indexer": indexer, "update_shape": update_shape,
      "update_dtype": update_dtype, "op": op
  } for name, index_specs in ADVANCED_INDEXING_TESTS_NO_REPEATS
    for shape, indexer in index_specs
    for op in UpdateOps
    for dtype in (all_dtypes if op == UpdateOps.UPDATE else default_dtypes)
    for update_shape in _broadcastable_shapes(_update_shape(shape, indexer))
    for update_dtype in all_dtypes
    for rng_factory in [jtu.rand_default]))
  def testAdvancedIndexing(self, shape, dtype, update_shape, update_dtype,
                           rng_factory, indexer, op):
    rng = rng_factory()
    args_maker = lambda: [rng(shape, dtype), rng(update_shape, update_dtype)]
    np_fn = lambda x, y: UpdateOps.np_fn(op, indexer, x, y)
    tfnp_fn = lambda x, y: UpdateOps.tfnp_fn(op, indexer, x, y)
    self._CheckAgainstNumpy(np_fn, tfnp_fn, args_maker)
    self._CompileAndCheck(tfnp_fn, args_maker, check_incomplete_shape=True)

  @parameterized.named_parameters(jtu.cases_from_list({  # pylint: disable=g-complex-comprehension
      "testcase_name": "_{}_{}_{}_{}".format(
          jtu.format_shape_dtype_string(shape, dtype), indexer,
          jtu.format_shape_dtype_string(update_shape, update_dtype), op.name),
      "shape": shape, "dtype": dtype, "rng_factory": rng_factory,
      "indexer": indexer, "update_shape": update_shape,
      "update_dtype": update_dtype, "op": op
  } for name, index_specs in MIXED_ADVANCED_INDEXING_TESTS_NO_REPEATS
    for shape, indexer in index_specs
    for op in UpdateOps
    for dtype in (all_dtypes if op == UpdateOps.UPDATE else default_dtypes)
    for update_shape in _broadcastable_shapes(_update_shape(shape, indexer))
    for update_dtype in all_dtypes
    for rng_factory in [jtu.rand_default]))
  def testMixedAdvancedIndexing(self, shape, dtype, update_shape, update_dtype,
                                rng_factory, indexer, op):
    rng = rng_factory()
    args_maker = lambda: [rng(shape, dtype), rng(update_shape, update_dtype)]
    np_fn = lambda x, y: UpdateOps.np_fn(op, indexer, x, y)
    tfnp_fn = lambda x, y: UpdateOps.tfnp_fn(op, indexer, x, y)
    self._CheckAgainstNumpy(np_fn, tfnp_fn, args_maker)
    check_xla = not has_non_trivial_stride(indexer)  # b/123559667
    self._CompileAndCheck(tfnp_fn, args_maker, check_incomplete_shape=True,
                          check_experimental_compile=check_xla,
                          check_xla_forced_compile=check_xla)

  @parameterized.named_parameters(jtu.cases_from_list({  # pylint: disable=g-complex-comprehension
      "testcase_name": "_{}_{}_{}_{}".format(
          jtu.format_shape_dtype_string(shape, dtype), indexer,
          jtu.format_shape_dtype_string(update_shape, update_dtype), op.name),
      "shape": shape, "dtype": dtype, "rng_factory": rng_factory,
      "indexer": indexer, "update_shape": update_shape,
      "update_dtype": update_dtype, "op": op
  } for name, index_specs in STATIC_INDEXING_TESTS
    for shape, indexer in index_specs
    for op in [UpdateOps.ADD, UpdateOps.UPDATE]
    for dtype in float_dtypes
    for update_shape in _broadcastable_shapes(_update_shape(shape, indexer))
    for update_dtype in float_dtypes
    for rng_factory in [jtu.rand_default]))
  def testStaticIndexingGrads(self, shape, dtype, update_shape, update_dtype,
                              rng_factory, indexer, op):
    rng = rng_factory()
    tfnp_fn = lambda x, y: UpdateOps.tfnp_fn(op, indexer, x, y)
    x = rng(shape, dtype)
    y = rng(update_shape, update_dtype)
    self.check_grads(tfnp_fn, (x, y), rtol=1e-3, atol=1e-3, delta=1.)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_start_indices={}_update_shape={}".format(  # pylint: disable=g-complex-comprehension
          jtu.format_shape_dtype_string(shape, dtype),
          start_indices, update_shape),
       "shape": shape, "dtype": dtype, "start_indices": start_indices,
       "update_shape": update_shape, "rng_factory": rng_factory}
      for shape, start_indices, update_shape in [
        [(3,), (1,), (1,)],
        [(5, 3), (1, 1), (3, 1)],
        [(5, 3), (1, -2), (3, 1)],
        [(7, 5, 3), (4, 1, 0), (2, 0, 1)],
        [(), (), ()],
      ]
      for dtype in default_dtypes
      for rng_factory in [jtu.rand_default]))
  def testDynamicUpdateSlice(self, shape, dtype, start_indices, update_shape,
                             rng_factory):
    rng = rng_factory()
    def args_maker():
      return [rng(shape, dtype), rng(update_shape, dtype),
              onp.array(start_indices)]
    # update's shape must be fully known.
    # TODO(wangpeng): Support turning off check_incomplete_shape for individual
    #   arguments.
    self._CompileAndCheck(
        nje.dynamic_update_slice, args_maker, check_incomplete_shape=False
    )

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_start_indices={}_update_shape={}".format(  # pylint: disable=g-complex-comprehension
          jtu.format_shape_dtype_string(shape, dtype),
          start_indices, update_shape),
       "shape": shape, "dtype": dtype, "start_indices": start_indices,
       "update_shape": update_shape, "rng_factory": rng_factory}
      for shape, start_indices, update_shape in [
        [(3,), (1,), (1,)],
        [(5, 3), (1, 1), (3, 1)],
        [(5, 3), (1, -2), (3, 1)],
        [(7, 5, 3), (4, 1, 0), (2, 0, 1)],
        [(), (), ()],
      ]
      for dtype in default_dtypes
      for rng_factory in [jtu.rand_default]))
  def testDynamicUpdateSliceAgainstNumpy(self, shape, dtype, start_indices,
                                         update_shape, rng_factory):
    rng = rng_factory()
    def args_maker():
      return [rng(shape, dtype), rng(update_shape, dtype),
              onp.array(start_indices)]
    self._CheckAgainstNumpy(
        dynamic_update_slice_reference, nje.dynamic_update_slice, args_maker
    )

  def testDynamicUpdateSliceInDim(self):
    rng = jtu.rand_default()
    x = rng((6, 7), onp.int32)
    y = rng((3, 7), onp.int32)
    z = x.copy()
    z[2:5] = y
    self.assertAllClose(
        nje.dynamic_update_slice_in_dim(x, y, 2, 0), z, check_dtypes=True
    )


if __name__ == "__main__":
  tf_config.set_soft_device_placement(False)
  absltest.main()
