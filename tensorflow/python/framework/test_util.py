# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

# pylint: disable=invalid-name
"""Test utils for tensorflow."""

import collections
from collections import OrderedDict
from collections.abc import Callable, Iterator
import contextlib
import functools
import gc
import itertools
import math
import os
import random
import re
import tempfile
import threading
import time
from typing import Any, cast, Optional, overload, TypeVar, Union
import unittest

from absl.testing import parameterized
import numpy as np

from google.protobuf import descriptor_pool
from google.protobuf import text_format
from tensorflow.core.config import flags
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import pywrap_sanitizers
from tensorflow.python import tf2
from tensorflow.python.client import device_lib
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.client import session as s
from tensorflow.python.compat import v2_compat
from tensorflow.python.compat.compat import forward_compatibility_horizon
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import _test_metrics_util
from tensorflow.python.framework import config
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import gpu_util
from tensorflow.python.framework import importer
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import tfrt_utils
from tensorflow.python.framework import versions
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_util_v2
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import gen_sync_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variables


from tensorflow.python.ops.ragged import ragged_ops  # pylint: disable=unused-import
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.platform import _pywrap_stacktrace_handler
from tensorflow.python.platform import googletest
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import server_lib
from tensorflow.python.util import _pywrap_util_port
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.protobuf import compare
from tensorflow.python.util.tf_export import tf_export

_TC = TypeVar("_TC", bound=type["TensorFlowTestCase"])
_R = TypeVar("_R")


# If the below import is made available through the BUILD rule, then this
# function is overridden and will instead return True and cause Tensorflow
# graphs to be compiled with XLA.
def is_xla_enabled():
  return False


try:
  from tensorflow.python.framework.is_xla_test_true import is_xla_enabled  # pylint: disable=g-import-not-at-top, unused-import
except Exception:  # pylint: disable=broad-except
  pass


# Uses the same mechanism as above to selectively enable/disable MLIR
# compilation.
def is_mlir_bridge_enabled():
  return None


try:
  from tensorflow.python.framework.is_mlir_bridge_test_false import is_mlir_bridge_enabled  # pylint: disable=g-import-not-at-top, unused-import
except ImportError:
  try:
    from tensorflow.python.framework.is_mlir_bridge_test_true import is_mlir_bridge_enabled  # pylint: disable=g-import-not-at-top, unused-import
  except ImportError:
    pass


def is_asan_enabled():
  """Check if ASAN is enabled."""
  return pywrap_sanitizers.is_asan_enabled()


def is_msan_enabled():
  """Check if MSAN is enabled."""
  return pywrap_sanitizers.is_msan_enabled()


def is_tsan_enabled():
  """Check if TSAN is enabled."""
  return pywrap_sanitizers.is_tsan_enabled()


def is_ubsan_enabled():
  """Check if UBSAN is enabled."""
  return pywrap_sanitizers.is_ubsan_enabled()


def _get_object_count_by_type(exclude=()):
  return (
      collections.Counter([type(obj).__name__ for obj in gc.get_objects()]) -
      collections.Counter([type(obj).__name__ for obj in exclude]))


@tf_export("test.gpu_device_name")
def gpu_device_name():
  """Returns the name of a GPU device if available or a empty string.

  This method should only be used in tests written with `tf.test.TestCase`.

  >>> class MyTest(tf.test.TestCase):
  ...
  ...   def test_add_on_gpu(self):
  ...     if not tf.test.is_built_with_gpu_support():
  ...       self.skipTest("test is only applicable on GPU")
  ...
  ...     with tf.device(tf.test.gpu_device_name()):
  ...       self.assertEqual(tf.math.add(1.0, 2.0), 3.0)

  """
  for x in device_lib.list_local_devices():
    if x.device_type == "GPU":
      return compat.as_str(x.name)
  return ""


def assert_ops_in_graph(expected_ops, graph):
  """Assert all expected operations are found.

  Args:
    expected_ops: `dict<string, string>` of op name to op type.
    graph: Graph to check.

  Returns:
    `dict<string, node>` of node name to node.

  Raises:
    ValueError: If the expected ops are not present in the graph.
  """
  actual_ops = {}
  gd = graph.as_graph_def()
  for node in gd.node:
    if node.name in expected_ops:
      if expected_ops[node.name] != node.op:
        raise ValueError("Expected op for node %s is different. %s vs %s" %
                         (node.name, expected_ops[node.name], node.op))
      actual_ops[node.name] = node
  if set(expected_ops.keys()) != set(actual_ops.keys()):
    raise ValueError("Not all expected ops are present. Expected %s, found %s" %
                     (expected_ops.keys(), actual_ops.keys()))
  return actual_ops


@tf_export("test.assert_equal_graph_def", v1=[])
def assert_equal_graph_def_v2(expected, actual):
  """Asserts that two `GraphDef`s are (mostly) the same.

  Compares two `GraphDef` protos for equality, ignoring versions and ordering of
  nodes, attrs, and control inputs.  Node names are used to match up nodes
  between the graphs, so the naming of nodes must be consistent. This function
  ignores randomized attribute values that may appear in V2 checkpoints.

  Args:
    expected: The `GraphDef` we expected.
    actual: The `GraphDef` we have.

  Raises:
    AssertionError: If the `GraphDef`s do not match.
    TypeError: If either argument is not a `GraphDef`.
  """
  assert_equal_graph_def(actual, expected, checkpoint_v2=True,
                         hash_table_shared_name=True)


@tf_export(v1=["test.assert_equal_graph_def"])
def assert_equal_graph_def_v1(actual, expected, checkpoint_v2=False,
                              hash_table_shared_name=False):
  """Asserts that two `GraphDef`s are (mostly) the same.

  Compares two `GraphDef` protos for equality, ignoring versions and ordering of
  nodes, attrs, and control inputs.  Node names are used to match up nodes
  between the graphs, so the naming of nodes must be consistent.

  Args:
    actual: The `GraphDef` we have.
    expected: The `GraphDef` we expected.
    checkpoint_v2: boolean determining whether to ignore randomized attribute
      values that appear in V2 checkpoints.
    hash_table_shared_name: boolean determining whether to ignore randomized
      shared_names that appear in HashTableV2 op defs.

  Raises:
    AssertionError: If the `GraphDef`s do not match.
    TypeError: If either argument is not a `GraphDef`.
  """
  assert_equal_graph_def(actual, expected, checkpoint_v2,
                         hash_table_shared_name)


def assert_equal_graph_def(actual, expected, checkpoint_v2=False,
                           hash_table_shared_name=False):
  if not isinstance(actual, graph_pb2.GraphDef):
    raise TypeError("Expected tf.GraphDef for actual, got %s" %
                    type(actual).__name__)
  if not isinstance(expected, graph_pb2.GraphDef):
    raise TypeError("Expected tf.GraphDef for expected, got %s" %
                    type(expected).__name__)

  if checkpoint_v2:
    _strip_checkpoint_v2_randomized(actual)
    _strip_checkpoint_v2_randomized(expected)

  if hash_table_shared_name:
    _strip_hash_table_shared_name(actual)
    _strip_hash_table_shared_name(expected)

  diff = pywrap_tf_session.EqualGraphDefWrapper(actual.SerializeToString(),
                                                expected.SerializeToString())
  if diff:
    raise AssertionError(compat.as_str(diff))


def assert_meta_graph_protos_equal(tester, a, b):
  """Compares MetaGraphDefs `a` and `b` in unit test class `tester`."""
  # Carefully check the collection_defs
  tester.assertEqual(set(a.collection_def), set(b.collection_def))
  collection_keys = a.collection_def.keys()
  for k in collection_keys:
    a_value = a.collection_def[k]
    b_value = b.collection_def[k]
    proto_type = ops.get_collection_proto_type(k)
    if proto_type:
      a_proto = proto_type()
      b_proto = proto_type()
      # Number of entries in the collections is the same
      tester.assertEqual(
          len(a_value.bytes_list.value), len(b_value.bytes_list.value))
      for (a_value_item, b_value_item) in zip(a_value.bytes_list.value,
                                              b_value.bytes_list.value):
        a_proto.ParseFromString(a_value_item)
        b_proto.ParseFromString(b_value_item)
        tester.assertProtoEquals(a_proto, b_proto)
    else:
      tester.assertEqual(a_value, b_value)
  # Compared the fields directly, remove their raw values from the
  # proto comparison below.
  a.ClearField("collection_def")
  b.ClearField("collection_def")

  # Check the graph_defs.
  assert_equal_graph_def(a.graph_def, b.graph_def, checkpoint_v2=True)
  # Check graph_def versions (ignored by assert_equal_graph_def).
  tester.assertProtoEquals(a.graph_def.versions, b.graph_def.versions)
  # Compared the fields directly, remove their raw values from the
  # proto comparison below.
  a.ClearField("graph_def")
  b.ClearField("graph_def")

  tester.assertProtoEquals(a, b)


# Matches attributes named via _SHARDED_SUFFIX in
# tensorflow/python/training/saver.py
_SHARDED_SAVE_OP_PATTERN = "_temp_[0-9a-z]{32}/part"


def _strip_checkpoint_v2_randomized(graph_def):
  for node in graph_def.node:
    delete_keys = []
    for attr_key in node.attr:
      attr_tensor_value = node.attr[attr_key].tensor
      if attr_tensor_value and len(attr_tensor_value.string_val) == 1:
        attr_tensor_string_value = attr_tensor_value.string_val[0]
        if (attr_tensor_string_value and
            re.match(compat.as_bytes(_SHARDED_SAVE_OP_PATTERN),
                     attr_tensor_string_value)):
          delete_keys.append(attr_key)
    for attr_key in delete_keys:
      del node.attr[attr_key]


_TABLE_SHARED_NAME_PATTERN = r"hash_table_[0-9a-z\-]+"


def _strip_hash_table_shared_name(graph_def):
  for node in graph_def.node:
    delete_keys = []
    if node.op == "HashTableV2" and "shared_name" in node.attr:
      if re.match(compat.as_bytes(_TABLE_SHARED_NAME_PATTERN),
                  node.attr["shared_name"].s):
        delete_keys.append("shared_name")
    for attr_key in delete_keys:
      del node.attr[attr_key]


def IsGoogleCudaEnabled():
  return _pywrap_util_port.IsGoogleCudaEnabled()


def IsBuiltWithROCm():
  return _pywrap_util_port.IsBuiltWithROCm()


def IsBuiltWithXLA():
  return _pywrap_util_port.IsBuiltWithXLA()


def IsBuiltWithNvcc():
  return _pywrap_util_port.IsBuiltWithNvcc()


def GpuSupportsHalfMatMulAndConv():
  return _pywrap_util_port.GpuSupportsHalfMatMulAndConv()


def IsMklEnabled():
  return _pywrap_util_port.IsMklEnabled()


def InstallStackTraceHandler():
  _pywrap_stacktrace_handler.InstallStacktraceHandler()


def NHWCToNCHW(input_tensor):
  """Converts the input from the NHWC format to NCHW.

  Args:
    input_tensor: a 3-, 4-, or 5-D tensor, or an array representing shape

  Returns:
    converted tensor or shape array
  """
  # tensor dim -> new axis order
  new_axes = {3: [0, 2, 1], 4: [0, 3, 1, 2], 5: [0, 4, 1, 2, 3]}
  if isinstance(input_tensor, tensor_lib.Tensor):
    ndims = input_tensor.shape.ndims
    return array_ops.transpose(input_tensor, new_axes[ndims])
  else:
    ndims = len(input_tensor)
    return [input_tensor[a] for a in new_axes[ndims]]


def NHWCToNCHW_VECT_C(input_shape_or_tensor):
  """Transforms the input from the NHWC layout to NCHW_VECT_C layout.

  Note: Does not include quantization or type conversion steps, which should
  be applied afterwards.

  Args:
    input_shape_or_tensor: a 4- or 5-D tensor, or an array representing shape

  Returns:
    tensor or shape array transformed into NCHW_VECT_C

  Raises:
    ValueError: if last dimension of `input_shape_or_tensor` is not evenly
        divisible by 4.
  """
  permutations = {5: [0, 3, 1, 2, 4], 6: [0, 4, 1, 2, 3, 5]}
  is_tensor = isinstance(input_shape_or_tensor, tensor_lib.Tensor)
  temp_shape = (
      input_shape_or_tensor.shape.as_list()
      if is_tensor else input_shape_or_tensor)
  if temp_shape[-1] % 4 != 0:
    raise ValueError(
        "Last dimension of input must be evenly divisible by 4 to convert to "
        "NCHW_VECT_C.")
  temp_shape[-1] //= 4
  temp_shape.append(4)
  permutation = permutations[len(temp_shape)]
  if is_tensor:
    t = array_ops.reshape(input_shape_or_tensor, temp_shape)
    return array_ops.transpose(t, permutation)
  else:
    return [temp_shape[a] for a in permutation]


def NCHW_VECT_CToNHWC(input_shape_or_tensor):
  """Transforms the input from the NCHW_VECT_C layout to NHWC layout.

  Note: Does not include de-quantization or type conversion steps, which should
  be applied beforehand.

  Args:
    input_shape_or_tensor: a 5- or 6-D tensor, or an array representing shape

  Returns:
    tensor or shape array transformed into NHWC

  Raises:
    ValueError: if last dimension of `input_shape_or_tensor` is not 4.
  """
  permutations = {5: [0, 2, 3, 1, 4], 6: [0, 2, 3, 4, 1, 5]}
  is_tensor = isinstance(input_shape_or_tensor, tensor_lib.Tensor)
  input_shape = (
      input_shape_or_tensor.shape.as_list()
      if is_tensor else input_shape_or_tensor)
  if input_shape[-1] != 4:
    raise ValueError("Last dimension of NCHW_VECT_C must be 4.")
  permutation = permutations[len(input_shape)]
  nhwc_shape = [input_shape[a] for a in permutation[:-1]]
  nhwc_shape[-1] *= input_shape[-1]
  if is_tensor:
    t = array_ops.transpose(input_shape_or_tensor, permutation)
    return array_ops.reshape(t, nhwc_shape)
  else:
    return nhwc_shape


def NCHWToNHWC(input_tensor):
  """Converts the input from the NCHW format to NHWC.

  Args:
    input_tensor: a 4- or 5-D tensor, or an array representing shape

  Returns:
    converted tensor or shape array
  """
  # tensor dim -> new axis order
  new_axes = {4: [0, 2, 3, 1], 5: [0, 2, 3, 4, 1]}
  if isinstance(input_tensor, tensor_lib.Tensor):
    ndims = input_tensor.shape.ndims
    return array_ops.transpose(input_tensor, new_axes[ndims])
  else:
    ndims = len(input_tensor)
    return [input_tensor[a] for a in new_axes[ndims]]


def skip_if(condition):
  """Skips the decorated function if condition is or evaluates to True.

  Args:
    condition: Either an expression that can be used in "if not condition"
      statement, or a callable whose result should be a boolean.

  Returns:
    The wrapped function
  """

  def real_skip_if(fn):

    def wrapper(*args, **kwargs):
      if callable(condition):
        skip = condition()
      else:
        skip = condition
      if not skip:
        return fn(*args, **kwargs)

    return wrapper

  return real_skip_if


@contextlib.contextmanager
def skip_if_error(test_obj, error_type, messages=None):
  """Context manager to skip cases not considered failures by the tests.

  Note that this does not work if used in setUpClass/tearDownClass.
  Usage in setUp/tearDown works fine just like regular test methods.

  Args:
    test_obj: A test object provided as `self` in the test methods; this object
      is usually an instance of `unittest.TestCase`'s subclass and should have
      `skipTest` method.
    error_type: The error type to skip. Note that if `messages` are given, both
      `error_type` and `messages` need to match for the test to be skipped.
    messages: Optional, a string or list of strings. If `None`, the test will be
      skipped if `error_type` matches what is raised; otherwise, the test is
      skipped if any of the `messages` is contained in the message of the error
      raised, and `error_type` matches the error raised.

  Yields:
    Nothing.
  """
  if messages:
    messages = nest.flatten(messages)
  try:
    yield
  except error_type as e:
    if not messages or any(message in str(e) for message in messages):
      test_obj.skipTest("Skipping error: {}: {}".format(type(e), str(e)))
    else:
      raise


def enable_c_shapes(fn):
  """No-op. TODO(b/74620627): Remove this."""
  return fn


def with_c_shapes(cls):
  """No-op. TODO(b/74620627): Remove this."""
  return cls


def enable_control_flow_v2(fn):
  """Decorator for enabling CondV2 and WhileV2 on a test.

  Note this enables using CondV2 and WhileV2 after running the test class's
  setup/teardown methods.

  In addition to this, callers must import the while_v2 module in order to set
  the _while_v2 module in control_flow_ops.

  Args:
    fn: the function to be wrapped

  Returns:
    The wrapped function
  """

  def wrapper(*args, **kwargs):
    enable_control_flow_v2_old = control_flow_util.ENABLE_CONTROL_FLOW_V2
    control_flow_util.ENABLE_CONTROL_FLOW_V2 = True
    try:
      return fn(*args, **kwargs)
    finally:
      control_flow_util.ENABLE_CONTROL_FLOW_V2 = enable_control_flow_v2_old

  return wrapper


def with_control_flow_v2(cls):
  """Adds methods that call original methods with WhileV2 and CondV2 enabled.

  Note this enables CondV2 and WhileV2 in new methods after running the test
  class's setup method.

  In addition to this, callers must import the while_v2 module in order to set
  the _while_v2 module in control_flow_ops.

  If a test function has _disable_control_flow_v2 attr set to True (using the
  @disable_control_flow_v2 decorator), the v2 function is not generated for it.

  Example:

  @test_util.with_control_flow_v2
  class ControlFlowTest(test.TestCase):

    def testEnabledForV2(self):
      ...

    @test_util.disable_control_flow_v2("b/xyzabc")
    def testDisabledForV2(self):
      ...

  Generated class:
  class ControlFlowTest(test.TestCase):

    def testEnabledForV2(self):
      ...

    def testEnabledForV2WithControlFlowV2(self):
      // Enable V2 flags.
      testEnabledForV2(self)
      // Restore V2 flags.

    def testDisabledForV2(self):
      ...

  Args:
    cls: class to decorate

  Returns:
    cls with new test methods added
  """
  if control_flow_util.ENABLE_CONTROL_FLOW_V2:
    return cls

  for name, value in cls.__dict__.copy().items():
    if (callable(value) and
        name.startswith(unittest.TestLoader.testMethodPrefix) and
        not getattr(value, "_disable_control_flow_v2", False)):
      setattr(cls, name + "WithControlFlowV2", enable_control_flow_v2(value))
  return cls


def disable_control_flow_v2(unused_msg):
  """Decorator for a function in a with_control_flow_v2 enabled test class.

  Blocks the function from being run with v2 control flow ops.

  Args:
    unused_msg: Reason for disabling.

  Returns:
    The wrapped function with _disable_control_flow_v2 attr set to True.
  """

  def wrapper(func):
    func._disable_control_flow_v2 = True
    return func

  return wrapper


def enable_output_all_intermediates(fn):
  """Force-enable outputing all intermediates from functional control flow ops.

  Args:
    fn: the function to be wrapped

  Returns:
    The wrapped function
  """

  def wrapper(*args, **kwargs):
    output_all_intermediates_old = \
        control_flow_util_v2._EXPERIMENTAL_OUTPUT_ALL_INTERMEDIATES_OVERRIDE
    control_flow_util_v2._EXPERIMENTAL_OUTPUT_ALL_INTERMEDIATES_OVERRIDE = True
    try:
      return fn(*args, **kwargs)
    finally:
      control_flow_util_v2._EXPERIMENTAL_OUTPUT_ALL_INTERMEDIATES_OVERRIDE = \
          output_all_intermediates_old

  return wrapper


def assert_no_new_pyobjects_executing_eagerly(
    warmup_iters: int = 2,
) -> Callable[[Callable[..., Any]], Callable[..., None]]:
  """Decorator for asserting that no new Python objects persist after a test.

  Returns a decorator that runs the test multiple times executing eagerly,
  first as a warmup and then to let objects accumulate. The warmup helps ignore
  caches which do not grow as the test is run repeatedly.

  Useful for checking that there are no missing Py_DECREFs in the C exercised by
  a bit of Python.

  Args:
    warmup_iters: The numer of warmup iterations, excluded from measuring.

  Returns:
    A decorator function which can be applied to the test function.
  """

  def wrap_f(f: Callable[..., Any]) -> Callable[..., None]:
    def decorator(self: "TensorFlowTestCase", *args, **kwargs) -> None:
      """Warms up, gets object counts, runs the test, checks for new objects."""
      with context.eager_mode():
        gc.disable()
        # Python 3.11 removed "errors" and "skipped" as members of
        # unittest.case._Outcome so get them from the test result object
        # instead.
        test_errors = None
        test_skipped = None
        if hasattr(self._outcome, "errors"):
          test_errors = self._outcome.errors
          test_skipped = self._outcome.skipped
        else:
          test_errors = self._outcome.result.errors
          test_skipped = self._outcome.result.skipped
        # Run the test 2 times as warmup, in an attempt to fill up caches, which
        # should not grow as the test is run repeatedly below.
        #
        # TODO(b/117156879): Running warmup twice is black magic; we have seen
        # tests that fail with 1 warmup run, and pass with 2, on various
        # versions of python2.7.x.
        for _ in range(warmup_iters):
          f(self, *args, **kwargs)
        # Since we aren't in the normal test lifecycle, we need to manually run
        # cleanups to clear out their object references.
        self.doCleanups()

        # Some objects are newly created by _get_object_count_by_type().  So
        # create and save as a dummy variable to include it as a baseline.
        obj_count_by_type = _get_object_count_by_type()
        gc.collect()

        # Make sure any registered functions are cleaned up in the C++ runtime.
        registered_function_names = context.context().list_function_names()

        # unittest.doCleanups adds to self._outcome with each unwound call.
        # These objects are retained across gc collections so we exclude them
        # from the object count calculation.
        obj_count_by_type = _get_object_count_by_type(
            exclude=gc.get_referents(test_errors, test_skipped))

        if ops.has_default_graph():
          collection_sizes_before = {
              collection: len(ops.get_collection(collection))
              for collection in ops.get_default_graph().collections
          }
        for _ in range(3):
          f(self, *args, **kwargs)
        # Since we aren't in the normal test lifecycle, we need to manually run
        # cleanups to clear out their object references.
        self.doCleanups()
        # Note that gc.get_objects misses anything that isn't subject to garbage
        # collection (C types). Collections are a common source of leaks, so we
        # test for collection sizes explicitly.
        if ops.has_default_graph():
          for collection_key in ops.get_default_graph().collections:
            collection = ops.get_collection(collection_key)
            size_before = collection_sizes_before.get(collection_key, 0)
            if len(collection) > size_before:
              raise AssertionError(
                  ("Collection %s increased in size from "
                   "%d to %d (current items %s).") %
                  (collection_key, size_before, len(collection), collection))
            # Make sure our collection checks don't show up as leaked memory by
            # removing references to temporary variables.
            del collection
            del collection_key
            del size_before
          del collection_sizes_before
        gc.collect()

        # There should be no new Python objects hanging around.
        obj_count_by_type = (
            _get_object_count_by_type(
                exclude=gc.get_referents(test_errors, test_skipped)) -
            obj_count_by_type)

        # There should be no newly registered functions hanging around.
        leftover_functions = (
            context.context().list_function_names() - registered_function_names)
        assert not leftover_functions, (
            "The following functions were newly created: %s" %
            leftover_functions)

        # In some cases (specifically on MacOS), new_count is somehow
        # smaller than previous_count.
        # Using plain assert because not all classes using this decorator
        # have assertLessEqual
        assert not obj_count_by_type, (
            "The following objects were newly created: %s" %
            str(obj_count_by_type))
        gc.enable()
    return tf_decorator.make_decorator(f, decorator)

  return wrap_f


def assert_no_new_tensors(f):
  """Decorator for asserting that no new Tensors persist after a test.

  Mainly useful for checking that code using the Python C API has correctly
  manipulated reference counts.

  Clears the caches that it knows about, runs the garbage collector, then checks
  that there are no Tensor or Tensor-like objects still around. This includes
  Tensors to which something still has a reference (e.g. from missing
  Py_DECREFs) and uncollectable cycles (i.e. Python reference cycles where one
  of the objects has __del__ defined).

  Args:
    f: The test case to run.

  Returns:
    The decorated test case.
  """

  def decorator(*args, **kwargs):
    """Finds existing Tensors, runs the test, checks for new Tensors."""

    def _is_tensorflow_object(obj):
      try:
        return isinstance(obj,
                          (tensor_lib.Tensor, variables.Variable,
                           tensor_shape.Dimension, tensor_shape.TensorShape))
      except (ReferenceError, AttributeError):
        # If the object no longer exists, we don't care about it.
        return False

    tensors_before = set(
        id(obj) for obj in gc.get_objects() if _is_tensorflow_object(obj))
    outside_executed_eagerly = context.executing_eagerly()
    # Run the test in a new graph so that collections get cleared when it's
    # done, but inherit the graph key so optimizers behave.
    outside_graph_key = ops.get_default_graph()._graph_key
    with ops.Graph().as_default():
      ops.get_default_graph()._graph_key = outside_graph_key
      if outside_executed_eagerly:
        with context.eager_mode():
          result = f(*args, **kwargs)
      else:
        result = f(*args, **kwargs)
    # Make an effort to clear caches, which would otherwise look like leaked
    # Tensors.
    context.context()._clear_caches()  # pylint: disable=protected-access
    gc.collect()
    tensors_after = [
        obj for obj in gc.get_objects()
        if _is_tensorflow_object(obj) and id(obj) not in tensors_before
    ]
    if tensors_after:
      raise AssertionError(("%d Tensors not deallocated after test: %s" % (
          len(tensors_after),
          str(tensors_after),
      )))
    return result

  return tf_decorator.make_decorator(f, decorator)


def _find_reference_cycle(objects, idx):

  def get_ignore_reason(obj, denylist):
    """Tests whether an object should be omitted from the dependency graph."""
    if len(denylist) > 100:
      return "<depth limit>"
    if tf_inspect.isframe(obj):
      if "test_util.py" in tf_inspect.getframeinfo(obj)[0]:
        return "<test code>"
    for b in denylist:
      if b is obj:
        return "<test code>"
    if obj is denylist:
      return "<test code>"
    return None

  # Note: this function is meant to help with diagnostics. Its output is purely
  # a human-readable representation, so you may freely modify it to suit your
  # needs.
  def describe(obj, denylist, leaves_only=False):
    """Returns a custom human-readable summary of obj.

    Args:
      obj: the value to describe.
      denylist: same as denylist in get_ignore_reason.
      leaves_only: boolean flag used when calling describe recursively. Useful
        for summarizing collections.
    """
    if get_ignore_reason(obj, denylist):
      return "{}{}".format(get_ignore_reason(obj, denylist), type(obj))
    if tf_inspect.isframe(obj):
      return "frame: {}".format(tf_inspect.getframeinfo(obj))
    elif tf_inspect.ismodule(obj):
      return "module: {}".format(obj.__name__)
    else:
      if leaves_only:
        return "{}, {}".format(type(obj), id(obj))
      elif isinstance(obj, list):
        return "list({}): {}".format(
            id(obj), [describe(e, denylist, leaves_only=True) for e in obj])
      elif isinstance(obj, tuple):
        return "tuple({}): {}".format(
            id(obj), [describe(e, denylist, leaves_only=True) for e in obj])
      elif isinstance(obj, dict):
        return "dict({}): {} keys".format(id(obj), len(obj.keys()))
      elif tf_inspect.isfunction(obj):
        return "function({}) {}; globals ID: {}".format(
            id(obj), obj.__name__, id(obj.__globals__))
      else:
        return "{}, {}".format(type(obj), id(obj))

  def build_ref_graph(obj, graph, reprs, denylist):
    """Builds a reference graph as <referrer> -> <list of referents>.

    Args:
      obj: The object to start from. The graph will be built by recursively
        adding its referrers.
      graph: Dict holding the graph to be built. To avoid creating extra
        references, the graph holds object IDs rather than actual objects.
      reprs: Auxiliary structure that maps object IDs to their human-readable
        description.
      denylist: List of objects to ignore.
    """
    referrers = gc.get_referrers(obj)
    denylist = denylist + (referrers,)

    obj_id = id(obj)
    for r in referrers:
      if get_ignore_reason(r, denylist) is None:
        r_id = id(r)
        if r_id not in graph:
          graph[r_id] = []
        if obj_id not in graph[r_id]:
          graph[r_id].append(obj_id)
          build_ref_graph(r, graph, reprs, denylist)
          reprs[r_id] = describe(r, denylist)

  def find_cycle(el, graph, reprs, path):
    """Finds and prints a single cycle in the dependency graph."""
    if el not in graph:
      return
    for r in graph[el]:
      if r in path:
        logging.error("Reference cycle sample:")
        for p in path + (r,):
          logging.error(reprs.get(p, "unknown object " + str(p)))
        return True
      else:
        if find_cycle(r, graph, reprs, path + (r,)):
          return True
    return False

  obj = objects[idx]
  graph = {}  # referrer ID -> object ID
  reprs = {}  # object ID -> description
  build_ref_graph(obj, graph, reprs, (objects, graph, reprs, get_ignore_reason,
                                      describe, build_ref_graph, find_cycle))
  for k in graph:
    if find_cycle(k, graph, reprs, ()):
      return True
  return False


def assert_no_garbage_created(f):
  """Test method decorator to assert that no garbage has been created.

  Note that this decorator sets DEBUG_SAVEALL, which in some Python interpreters
  cannot be un-set (i.e. will disable garbage collection for any other unit
  tests in the same file/shard).

  Args:
    f: The function to decorate.

  Returns:
    The decorated function.
  """

  # FIXME(power) -- Update documentation, we no longer care if garbage is
  # created, we only want to verify we don't have memory leaks.
  def decorator(self, **kwargs):
    """Sets DEBUG_SAVEALL, runs the test, and checks for new garbage."""
    gc.disable()
    previous_debug_flags = gc.get_debug()
    gc.set_debug(gc.DEBUG_UNCOLLECTABLE)
    gc.collect()
    previous_garbage = len(gc.garbage)
    result = f(self, **kwargs)
    gc.collect()
    new_garbage = len(gc.garbage)
    if new_garbage > previous_garbage:

      for i, obj in enumerate(gc.garbage[previous_garbage:]):
        # Known false positive for ast.fix_missing_locations.
        if getattr(obj, "__module__", "") == "ast":
          new_garbage -= 3

    if new_garbage > previous_garbage:
      logging.error(
          "The decorated test created work for Python's garbage collector, "
          "likely due to a reference cycle. New objects in cycle(s):")
      for i, obj in enumerate(gc.garbage[previous_garbage:]):
        try:
          logging.error("Object %d of %d", i,
                        len(gc.garbage) - previous_garbage)

          def _safe_object_str(obj):
            return "<%s %d>" % (obj.__class__.__name__, id(obj))

          logging.error("  Object type: %s", _safe_object_str(obj))
          logging.error(
              "  Referrer types: %s", ", ".join(
                  [_safe_object_str(ref) for ref in gc.get_referrers(obj)]))
          logging.error(
              "  Referent types: %s", ", ".join(
                  [_safe_object_str(ref) for ref in gc.get_referents(obj)]))
          logging.error("  Object attribute names: %s", dir(obj))
          logging.error("  Object __str__:")
          logging.error(obj)
          logging.error("  Object __repr__:")
          logging.error(repr(obj))
        except Exception:  # pylint: disable=broad-except
          logging.error("(Exception while printing object)")

    # When garbage is created, this call can help identify reference cycles,
    # which are typically the cause of such garbage.
    if new_garbage > previous_garbage:
      for i in range(previous_garbage, new_garbage):
        if _find_reference_cycle(gc.garbage, i):
          break

    # This will fail if any garbage has been created, typically because of a
    # reference cycle.
    self.assertEqual(previous_garbage, new_garbage)
    # TODO(allenl): Figure out why this debug flag reset doesn't work. It would
    # be nice to be able to decorate arbitrary tests in a large test suite and
    # not hold on to every object in other tests.
    gc.set_debug(previous_debug_flags)
    gc.enable()
    return result

  return decorator


def _combine_named_parameters(**kwargs):
  """Generate combinations based on its keyword arguments.

  Two sets of returned combinations can be concatenated using +.  Their product
  can be computed using `times()`.

  Args:
    **kwargs: keyword arguments of form `option=[possibilities, ...]` or
      `option=the_only_possibility`.

  Returns:
    a list of dictionaries for each combination. Keys in the dictionaries are
    the keyword argument names.  Each key has one value - one of the
    corresponding keyword argument values.
  """
  sort_by_key = lambda k: k[0]
  combinations = []
  for key, values in sorted(kwargs.items(), key=sort_by_key):
    if not isinstance(values, list):
      values = [values]
    combinations.append([(key, value) for value in values])

  return [OrderedDict(result) for result in itertools.product(*combinations)]


def generate_combinations_with_testcase_name(**kwargs):
  """Generate combinations based on its keyword arguments using combine().

  This function calls combine() and appends a testcase name to the list of
  dictionaries returned. The 'testcase_name' key is a required for named
  parameterized tests.

  Args:
    **kwargs: keyword arguments of form `option=[possibilities, ...]` or
      `option=the_only_possibility`.

  Returns:
    a list of dictionaries for each combination. Keys in the dictionaries are
    the keyword argument names.  Each key has one value - one of the
    corresponding keyword argument values.
  """
  combinations = _combine_named_parameters(**kwargs)
  named_combinations = []
  for combination in combinations:
    assert isinstance(combination, OrderedDict)
    name = "".join([
        "_{}_{}".format("".join(filter(str.isalnum, key)),
                        "".join(filter(str.isalnum, str(value))))
        for key, value in combination.items()
    ])
    named_combinations.append(
        OrderedDict(
            list(combination.items()) +
            [("testcase_name", "_test{}".format(name))]))

  return named_combinations


def run_all_in_graph_and_eager_modes(cls):
  """Execute all test methods in the given class with and without eager."""
  base_decorator = run_in_graph_and_eager_modes
  for name in dir(cls):
    if (not name.startswith(unittest.TestLoader.testMethodPrefix) or
        name.startswith("testSkipEager") or
        name.startswith("test_skip_eager") or
        name == "test_session" or
        name == "test_scope"):
      continue
    value = getattr(cls, name, None)
    if callable(value):
      setattr(cls, name, base_decorator(value))
  return cls


def run_class_in_v1_v2(cls):
  """Execute all test methods in a given class in v1 and v2 modes."""
  base_decorator = run_in_v1_v2
  for name in dir(cls):
    if (not name.startswith(unittest.TestLoader.testMethodPrefix) or
        name.startswith("testSkipEager") or
        name.startswith("test_skip_eager") or
        name == "test_session" or
        name == "test_scope"):
      continue

    attr = getattr(cls, name, None)
    if not callable(attr):
      continue

    setattr(cls, name, base_decorator(attr))
  return cls


def enable_nested_function_shape_inference(fn):
  """Decorator for enabling nested_function_shape_inference on a test.

  This function returns a decorator intended to be applied to test methods in
  a `tf.test.TestCase` class. Doing so will set nested_function_shape_inference,
  reset the context, execute the test, then reset the context to the state
  it was in prior to this test.

  Example:

  class MyTest(test.TestCase):

    @enable_nested_function_shape_inference
    def testFoo(self):
      ...

  Args:
    fn: the function to be wrapped.

  Returns:
    The wrapped function.
  """

  def wrapper(*args, **kwargs):
    # If `nested_function_shape_inference` is already enabled do nothing.
    if flags.config().enable_nested_function_shape_inference.value():
      return fn(*args, **kwargs)

    flags.config().enable_nested_function_shape_inference.reset(True)
    try:
      return fn(*args, **kwargs)
    finally:
      flags.config().enable_nested_function_shape_inference.reset(False)

  return wrapper


def enable_quantized_dtypes_training(fn):
  """Decorator for enabling quantized_dtypes_training on a test.

  This function returns a decorator intended to be applied to test methods in
  a `tf.test.TestCase` class. Doing so will set quantized_dtypes_training,
  reset the context, execute the test, then reset the context to the state
  it was in prior to this test.

  Example:

  class MyTest(test.TestCase):

    @enable_quantized_dtypes_training
    def testFoo(self):
      ...

  Args:
    fn: the function to be wrapped.

  Returns:
    The wrapped function.
  """

  def wrapper(*args, **kwargs):
    # If `enable_quantized_dtypes_training` is already enabled do nothing.
    if flags.config().enable_quantized_dtypes_training.value():
      return fn(*args, **kwargs)

    flags.config().enable_quantized_dtypes_training.reset(True)
    try:
      return fn(*args, **kwargs)
    finally:
      flags.config().enable_quantized_dtypes_training.reset(False)

  return wrapper


def enable_eager_op_as_function(fn):
  """Returns the same fn. This will be removed once all usages are removed.

  Args:
    fn: the function to be wrapped.

  Returns:
    The wrapped function.
  """

  def wrapper(*args, **kwargs):
    return fn(*args, **kwargs)

  return wrapper


@tf_export("test.with_eager_op_as_function")
def with_eager_op_as_function(cls=None, only_as_function=False):  # pylint: disable=unused-argument
  """Returns the same class. This will be removed once all usages are removed.

  Args:
    cls: class to decorate.
    only_as_function: unused argument.

  Returns:
    cls
  """

  def decorator(cls):
    return cls

  if cls is not None:
    return decorator(cls)

  return decorator


def enable_graph_building_optimization(fn):
  """Decorator for enabling graph_building_optimization on a test.

  This function returns a decorator intended to be applied to test methods in
  a `tf.test.TestCase` class. Doing so will enable graph_building_optimization,
  execute the test, then reset the feature flag to its default value.

  Example:

  class MyTest(test.TestCase):

    @enable_graph_building_optimization
    def testFoo(self):
      ...

  Args:
    fn: the function to be wrapped.

  Returns:
    The wrapped function.
  """

  def wrapper(*args, **kwargs):
    # If `graph_building_optimization` is already enabled do nothing.
    if flags.config().graph_building_optimization.value():
      return fn(*args, **kwargs)

    flags.config().graph_building_optimization.reset(True)
    try:
      return fn(*args, **kwargs)
    finally:
      flags.config().graph_building_optimization.reset(False)

  return wrapper


def add_graph_building_optimization_tests(cls: _TC) -> _TC:
  """Adds methods with graph_building_optimization enabled to the test suite.

  Example:

  @test_util.add_graph_building_optimization_tests
  class FooTest(test.TestCase):

    def testBar(self):
      ...

  Generated class:
  class FooTest(test.TestCase):

    def testBar(self):
      ...

    def testBarWithGraphBuildingOptimization(self):
      // Enable graph_building_optimization
      testBar(self)
      // Disable graph_building_optimization

  Args:
    cls: class to decorate.

  Returns:
    cls with new test methods added.
  """

  if flags.config().graph_building_optimization.value():
    return cls

  for name, value in cls.__dict__.copy().items():
    if (callable(value) and
        (name.startswith(unittest.TestLoader.testMethodPrefix) or
         name.startswith("benchmark"))):
      setattr(cls, name + "WithGraphBuildingOptimization",
              enable_graph_building_optimization(value))
  return cls


def disable_eager_op_as_function(unused_msg):
  """Decorator for a function in a with_eager_op_as_function enabled test class.

  Blocks the function from being run with eager_op_as_function enabled.

  Args:
    unused_msg: Reason for disabling.

  Returns:
    The wrapped function with _disable_eager_op_as_function attr set to True.
  """
  return _disable_test(execute_func=False)


def set_xla_env_flag(
    flag: str = "",
) -> Callable[[Callable[..., _R]], Callable[..., _R]]:
  """Decorator for setting XLA_FLAGS prior to running a test.

  This function returns a decorator intended to be applied to test methods in
  a `tf.test.TestCase` class. Doing so will allow users to set any xla flags
  exposed via the XLA_FLAGS environment variable, execute the test, then reset
  the XLA_FLAGS to the state it was in prior to this test.

  Example:

  class MyTest(test.TestCase):

    @set_xla_env_flag(flag='--xla_gpu_enable_fast_min_max=false')
    def testFoo(self):
      ...

  Args:
    flag: The xla flag to be set in the XLA_FLAGS env variable.

  Returns:
    A decorator which sets the configured flag in XLA_FLAGS for the decorated
    function.
  """

  def decorator(f: Callable[..., _R]) -> Callable[..., _R]:

    @functools.wraps(f)
    def decorated(*args, **kwargs) -> _R:
      original_xla_flags = os.environ.get("XLA_FLAGS")
      new_xla_flags = flag
      if original_xla_flags:
        new_xla_flags = new_xla_flags + " " + original_xla_flags
      os.environ["XLA_FLAGS"] = new_xla_flags
      try:
        return f(*args, **kwargs)
      finally:
        if original_xla_flags is None:
          del os.environ["XLA_FLAGS"]
        else:
          os.environ["XLA_FLAGS"] = original_xla_flags

    return decorated

  return decorator


def build_as_function_and_v1_graph(
    func: Callable[..., Any],
) -> Callable[..., None]:
  """Run a test case in v1 graph mode and inside tf.function in eager mode.

  WARNING: This decorator can only be used in test cases that statically checks
  generated graph. Attempting to evaluate graph or function results via.
  session.run() or self.evaluate() will fail.

  WARNING: This decorator can only be used for test cases that inherit from
  absl.testing.parameterized.TestCase.

  Args:
    func: Test case function to be decorated.

  Returns:
    Decorated test case function.
  """

  if tf_inspect.isclass(func):
    raise ValueError(
        "`run_in_graph_mode_and_function` only supports test methods.")

  @parameterized.named_parameters(("_v1_graph", "v1_graph"),
                                  ("_function", "function"))
  @functools.wraps(func)
  def decorated(
      self: "TensorFlowTestCase",
      run_mode: str,
      *args,
      **kwargs,
  ) -> None:
    if run_mode == "v1_graph":
      with ops.Graph().as_default():
        func(self, *args, **kwargs)
    elif run_mode == "function":

      @def_function.function
      def function_in_eager():
        func(self, *args, **kwargs)

      # Create a new graph for the eagerly executed version of this test for
      # better isolation.
      graph_for_eager_test = ops.Graph()
      with graph_for_eager_test.as_default(), context.eager_mode():
        function_in_eager()
      ops.dismantle_graph(graph_for_eager_test)
    else:
      raise ValueError("Unknown run mode %s" % run_mode)

  return decorated


def run_in_async_and_sync_mode(f):
  """Execute the test in async mode and sync mode."""

  @parameterized.named_parameters([("Async", True), ("", False)])
  @functools.wraps(f)
  def decorator(self, async_mode, *args, **kwargs):
    if async_mode:
      with context.execution_mode(context.ASYNC):
        f(self, *args, **kwargs)
    else:
      with context.execution_mode(context.SYNC):
        f(self, *args, **kwargs)
  return decorator


def run_in_graph_and_eager_modes(func=None,
                                 config=None,
                                 use_gpu=True,
                                 assert_no_eager_garbage=False):
  """Execute the decorated test with and without enabling eager execution.

  This function returns a decorator intended to be applied to test methods in
  a `tf.test.TestCase` class. Doing so will cause the contents of the test
  method to be executed twice - once normally, and once with eager execution
  enabled. This allows unittests to confirm the equivalence between eager
  and graph execution (see `tf.compat.v1.enable_eager_execution`).

  For example, consider the following unittest:

  ```python
  class MyTests(tf.test.TestCase):

    @run_in_graph_and_eager_modes
    def test_foo(self):
      x = tf.constant([1, 2])
      y = tf.constant([3, 4])
      z = tf.add(x, y)
      self.assertAllEqual([4, 6], self.evaluate(z))

  if __name__ == "__main__":
    tf.test.main()
  ```

  This test validates that `tf.add()` has the same behavior when computed with
  eager execution enabled as it does when constructing a TensorFlow graph and
  executing the `z` tensor in a session.

  `deprecated_graph_mode_only`, `run_v1_only`, `run_v2_only`, and
  `run_in_graph_and_eager_modes` are available decorators for different
  v1/v2/eager/graph combinations.


  Args:
    func: function to be annotated. If `func` is None, this method returns a
      decorator the can be applied to a function. If `func` is not None this
      returns the decorator applied to `func`.
    config: An optional config_pb2.ConfigProto to use to configure the session
      when executing graphs.
    use_gpu: If True, attempt to run as many operations as possible on GPU.
    assert_no_eager_garbage: If True, sets DEBUG_SAVEALL on the garbage
      collector and asserts that no extra garbage has been created when running
      the test with eager execution enabled. This will fail if there are
      reference cycles (e.g. a = []; a.append(a)). Off by default because some
      tests may create garbage for legitimate reasons (e.g. they define a class
      which inherits from `object`), and because DEBUG_SAVEALL is sticky in some
      Python interpreters (meaning that tests which rely on objects being
      collected elsewhere in the unit test file will not work). Additionally,
      checks that nothing still has a reference to Tensors that the test
      allocated.

  Returns:
    Returns a decorator that will run the decorated test method twice:
    once by constructing and executing a graph in a session and once with
    eager execution enabled.
  """

  def decorator(f):
    if tf_inspect.isclass(f):
      raise ValueError(
          "`run_in_graph_and_eager_modes` only supports test methods. "
          "Did you mean to use `run_all_in_graph_and_eager_modes`?")

    def decorated(self, *args, **kwargs):
      logging.info("Running %s in GRAPH mode.", f.__name__)
      try:
        with context.graph_mode(), self.subTest("graph_mode"):
          # XLATestCase uses `session`, which also doesn't take any args,
          # instead of `test_session`
          for class_ in self.__class__.mro():
            if class_.__name__ == "XLATestCase":
              session_func = self.session
              session_kwargs = {}
              break
          else:
            session_func = self.test_session
            session_kwargs = dict(use_gpu=use_gpu, config=config)
          with session_func(**session_kwargs):
            f(self, *args, **kwargs)
      except unittest.case.SkipTest:
        pass

      def run_eagerly(self, **kwargs):
        logging.info("Running %s in EAGER mode.", f.__name__)
        if not use_gpu:
          with ops.device("/device:CPU:0"):
            f(self, *args, **kwargs)
        else:
          f(self, *args, **kwargs)

      if assert_no_eager_garbage:
        ops.reset_default_graph()
        run_eagerly = assert_no_new_tensors(
            assert_no_garbage_created(run_eagerly))

      # This decorator runs the wrapped test twice.
      # Reset the test environment between runs.
      self.tearDown()
      self._tempdir = None
      # Create a new graph for the eagerly executed version of this test for
      # better isolation.
      graph_for_eager_test = ops.Graph()
      with (
          graph_for_eager_test.as_default(),
          context.eager_mode(),
          self.subTest("eager_mode"),
      ):
        self.setUp()
        run_eagerly(self, **kwargs)
      ops.dismantle_graph(graph_for_eager_test)

    return tf_decorator.make_decorator(f, decorated)

  if func is not None:
    return decorator(func)

  return decorator


def run_in_v1_v2(
    device_to_use: Optional[str] = None,
    assert_no_eager_garbage: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., None]]:
  """Execute the decorated test in v1 and v2 modes.

  The overall execution is similar to that of `run_in_graph_and_eager_mode`.

  Args:
    device_to_use: A string in the following format: "/device:CPU:0".
    assert_no_eager_garbage: If True, sets DEBUG_SAVEALL on the garbage
      collector and asserts that no extra garbage has been created when running
      the test with eager execution enabled. This will fail if there are
      reference cycles (e.g. a = []; a.append(a)). Off by default because some
      tests may create garbage for legitimate reasons (e.g. they define a class
      which inherits from `object`), and because DEBUG_SAVEALL is sticky in some
      Python interpreters (meaning that tests which rely on objects being
      collected elsewhere in the unit test file will not work). Additionally,
      checks that nothing still has a reference to Tensors that the test
      allocated.

  Returns:
    A decorator that runs a given test in v1 and v2 modes.
  """

  def decorator(f: Callable[..., Any]) -> Callable[..., None]:
    decorator_tag = "wrapped_with_v1_v2_decorator"
    if hasattr(f, decorator_tag):
      # Already decorated with this very same decorator
      return f

    def decorated(self: "TensorFlowTestCase", *args, **kwargs) -> None:
      logging.info("Running %s in V1 mode.", f.__name__)
      try:
        with self.subTest("V1_mode"):
          v2_compat.disable_v2_behavior()
          f(self, *args, **kwargs)
      except unittest.case.SkipTest:
        pass

      def run_v2(self, **kwargs):
        logging.info("Running %s in V2 mode.", f.__name__)
        if device_to_use:
          with ops.device(device_to_use):
            f(self, *args, **kwargs)
        else:
          f(self, *args, **kwargs)

      if assert_no_eager_garbage:
        ops.reset_default_graph()
        run_v2 = assert_no_new_tensors(
            assert_no_garbage_created(run_v2))

      # This decorator runs the wrapped test twice.
      # Reset the test environment between runs.
      self.tearDown()
      self._tempdir = None  # pylint:disable=protected-access

      ops.reset_default_graph()
      v2_compat.enable_v2_behavior()
      with self.subTest("V2_mode"):
        self.setUp()
        run_v2(self, **kwargs)

    tf_decorated = tf_decorator.make_decorator(f, decorated)
    tf_decorated.__dict__[decorator_tag] = True
    return tf_decorated

  return decorator


def py_func_if_in_function(f):

  def decorated(*args, **kwds):
    if not ops.inside_function():
      return f(*args, **kwds)

    tensor_args = []
    tensor_indices = []
    for i, arg in enumerate(args):
      if isinstance(arg, (tensor_lib.Tensor, variables.Variable)):
        tensor_args.append(arg)
        tensor_indices.append(i)

    def inner_f(*inner_tensor_args):
      my_args = list(args)
      for i, n in zip(tensor_indices, inner_tensor_args):
        my_args[i] = n
      return f(*my_args, **kwds)

    return script_ops.py_func(inner_f, tensor_args, [])

  return tf_decorator.make_decorator(f, decorated)


def also_run_as_tf_function(f):
  """Runs the decorated test twice--once as is, once inside a tf.function.

  This allows you to run a test both in eager execution and inside a
  tf.function, exercising the two execution modes supported in tf 2.0. The test
  assertions are automatically done inside tf.py_funcs, and tf.function ensures
  that they run in the proper order and with the proper side effects.

  Currently variable creation is not supported in tests annotated with this
  decorator since it's tricky to ensure the variable doesn't get repeatedly
  created when retracing the tf.function.

  Args:
    f: the test method to be decorated

  Returns:
    The decorated test method, which will run both in eager and inside a
    tf.function.
  """

  def decorated(*args, **kwds):

    def bound_f():
      f(*args, **kwds)

    with context.eager_mode():
      # Running in eager mode
      bound_f()
      # Running as TF function
      # TODO(b/121143941): Remove the autograph override.
      def_function.function(bound_f, autograph=False)()

  return decorated


@overload
def deprecated_graph_mode_only(func: Callable[..., _R]) -> Callable[..., _R]:
  ...


@overload
def deprecated_graph_mode_only(func: _TC) -> Optional[_TC]:
  ...


def deprecated_graph_mode_only(
    func: Union[_TC, Callable[..., _R]],
) -> Union[_TC, Callable[..., _R]]:
  """Execute the decorated test in graph mode.

  This is a decorator intended to be applied to tests that are not compatible
  with eager mode. When this decorator is applied, the test body will be run in
  an environment where API calls construct graphs instead of executing eagerly.

  `deprecated_graph_mode_only`, `run_v1_only`, `run_v2_only`, and
  `run_in_graph_and_eager_modes` are available decorators for different
  v1/v2/eager/graph combinations.

  Args:
    func: function or class to be annotated.
      If `func` is a function this returns the decorator applied to `func`.
      If `func` is a unit test class this returns that class with the decorator
      applied to all test functions within that class.

  Returns:
    Returns a function or class that will run the decorated test(s)
    in graph mode.
  """

  if tf_inspect.isclass(func):
    setup = func.__dict__.get("setUp")
    if setup is not None:
      setattr(func, "setUp", deprecated_graph_mode_only(setup))

    for name, value in func.__dict__.copy().items():
      if (callable(value) and
          name.startswith(unittest.TestLoader.testMethodPrefix)):
        setattr(func, name, deprecated_graph_mode_only(value))

    return func

  def decorated(*args, **kwargs):
    if context.executing_eagerly():
      with context.graph_mode():
        return func(*args, **kwargs)
    else:
      return func(*args, **kwargs)

  return tf_decorator.make_decorator(func, decorated)


run_deprecated_v1 = deprecated_graph_mode_only


def run_all_in_deprecated_graph_mode_only(cls):
  """Execute all tests in a class in graph mode."""
  base_decorator = deprecated_graph_mode_only
  for name in dir(cls):
    if (not name.startswith(unittest.TestLoader.testMethodPrefix) or
        name == "test_session"):
      continue
    value = getattr(cls, name, None)
    if callable(value):
      setattr(cls, name, base_decorator(value))
  return cls


def _run_vn_only(func=None, v2=True, reason=None):
  """Execute the decorated test only if running in the specified mode.

  This function is intended to be applied to tests that exercise functionality
   that belongs to either only v2, or v1.
   If the test is run in the mode opposite of the specified one, it will simply
   be skipped.

   It shouldn't be used directly, instead, use the `run_v1_only` or
   `run_v2_only` wrappers that call it.

  `deprecated_graph_mode_only`, `run_v1_only`, `run_v2_only`, and
  `run_in_graph_and_eager_modes` are available decorators for different
  v1/v2/eager/graph combinations.

  Args:
    func: function to be annotated. If `func` is None, this method returns a
      decorator the can be applied to a function. If `func` is not None this
      returns the decorator applied to `func`.
    v2: a boolean value indicating whether the test should be skipped in v2,
      or v1.
    reason: string giving a reason for limiting the test to a particular mode.

  Returns:
    A decorator that will skip the test method in the specified version.
  """
  if not reason:
    reason = f"Test is only compatible with {'v2 ' if v2 else 'v1'}"

  def decorator(f):
    if tf_inspect.isclass(f):
      # To skip an entire test suite class, we only decorate the setUp method
      # to skip all tests. There are cases when setUp is not defined (not
      # overridden in subclasses of TestCase, so not available in f.__dict__
      # below). For those cases, we walk the method resolution order list and
      # pick the first setUp method we find (usually this should be the one in
      # the parent class since that's the TestCase class).
      for cls in type.mro(f):
        setup = cls.__dict__.get("setUp")
        if setup is not None:
          setattr(f, "setUp", decorator(setup))
          break

      return f
    else:
      # If f is just a function, just create a decorator for it and return it
      def decorated(self, *args, **kwargs):
        tf2_enabled = tf2.enabled()
        # Skip if TF is in v2 mode, but the test is expected to only be run
        # in v1, and vice versa
        if (tf2_enabled and not v2) or (not tf2_enabled and v2):
          self.skipTest(reason)

        return f(self, *args, **kwargs)

      return tf_decorator.make_decorator(f, decorated)

  if func is not None:
    return decorator(func)

  return decorator


def run_v1_only(reason=None, func=None):
  """Only execute the test if Tensorflow is in v1 mode."""
  return _run_vn_only(func=func, v2=False, reason=reason)


def run_v2_only(func=None, reason=None):
  """Only execute the test if Tensorflow is in v2 mode."""
  return _run_vn_only(func=func, v2=True, reason=reason)


def run_gpu_only(func: Callable[..., _R]) -> Callable[..., _R]:
  """Execute the decorated test only if a GPU is available.

  This function is intended to be applied to tests that require the presence
  of a GPU. If a GPU is absent, it will simply be skipped.

  Args:
    func: function to be annotated.

  Returns:
    Returns a function that will conditionally skip the decorated test method.
  """

  if tf_inspect.isclass(func):
    raise ValueError("`run_gpu_only` only supports test methods.")

  def decorated(self: "TensorFlowTestCase", *args, **kwargs) -> _R:
    if not is_gpu_available():
      self.skipTest("Test requires GPU")

    return func(self, *args, **kwargs)

  return decorated


def run_cuda_only(func: Callable[..., _R]) -> Callable[..., _R]:
  """Execute the decorated test only if a GPU is available.

  This function is intended to be applied to tests that require the presence
  of a CUDA GPU. If a CUDA GPU is absent, it will simply be skipped.

  Args:
    func: function to be annotated.

  Returns:
    Returns a function that will conditionally skip the decorated test method.
  """

  if tf_inspect.isclass(func):
    raise ValueError("`run_cuda_only` only supports test methods.")

  def decorated(self: "TensorFlowTestCase", *args, **kwargs) -> _R:
    if not is_gpu_available(cuda_only=True):
      self.skipTest("Test requires CUDA GPU")

    return func(self, *args, **kwargs)

  return decorated


def run_gpu_or_tpu(func: Callable[..., _R]) -> Callable[..., _R]:
  """Execute the decorated test only if a physical GPU or TPU is available.

  This function is intended to be applied to tests that require the presence
  of a physical GPU or TPU. It complies with the following rules:
  - If a GPU is available, the test will run on the GPU.
  - If a GPU is absent and a TPU is available, the test will run on the TPU.
  - If both GPU and TPU are absent, the test will be skipped.

  Args:
    func: function to be annotated.

  Returns:
    Returns a function that will conditionally skip the decorated test method.
  """

  if tf_inspect.isclass(func):
    raise ValueError("`run_gpu_or_tpu` only supports test methods.")

  def decorated(self: "TensorFlowTestCase", *args, **kwargs) -> _R:
    if config.list_physical_devices("GPU"):
      return func(self, "GPU", *args, **kwargs)

    if config.list_physical_devices("TPU"):
      return func(self, "TPU", *args, **kwargs)

    self.skipTest("Test requires GPU or TPU")

  return decorated


def with_forward_compatibility_horizons(*horizons):
  """Executes the decorated test with the specified forward-compat horizons.

  Args:
    *horizons: A list of (year, month, day) tuples.  If the list includes
      `None`, then the test will also be run with no forward-compatibility
      horizon set.

  Returns:
    A decorator that will execute the test with the specified horizons.
  """
  if not horizons:
    raise ValueError("Expected at least one horizon.")
  for horizon in horizons:
    if not ((horizon is None) or
            (len(horizon) == 3 and all(isinstance(x, int) for x in horizon))):
      raise ValueError("Bad horizon value: %r" % horizon)

  def decorator(f):
    if tf_inspect.isclass(f):
      raise ValueError("`with_forward_compatibility_horizons` only "
                       "supports test methods.")
    def decorated(*args, **kwargs):
      for horizon in horizons:
        if horizon is None:
          f(*args, **kwargs)
        else:
          (year, month, day) = horizon
          with forward_compatibility_horizon(year, month, day):
            f(*args, **kwargs)
    return tf_decorator.make_decorator(f, decorated)

  return decorator


@deprecation.deprecated(None,
                        "Use `tf.config.list_physical_devices('GPU')` instead.")
@tf_export("test.is_gpu_available")
def is_gpu_available(cuda_only=False, min_cuda_compute_capability=None):
  """Returns whether TensorFlow can access a GPU.

  Warning: if a non-GPU version of the package is installed, the function would
  also return False. Use `tf.test.is_built_with_cuda` to validate if TensorFlow
  was build with CUDA support.

  For example,
  >>> gpu_available = tf.test.is_gpu_available()
  >>> is_cuda_gpu_available = tf.test.is_gpu_available(cuda_only=True)
  >>> is_cuda_gpu_min_3 = tf.test.is_gpu_available(True, (3,0))

  Args:
    cuda_only: limit the search to CUDA GPUs.
    min_cuda_compute_capability: a (major,minor) pair that indicates the minimum
      CUDA compute capability required, or None if no requirement.

  Note that the keyword arg name "cuda_only" is misleading (since routine will
  return true when a GPU device is available irrespective of whether TF was
  built with CUDA support or ROCm support. However no changes here because

  ++ Changing the name "cuda_only" to something more generic would break
     backward compatibility

  ++ Adding an equivalent "rocm_only" would require the implementation check
     the build type. This in turn would require doing the same for CUDA and thus
     potentially break backward compatibility

  ++ Adding a new "cuda_or_rocm_only" would not break backward compatibility,
     but would require most (if not all) callers to update the call to use
     "cuda_or_rocm_only" instead of "cuda_only"

  Returns:
    True if a GPU device of the requested kind is available.
  """

  # This was needed earlier when we had support for SYCL in TensorFlow.
  del cuda_only

  try:
    for local_device in device_lib.list_local_devices():
      if local_device.device_type == "GPU":
        gpu_info = gpu_util.compute_capability_from_device_desc(local_device)
        cc = gpu_info.compute_capability or (0, 0)
        if not min_cuda_compute_capability or cc >= min_cuda_compute_capability:
          return True
    return False
  except errors_impl.NotFoundError as e:
    if not all(x in str(e) for x in ["CUDA", "not find"]):
      raise e
    else:
      logging.error(str(e))
      return False


@contextlib.contextmanager
def device(use_gpu):
  """Uses gpu when requested and available."""
  if use_gpu and is_gpu_available():
    dev = "/device:GPU:0"
  else:
    dev = "/device:CPU:0"
  with ops.device(dev):
    yield


@contextlib.contextmanager
def use_gpu():
  """Uses gpu when requested and available."""
  with device(use_gpu=True):
    yield


@contextlib.contextmanager
def force_gpu():
  """Force the gpu to be used."""
  with ops.device("/device:GPU:0"):
    yield


@contextlib.contextmanager
def force_cpu():
  """Force the cpu to be used."""
  with ops.device("/device:CPU:0"):
    yield


@contextlib.contextmanager
def deterministic_ops():
  """Enables deterministic ops."""
  try:
    config.enable_op_determinism()
    yield
  finally:
    config.disable_op_determinism()


class CapturedWrites:
  """A utility class to load the captured writes made to a stream."""

  def __init__(self, capture_location):
    self.capture_location = capture_location

  def contents(self):
    """Get the captured writes as a single string."""
    with open(self.capture_location) as tmp_file:
      output_data = "".join(tmp_file.readlines())
    return output_data


class FakeEagerSession:
  """Fake session so tests that conditionally use placeholders can use eager.

  There are a number of tests that conditionally use placeholders for shape
  inference. The pattern is demonstrated here:

  ```python
  with self.cached_session() as sess:
    if static_shape:
      y = math_ops.matmul(x, ...)
      feed_dict = {}
    else:
      x_ph = array_ops.placeholder(...)
      y = math_ops.matmul(x_ph, ...)
      feed_dict = {x_ph: x}
    val = sess.run(y, feed_dict=feed_dict)
  ```

  Since the feed_dict is empty when not using placeholders we should be able to
  call self.evaluate(), however this requires rewriting the test case.
  This class should be considered a stop-gap solution to get tests running with
  eager with minimal changes to the actual test.
  """

  def __init__(self, test_case):
    self._test_case = test_case

  def run(self, fetches, *args, **kwargs):
    """Evaluate `fetches`.

    Fail if additional args are specified.

    Args:
      fetches: A Tensor or a nested list/tuple of Tensors.
      *args: Positional arguments
      **kwargs: Keyword arguments

    Raises:
      RuntimeError: If args or kwargs are specified.

    Returns:
      Tensors as numpy values.
    """
    feed_dict = kwargs.pop("feed_dict", {})
    if feed_dict:
      raise RuntimeError(
          "feed_dict is not supported when eager execution is enabled "
          "(in this case, sess.run(t) is shorthand for t.numpy()")

    if args or kwargs:
      raise RuntimeError(
          "Optional args are not supported when eager execution is enabled "
          "(in this case, sess.run(t) is shorthand for t.numpy()")

    return self._test_case.evaluate(fetches)


class ErrorLoggingSession(s.Session):
  """Wrapper around a Session that logs errors in run()."""

  def run(self, *args, **kwargs):
    try:
      return super().run(*args, **kwargs)
    except Exception as e:  # pylint: disable=broad-except
      # Note: disable the logging for OutOfRangeError, which makes the output
      # of tf.data tests hard to read, because OutOfRangeError is used as the
      # signal completion
      if not isinstance(e, errors.OutOfRangeError):
        logging.error(str(e))
      raise


def disable_cudnn_autotune(func: Callable[..., _R]) -> Callable[..., _R]:
  """Disable autotuning during the call to this function.

  Some tests want to base assertions on a graph being isomorphic with a copy.
  To ensure this, this decorator disables autotuning.

  Args:
    func: Function to run with CuDNN autotuning turned off.

  Returns:
    Decorated function.
  """

  def decorated(*args, **kwargs) -> _R:
    original_tf_cudnn_use_autotune = os.environ.get("TF_CUDNN_USE_AUTOTUNE")
    os.environ["TF_CUDNN_USE_AUTOTUNE"] = "false"
    original_xla_flags = os.environ.get("XLA_FLAGS")
    new_xla_flags = "--xla_gpu_autotune_level=0"
    if original_xla_flags:
      new_xla_flags = original_xla_flags + " " + new_xla_flags
    os.environ["XLA_FLAGS"] = new_xla_flags

    result = func(*args, **kwargs)

    if (original_tf_cudnn_use_autotune is None):
      del os.environ["TF_CUDNN_USE_AUTOTUNE"]
    else:
      os.environ["TF_CUDNN_USE_AUTOTUNE"] = original_tf_cudnn_use_autotune
    if (original_xla_flags is None):
      del os.environ["XLA_FLAGS"]
    else:
      os.environ["XLA_FLAGS"] = original_xla_flags

    return result

  return tf_decorator.make_decorator(func, decorated)


# The description is just for documentation purposes.
def enable_tf_xla_constant_folding(
    description: str,
) -> Callable[[Callable[..., _R]], Callable[..., _R]]:

  if not isinstance(description, str):
    raise ValueError("'description' should be string, got {}".format(
        type(description)))

  def enable_tf_xla_constant_folding_impl(
      func: Callable[..., _R],
  ) -> Callable[..., _R]:
    """Enable constant folding during the call to this function.

    Some tests fail without constant folding.

    Args:
      func: Function to run with constant folding turned on.

    Returns:
      Decorated function.
    """

    def decorated(*args, **kwargs) -> _R:
      original_var = pywrap_tf_session.TF_GetXlaConstantFoldingDisabled()
      pywrap_tf_session.TF_SetXlaConstantFoldingDisabled(False)
      result = func(*args, **kwargs)
      pywrap_tf_session.TF_SetXlaConstantFoldingDisabled(original_var)
      return result

    return tf_decorator.make_decorator(func, decorated)

  return enable_tf_xla_constant_folding_impl


# Updates test function by selectively disabling it.
def _disable_test(
    execute_func: bool,
) -> Callable[[Callable[..., _R]], Callable[..., _R]]:

  def disable_test_impl(func: Callable[..., _R]) -> Callable[..., _R]:

    def decorated(*args, **kwargs) -> _R:
      if execute_func:
        return func(*args, **kwargs)

    return tf_decorator.make_decorator(func, decorated)

  return disable_test_impl


# The description is just for documentation purposes.
def disable_xla(
    description: str,  # pylint: disable=unused-argument
) -> Callable[[Callable[..., _R]], Callable[..., _R]]:
  """Execute the test method only if xla is not enabled."""
  execute_func = not is_xla_enabled()
  return _disable_test(execute_func)


# The description is just for documentation purposes.
def disable_mlir_bridge(
    description: str,  # pylint: disable=unused-argument
) -> Callable[[Callable[..., _R]], Callable[..., _R]]:
  """Execute the test method only if MLIR bridge is not enabled."""
  execute_func = not is_mlir_bridge_enabled()
  return _disable_test(execute_func)


# The description is just for documentation purposes.
def disable_asan(
    description: str,  # pylint: disable=unused-argument
) -> Callable[[Callable[..., _R]], Callable[..., _R]]:
  """Execute the test method only if ASAN is not enabled."""
  execute_func = not is_asan_enabled()
  return _disable_test(execute_func)


# The description is just for documentation purposes.
def disable_msan(
    description: str,  # pylint: disable=unused-argument
) -> Callable[[Callable[..., _R]], Callable[..., _R]]:
  """Execute the test method only if MSAN is not enabled."""
  execute_func = not is_msan_enabled()
  return _disable_test(execute_func)


# The description is just for documentation purposes.
def disable_tsan(
    description: str,  # pylint: disable=unused-argument
) -> Callable[[Callable[..., _R]], Callable[..., _R]]:
  """Execute the test method only if TSAN is not enabled."""
  execute_func = not is_tsan_enabled()
  return _disable_test(execute_func)


# The description is just for documentation purposes.
def disable_ubsan(
    description: str,  # pylint: disable=unused-argument
) -> Callable[[Callable[..., _R]], Callable[..., _R]]:
  """Execute the test method only if UBSAN is not enabled."""
  execute_func = not is_ubsan_enabled()
  return _disable_test(execute_func)


# The description is just for documentation purposes.
def disable_tfrt(
    unused_description: str,  # pylint: disable=unused-argument
) -> Callable[
    [Union[_TC, Callable[..., _R]]],
    Union[_TC, Callable[..., _R], None]
]:

  def disable_tfrt_impl(
      cls_or_func: Union[_TC, Callable[..., _R]]
  ) -> Union[_TC, Callable[..., _R], None]:
    """Execute the test only if tfrt is not enabled."""

    if tf_inspect.isclass(cls_or_func):
      if tfrt_utils.enabled():
        return None
      else:
        return cast(_TC, cls_or_func)
    else:
      func = cast(Callable[..., _R], cls_or_func)
      def decorated(*args, **kwargs) -> _R:
        if tfrt_utils.enabled():
          return
        else:
          return func(*args, **kwargs)

      return tf_decorator.make_decorator(cls_or_func, decorated)

  return disable_tfrt_impl


def for_all_test_methods(decorator, *args, **kwargs):
  """Generate class-level decorator from given method-level decorator.

  It is expected for the given decorator to take some arguments and return
  a method that is then called on the test method to produce a decorated
  method.

  Args:
    decorator: The decorator to apply.
    *args: Positional arguments
    **kwargs: Keyword arguments
  Returns: Function that will decorate a given classes test methods with the
    decorator.
  """

  def all_test_methods_impl(cls):
    """Apply decorator to all test methods in class."""
    for name in dir(cls):
      value = getattr(cls, name)
      if callable(value) and name.startswith(
          "test") and (name != "test_session"):
        setattr(cls, name, decorator(*args, **kwargs)(value))
    return cls

  return all_test_methods_impl


# The description is just for documentation purposes.
def no_xla_auto_jit(
    description: str,  # pylint: disable=unused-argument
) -> Callable[[Callable[..., _R]], Callable[..., _R]]:
  """This test is not intended to be run with XLA auto jit enabled."""
  execute_func = not is_xla_enabled()
  return _disable_test(execute_func)


# The description is just for documentation purposes.
def xla_allow_fallback(
    description: str,  # pylint: disable=unused-argument
):

  def xla_allow_fallback_impl(func: Callable[..., _R]) -> Callable[..., _R]:
    """Allow fallback to TF even though testing xla."""

    def decorated(*args, **kwargs) -> _R:
      if is_xla_enabled():
        # Update the global XLABuildOpsPassFlags to enable lazy compilation,
        # which allows the compiler to fall back to TF classic. Remember the
        # old value so that we can reset it.
        old_value = pywrap_tf_session.TF_SetXlaEnableLazyCompilation(True)
        result = func(*args, **kwargs)
        pywrap_tf_session.TF_SetXlaEnableLazyCompilation(old_value)
        return result
      else:
        return func(*args, **kwargs)

    return tf_decorator.make_decorator(func, decorated)

  return xla_allow_fallback_impl


# The description is just for documentation purposes.
def run_without_tensor_float_32(description):  # pylint: disable=unused-argument
  """Execute test with TensorFloat-32 disabled.

  While almost every real-world deep learning model runs fine with
  TensorFloat-32, many tests use assertAllClose or similar methods.
  TensorFloat-32 matmuls typically will cause such methods to fail with the
  default tolerances.

  Args:
    description: A description used for documentation purposes, describing why
      the test requires TensorFloat-32 to be disabled.

  Returns:
    Decorator which runs a test with TensorFloat-32 disabled.
  """

  def decorator(f):

    @functools.wraps(f)
    def decorated(*args, **kwargs):
      allowed = config.tensor_float_32_execution_enabled()
      try:
        config.enable_tensor_float_32_execution(False)
        f(*args, **kwargs)
      finally:
        config.enable_tensor_float_32_execution(allowed)

    return tf_decorator.make_decorator(f, decorated)

  return decorator


# The description is just for documentation purposes.
def run_all_without_tensor_float_32(description):  # pylint: disable=unused-argument
  """Execute all tests in a class with TensorFloat-32 disabled."""
  return for_all_test_methods(run_without_tensor_float_32, description)


def matmul_without_tf32(a, b, *args, **kwargs):
  """Run matmul but cast float32 inputs to float64 if TensorFloat-32 is enabled.

  This effectively runs matmul without TensorFloat-32. It should only be used in
  tests when verifying some other op or functions works correctly, e.g. to test
  `tf.linalg.sqrtm` by matrix multiplying the output of the op by itself. In
  such cases, the matmul itself is not being tested so it's OK to run it with
  higher precision.

  If a matmul itself is being tested, or some other op which uses matmul, use
  `run_without_tensor_float_32` instead.

  This also casts complex64 inputs to complex128, since TensorFloat-32 can also
  be used with complex64

  Args:
    a: First input to tf.linalg.matmul
    b: Second input to tf.linalg.matmul
    args: Other positional arguments to tf.linalg.matmul
    **kwargs: Other keyword arguments to tf.linalg.matmul

  Returns:
    A tensor with the same type as `a`.
  """
  if config.tensor_float_32_execution_enabled() and a.dtype == "float32":
    a = math_ops.cast(a, "float64")
    b = math_ops.cast(b, "float64")
    ret = math_ops.matmul(a, b, *args, **kwargs)
    return math_ops.cast(ret, a.dtype)
  elif config.tensor_float_32_execution_enabled() and a.dtype == "complex64":
    a = math_ops.cast(a, "complex128")
    b = math_ops.cast(b, "complex128")
    ret = math_ops.matmul(a, b, *args, **kwargs)
    return math_ops.cast(ret, a.dtype)
  else:
    return math_ops.matmul(a, b, *args, **kwargs)


class EagerSessionWarner:

  def __getattr__(self, attr):
    raise AttributeError(
        "Trying to access properties or call methods on the result of "
        "self.session(), self.cached_session(), etc while eager execution "
        "is enabled. If you're porting this test case to TF 2.0, either "
        "adapt the test to work with eager execution or insert a call to "
        "tf.disable_eager_execution() in the main() function of this test "
        "file.")

# TODO(b/286583977): Set it to True and remove.
_ENABLE_AUTO_BOTH_MODES = False


@tf_export("test.TestCase")
class TensorFlowTestCase(googletest.TestCase):
  """Base class for tests that need to test TensorFlow."""

  def __init__(self, methodName="runTest"):  # pylint: disable=invalid-name
    super().__init__(methodName)
    # Make sure we get unfiltered stack traces during the test
    traceback_utils.disable_traceback_filtering()
    if is_xla_enabled():
      pywrap_tf_session.TF_SetXlaAutoJitMode("2")
      pywrap_tf_session.TF_SetXlaMinClusterSize(1)
      pywrap_tf_session.TF_SetXlaEnableLazyCompilation(False)
      pywrap_tf_session.TF_SetTfXlaCpuGlobalJit(True)
      # Constant folding secretly runs code on TF:Classic CPU, so we also
      # disable it here.
      pywrap_tf_session.TF_SetXlaConstantFoldingDisabled(True)

    # Check if the mlir bridge has been explicitly enabled or disabled. If
    # is_mlir_bridge_enabled() returns None, the user did not explictly enable
    # or disable the bridge so do not update enable_mlir_bridge.
    if is_mlir_bridge_enabled():
      context.context().enable_mlir_bridge = True
    elif is_mlir_bridge_enabled() is not None:
      context.context().enable_mlir_bridge = False

    self._threads = []
    self._tempdir = None
    self._cached_session = None
    self._test_start_time = None
    # This flag provides the ability to control whether the graph mode gets
    # initialized for TF1 or not. Initializing for TF1, which is what was
    # happening earlier, was preventing enablement of 'eager mode' in the test.
    self._set_default_seed = True

  def setUp(self):
    super().setUp()
    self._ClearCachedSession()
    random.seed(random_seed.DEFAULT_GRAPH_SEED)
    np.random.seed(random_seed.DEFAULT_GRAPH_SEED)
    # Note: The following line is necessary because some test methods may error
    # out from within nested graph contexts (e.g., via assertRaises and
    # assertRaisesRegexp), which may leave ops._default_graph_stack non-empty
    # under certain versions of Python. That would cause
    # ops.reset_default_graph() to throw an exception if the stack were not
    # cleared first.
    ops._default_graph_stack.reset()  # pylint: disable=protected-access
    ops.reset_default_graph()
    if self._set_default_seed:
      random_seed.set_random_seed(random_seed.DEFAULT_GRAPH_SEED)
    # Reset summary writer in case another test used set_as_default() with their
    # summary writer.
    summary_state = summary_ops_v2._summary_state  # pylint: disable=protected-access
    summary_state.writer = None

    # Avoiding calling setUp() for the poorly named test_session method.
    if self.id().endswith(".test_session"):
      self.skipTest("Not a test.")

    self._test_start_time = time.time()

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    if _ENABLE_AUTO_BOTH_MODES:
      run_all_in_graph_and_eager_modes(cls)

  def tearDown(self):
    # If a subclass overrides setUp and doesn't call the parent class's setUp,
    # then we may not have set the start time.
    if self._test_start_time is not None:
      logging.info("time(%s): %ss", self.id(),
                   round(time.time() - self._test_start_time, 2))

    for thread in self._threads:
      thread.check_termination()

    self._ClearCachedSession()
    super().tearDown()

  def _ClearCachedSession(self):
    if self._cached_session is not None:
      self._cached_session.close()
      self._cached_session = None

  def get_temp_dir(self):
    """Returns a unique temporary directory for the test to use.

    If you call this method multiple times during in a test, it will return the
    same folder. However, across different runs the directories will be
    different. This will ensure that across different runs tests will not be
    able to pollute each others environment.
    If you need multiple unique directories within a single test, you should
    use tempfile.mkdtemp as follows:
      tempfile.mkdtemp(dir=self.get_temp_dir()):

    Returns:
      string, the path to the unique temporary directory created for this test.
    """
    if not self._tempdir:
      self._tempdir = tempfile.mkdtemp(dir=googletest.GetTempDir())
    return self._tempdir

  @contextlib.contextmanager
  def captureWritesToStream(self, stream) -> Iterator[CapturedWrites]:
    """A context manager that captures the writes to a given stream.

    This context manager captures all writes to a given stream inside of a
    `CapturedWrites` object. When this context manager is created, it yields
    the `CapturedWrites` object. The captured contents can be accessed  by
    calling `.contents()` on the `CapturedWrites`.

    For this function to work, the stream must have a file descriptor that
    can be modified using `os.dup` and `os.dup2`, and the stream must support
    a `.flush()` method. The default python sys.stdout and sys.stderr are
    examples of this. Note that this does not work in Colab or Jupyter
    notebooks, because those use alternate stdout streams.

    Example:
    ```python
    class MyOperatorTest(test_util.TensorFlowTestCase):
      def testMyOperator(self):
        input = [1.0, 2.0, 3.0, 4.0, 5.0]
        with self.captureWritesToStream(sys.stdout) as captured:
          result = MyOperator(input).eval()
        self.assertStartsWith(captured.contents(), "This was printed.")
    ```

    Args:
      stream: The stream whose writes should be captured. This stream must have
        a file descriptor, support writing via using that file descriptor, and
        must have a `.flush()` method.

    Yields:
      A `CapturedWrites` object that contains all writes to the specified stream
      made during this context.
    """
    stream.flush()
    fd = stream.fileno()
    tmp_file, tmp_file_path = tempfile.mkstemp(dir=self.get_temp_dir())
    orig_fd = os.dup(fd)
    os.dup2(tmp_file, fd)
    try:
      yield CapturedWrites(tmp_file_path)
    finally:
      os.close(tmp_file)
      os.dup2(orig_fd, fd)

  def _AssertProtoEquals(self, a, b, msg=None, relative_tolerance=None):
    """Asserts that a and b are the same proto.

    Uses ProtoEq() first, as it returns correct results
    for floating point attributes, and then use assertProtoEqual()
    in case of failure as it provides good error messages.

    Args:
      a: a proto.
      b: another proto.
      msg: Optional message to report on failure.
      relative_tolerance: float. The allowable difference between the two values
        being compared is determined by multiplying the relative tolerance by
        the maximum of the two values. If this is not provided, then all floats
        are compared using string comparison.
    """
    if not compare.ProtoEq(a, b):
      compare.assertProtoEqual(
          self,
          a,
          b,
          normalize_numbers=True,
          msg=msg,
          relative_tolerance=relative_tolerance,
      )

  def assertProtoEquals(
      self,
      expected_message_maybe_ascii,
      message,
      msg=None,
      relative_tolerance=None,
  ):
    """Asserts that message is same as parsed expected_message_ascii.

    Creates another prototype of message, reads the ascii message into it and
    then compares them using self._AssertProtoEqual().

    Args:
      expected_message_maybe_ascii: proto message in original or ascii form.
      message: the message to validate.
      msg: Optional message to report on failure.
      relative_tolerance: float. The allowable difference between the two values
        being compared is determined by multiplying the relative tolerance by
        the maximum of the two values. If this is not provided, then all floats
        are compared using string comparison.
    """
    if isinstance(expected_message_maybe_ascii, type(message)):
      expected_message = expected_message_maybe_ascii
      self._AssertProtoEquals(
          expected_message,
          message,
          msg=msg,
          relative_tolerance=relative_tolerance,
      )
    elif isinstance(expected_message_maybe_ascii, (str, bytes)):
      expected_message = type(message)()
      text_format.Merge(
          expected_message_maybe_ascii,
          expected_message,
          descriptor_pool=descriptor_pool.Default())
      self._AssertProtoEquals(
          expected_message,
          message,
          msg=msg,
          relative_tolerance=relative_tolerance,
      )
    else:
      assert False, ("Can't compare protos of type %s and %s." %
                     (type(expected_message_maybe_ascii), type(message)))

  def assertProtoEqualsVersion(
      self,
      expected,
      actual,
      producer=versions.GRAPH_DEF_VERSION,
      min_consumer=versions.GRAPH_DEF_VERSION_MIN_CONSUMER,
      msg=None):
    expected = "versions { producer: %d min_consumer: %d };\n%s" % (
        producer, min_consumer, expected)
    self.assertProtoEquals(expected, actual, msg=msg)

  def assertStartsWith(self, actual, expected_start, msg=None):
    """Assert that actual.startswith(expected_start) is True.

    Args:
      actual: str
      expected_start: str
      msg: Optional message to report on failure.
    """
    if not actual.startswith(expected_start):
      fail_msg = "%r does not start with %r" % (actual, expected_start)
      fail_msg += " : %r" % (msg) if msg else ""
      self.fail(fail_msg)

  def _eval_tensor(self, tensor):
    if tensor is None:
      return None
    elif callable(tensor):
      return self._eval_helper(tensor())
    else:
      try:
        # for compatibility with TF1 test cases
        if sparse_tensor.is_sparse(tensor):
          return sparse_tensor.SparseTensorValue(tensor.indices.numpy(),
                                                 tensor.values.numpy(),
                                                 tensor.dense_shape.numpy())
        elif ragged_tensor.is_ragged(tensor):
          return ragged_tensor_value.RaggedTensorValue(
              self._eval_tensor(tensor.values),
              self._eval_tensor(tensor.row_splits))
        elif isinstance(tensor, indexed_slices.IndexedSlices):
          return indexed_slices.IndexedSlicesValue(
              values=tensor.values.numpy(),
              indices=tensor.indices.numpy(),
              dense_shape=None
              if tensor.dense_shape is None else tensor.dense_shape.numpy())
        else:
          if hasattr(tensor, "numpy") and callable(tensor.numpy):
            return tensor.numpy()
          else:
            # Try our best to convert CompositeTensor components to NumPy
            # arrays. Officially, we don't support NumPy arrays as
            # CompositeTensor components. So don't be surprised if this doesn't
            # work.
            return nest.map_structure(lambda t: t.numpy(), tensor,
                                      expand_composites=True)
      except AttributeError as e:
        raise ValueError(f"Unsupported type {type(tensor).__name__!r}.") from e

  def _eval_helper(self, tensors):
    if tensors is None:
      return None
    return nest.map_structure(self._eval_tensor, tensors)

  def evaluate(
      self, tensors
  ) -> Union[
      ragged_tensor_value.RaggedTensorValue,
      sparse_tensor.SparseTensorValue,
      None
  ]:
    """Evaluates tensors and returns numpy values.

    Args:
      tensors: A Tensor or a nested list/tuple of Tensors.

    Returns:
      tensors numpy values.
    """
    if context.executing_eagerly():
      return self._eval_helper(tensors)
    else:
      sess = ops.get_default_session()
      flattened_tensors = nest.flatten(tensors)
      if sess is None:
        with self.test_session() as sess:
          flattened_results = sess.run(flattened_tensors)
      else:
        flattened_results = sess.run(flattened_tensors)
      return nest.pack_sequence_as(tensors, flattened_results)

  # pylint: disable=g-doc-return-or-yield
  # pylint: disable=redefined-outer-name
  @contextlib.contextmanager
  def session(
      self, graph=None, config=None, use_gpu=True, force_gpu=False
  ) -> Iterator[s.Session]:
    """A context manager for a TensorFlow Session for use in executing tests.

    Note that this will set this session and the graph as global defaults.

    Use the `use_gpu` and `force_gpu` options to control where ops are run. If
    `force_gpu` is True, all ops are pinned to `/device:GPU:0`. Otherwise, if
    `use_gpu` is True, TensorFlow tries to run as many ops on the GPU as
    possible. If both `force_gpu and `use_gpu` are False, all ops are pinned to
    the CPU.

    Example:

    ``` python
    class MyOperatorTest(test_util.TensorFlowTestCase):
      def testMyOperator(self):
        with self.session():
          valid_input = [1.0, 2.0, 3.0, 4.0, 5.0]
          result = MyOperator(valid_input).eval()
          self.assertEqual(result, [1.0, 2.0, 3.0, 5.0, 8.0]
          invalid_input = [-1.0, 2.0, 7.0]
          with self.assertRaisesOpError("negative input not supported"):
            MyOperator(invalid_input).eval()
    ```

    Args:
      graph: Optional graph to use during the returned session.
      config: An optional config_pb2.ConfigProto to use to configure the
        session.
      use_gpu: If True, attempt to run as many ops as possible on GPU.
      force_gpu: If True, pin all ops to `/device:GPU:0`.

    Yields:
      A Session object that should be used as a context manager to surround
      the graph building and execution code in a test case.
    """
    if context.executing_eagerly():
      yield EagerSessionWarner()
    else:
      with self._create_session(graph, config, force_gpu) as sess:
        with self._constrain_devices_and_set_default(sess, use_gpu, force_gpu):
          yield sess

  @contextlib.contextmanager
  def cached_session(self,
                     graph=None,
                     config=None,
                     use_gpu=True,
                     force_gpu=False) -> Iterator[s.Session]:
    """Returns a TensorFlow Session for use in executing tests.

    This method behaves differently than self.session(): for performance reasons
    `cached_session` will by default reuse the same session within the same
    test. The session returned by this function will only be closed at the end
    of the test (in the TearDown function).

    Use the `use_gpu` and `force_gpu` options to control where ops are run. If
    `force_gpu` is True, all ops are pinned to `/device:GPU:0`. Otherwise, if
    `use_gpu` is True, TensorFlow tries to run as many ops on the GPU as
    possible. If both `force_gpu and `use_gpu` are False, all ops are pinned to
    the CPU.

    Example:
    ```python
    class MyOperatorTest(test_util.TensorFlowTestCase):
      def testMyOperator(self):
        with self.cached_session() as sess:
          valid_input = [1.0, 2.0, 3.0, 4.0, 5.0]
          result = MyOperator(valid_input).eval()
          self.assertEqual(result, [1.0, 2.0, 3.0, 5.0, 8.0]
          invalid_input = [-1.0, 2.0, 7.0]
          with self.assertRaisesOpError("negative input not supported"):
            MyOperator(invalid_input).eval()
    ```

    Args:
      graph: Optional graph to use during the returned session.
      config: An optional config_pb2.ConfigProto to use to configure the
        session.
      use_gpu: If True, attempt to run as many ops as possible on GPU.
      force_gpu: If True, pin all ops to `/device:GPU:0`.

    Yields:
      A Session object that should be used as a context manager to surround
      the graph building and execution code in a test case.
    """
    if context.executing_eagerly():
      yield FakeEagerSession(self)
    else:
      sess = self._get_cached_session(
          graph, config, force_gpu, crash_if_inconsistent_args=True)
      with self._constrain_devices_and_set_default(sess, use_gpu,
                                                   force_gpu) as cached:
        yield cached

  @contextlib.contextmanager
  @deprecation.deprecated(None, "Use `self.session()` or "
                          "`self.cached_session()` instead.")
  def test_session(self,
                   graph=None,
                   config=None,
                   use_gpu=True,
                   force_gpu=False):
    """Use cached_session instead."""
    if self.id().endswith(".test_session"):
      self.skipTest(
          "Tests that have the name \"test_session\" are automatically skipped "
          "by TensorFlow test fixture, as the name is reserved for creating "
          "sessions within tests. Please rename your test if you have a test "
          "with this name.")
    if context.executing_eagerly():
      yield None
    else:
      if graph is None:
        sess = self._get_cached_session(
            graph, config, force_gpu, crash_if_inconsistent_args=False)
        with self._constrain_devices_and_set_default(sess, use_gpu,
                                                     force_gpu) as cached:
          yield cached
      else:
        with self.session(graph, config, use_gpu, force_gpu) as sess:
          yield sess

  # pylint: enable=g-doc-return-or-yield

  class _CheckedThread(object):
    """A wrapper class for Thread that asserts successful completion.

    This class should be created using the TensorFlowTestCase.checkedThread()
    method.
    """

    def __init__(self, testcase, target, args=None, kwargs=None):
      """Constructs a new instance of _CheckedThread.

      Args:
        testcase: The TensorFlowTestCase for which this thread is being created.
        target: A callable object representing the code to be executed in the
          thread.
        args: A tuple of positional arguments that will be passed to target.
        kwargs: A dictionary of keyword arguments that will be passed to target.
      """
      self._testcase = testcase
      self._target = target
      self._args = () if args is None else args
      self._kwargs = {} if kwargs is None else kwargs
      self._thread = threading.Thread(target=self._protected_run)
      self._exception = None

      self._is_thread_joined = False

    def _protected_run(self):
      """Target for the wrapper thread. Sets self._exception on failure."""
      try:
        self._target(*self._args, **self._kwargs)
      except Exception as e:  # pylint: disable=broad-except
        self._exception = e

    def start(self):
      """Starts the thread's activity.

      This must be called at most once per _CheckedThread object. It arranges
      for the object's target to be invoked in a separate thread of control.
      """
      self._thread.start()

    def join(self):
      """Blocks until the thread terminates.

      Raises:
        self._testcase.failureException: If the thread terminates with due to
          an exception.
      """
      self._is_thread_joined = True
      self._thread.join()
      if self._exception is not None:
        self._testcase.fail("Error in checkedThread: %s" % str(self._exception))

    def is_alive(self):
      """Returns whether the thread is alive.

      This method returns True just before the run() method starts
      until just after the run() method terminates.

      Returns:
        True if the thread is alive, otherwise False.
      """
      return self._thread.is_alive()

    def check_termination(self):
      """Returns whether the checked thread was properly used and did terminate.

      Every checked thread should be "join"ed after starting, and before the
      test tears down. If it is not joined, it is possible the thread will hang
      and cause flaky failures in tests.

      Raises:
        self._testcase.failureException: If check_termination was called before
        thread was joined.

        RuntimeError: If the thread is not terminated. This means thread was not
        joined with the main thread.
      """
      if self._is_thread_joined:
        if self.is_alive():
          raise RuntimeError(
              "Thread was not joined with main thread, and is still running "
              "when the test finished.")
      else:
        self._testcase.fail("A checked thread was not joined.")

  def checkedThread(self, target, args=None, kwargs=None):
    """Returns a Thread wrapper that asserts 'target' completes successfully.

    This method should be used to create all threads in test cases, as
    otherwise there is a risk that a thread will silently fail, and/or
    assertions made in the thread will not be respected.

    Args:
      target: A callable object to be executed in the thread.
      args: The argument tuple for the target invocation. Defaults to ().
      kwargs: A dictionary of keyword arguments for the target invocation.
        Defaults to {}.

    Returns:
      A wrapper for threading.Thread that supports start() and join() methods.
    """
    ret = TensorFlowTestCase._CheckedThread(self, target, args, kwargs)
    self._threads.append(ret)
    return ret

  # pylint: enable=invalid-name
  @py_func_if_in_function
  def assertNear(self, f1, f2, err, msg=None):
    """Asserts that two floats are near each other.

    Checks that |f1 - f2| < err and asserts a test failure
    if not.

    Args:
      f1: A float value.
      f2: A float value.
      err: A float value.
      msg: An optional string message to append to the failure message.
    """
    # f1 == f2 is needed here as we might have: f1, f2 = inf, inf
    self.assertTrue(
        f1 == f2 or math.fabs(f1 - f2) <= err, "%f != %f +/- %f%s" %
        (f1, f2, err, " (%s)" % msg if msg is not None else ""))

  @py_func_if_in_function
  def assertArrayNear(self, farray1, farray2, err, msg=None):
    """Asserts that two float arrays are near each other.

    Checks that for all elements of farray1 and farray2
    |f1 - f2| < err.  Asserts a test failure if not.

    Args:
      farray1: a list of float values.
      farray2: a list of float values.
      err: a float value.
      msg: Optional message to report on failure.
    """
    self.assertEqual(len(farray1), len(farray2), msg=msg)
    for f1, f2 in zip(farray1, farray2):
      self.assertNear(float(f1), float(f2), err, msg=msg)

  def _NDArrayNear(self, ndarray1, ndarray2, err):
    return np.linalg.norm(ndarray1 - ndarray2) < err

  @py_func_if_in_function
  def assertNDArrayNear(self, ndarray1, ndarray2, err, msg=None):
    """Asserts that two numpy arrays have near values.

    Args:
      ndarray1: a numpy ndarray.
      ndarray2: a numpy ndarray.
      err: a float. The maximum absolute difference allowed.
      msg: Optional message to report on failure.
    """
    self.assertTrue(self._NDArrayNear(ndarray1, ndarray2, err), msg=msg)

  def _GetNdArray(self, a):
    # If a is tensor-like then convert it to ndarray
    if tensor_util.is_tf_type(a):
      if isinstance(a, ops._EagerTensorBase):
        a = a.numpy()
      else:
        a = self.evaluate(a)
    if not isinstance(a, np.ndarray):
      try:
        return np.array(a)
      except ValueError as e:
        # TODO(b/264461299): NumPy 1.24 no longer infers dtype=object from
        # ragged sequences.
        # See:
        # https://numpy.org/neps/nep-0034-infer-dtype-is-object.html
        # Fixing this correctly requires clarifying the API contract of this
        # function with respect to ragged sequences and possibly updating all
        # users. As a backwards compatibility measure, if array
        # creation fails with an "inhomogeneous shape" error, try again with
        # an explicit dtype=object, which should restore the previous behavior.
        if "inhomogeneous shape" in str(e):
          return np.array(a, dtype=object)
        else:
          raise
    return a

  def evaluate_if_both_tensors(self, a, b):
    if (tensor_util.is_tf_type(a) and tensor_util.is_tf_type(b) and
        not isinstance(a, ops._EagerTensorBase) and
        not isinstance(b, ops._EagerTensorBase)):
      return self.evaluate((a, b))
    else:
      return (a, b)

  def _assertArrayLikeAllClose(self, a, b, rtol=1e-6, atol=1e-6, msg=None):
    (a, b) = self.evaluate_if_both_tensors(a, b)
    a = self._GetNdArray(a)
    b = self._GetNdArray(b)
    # When the array rank is small, print its contents. Numpy array printing is
    # implemented using inefficient recursion so prints can cause tests to
    # time out.
    if a.shape != b.shape and (b.ndim <= 3 or b.size < 500):
      shape_mismatch_msg = ("Shape mismatch: expected %s, got %s with contents "
                            "%s.") % (a.shape, b.shape, b)
    else:
      shape_mismatch_msg = "Shape mismatch: expected %s, got %s." % (a.shape,
                                                                     b.shape)
    self.assertEqual(a.shape, b.shape, shape_mismatch_msg)

    msgs = [msg]
    # np.allclose does not always work for our custom bfloat16 and float8
    # extension types when type promotions are involved, so we first cast any
    # arrays of such types to float32.
    a_dtype = a.dtype
    custom_dtypes = (dtypes.bfloat16.as_numpy_dtype,
                     dtypes.float8_e5m2.as_numpy_dtype,
                     dtypes.float8_e4m3fn.as_numpy_dtype)
    a = a.astype(np.float32) if a.dtype in custom_dtypes else a
    b = b.astype(np.float32) if b.dtype in custom_dtypes else b
    if not np.allclose(a, b, rtol=rtol, atol=atol):
      # Adds more details to np.testing.assert_allclose.
      #
      # NOTE: numpy.allclose (and numpy.testing.assert_allclose)
      # checks whether two arrays are element-wise equal within a
      # tolerance. The relative difference (rtol * abs(b)) and the
      # absolute difference atol are added together to compare against
      # the absolute difference between a and b.  Here, we want to
      # tell user which elements violate such conditions.
      cond = np.logical_or(
          np.abs(a - b) > atol + rtol * np.abs(b),
          np.isnan(a) != np.isnan(b))
      if a.ndim:
        x = a[np.where(cond)]
        y = b[np.where(cond)]
        msgs.append("not close where = {}".format(np.where(cond)))
      else:
        # np.where is broken for scalars
        x, y = a, b
      msgs.append("not close lhs = {}".format(x))
      msgs.append("not close rhs = {}".format(y))
      msgs.append("not close dif = {}".format(np.abs(x - y)))
      msgs.append("not close tol = {}".format(atol + rtol * np.abs(y)))
      msgs.append("dtype = {}, shape = {}".format(a_dtype, a.shape))
      # TODO(xpan): There seems to be a bug:
      # tensorflow/compiler/tests:binary_ops_test pass with float32
      # nan even though the equal_nan is False by default internally.
      np.testing.assert_allclose(
          a, b, rtol=rtol, atol=atol, err_msg="\n".join(msgs), equal_nan=True)

  def _assertAllCloseRecursive(self,
                               a,
                               b,
                               rtol=1e-6,
                               atol=1e-6,
                               path=None,
                               msg=None):
    if ragged_tensor.is_ragged(a) or ragged_tensor.is_ragged(b):
      return self._assertRaggedClose(a, b, rtol, atol, msg)
    path = path or []
    path_str = (("[" + "][".join(str(p) for p in path) + "]") if path else "")
    msg = msg if msg else ""

    # Check if a and/or b are namedtuples.
    if hasattr(a, "_asdict"):
      a = a._asdict()
    if hasattr(b, "_asdict"):
      b = b._asdict()
    a_is_dict = isinstance(a, collections_abc.Mapping)
    if a_is_dict != isinstance(b, collections_abc.Mapping):
      raise ValueError("Can't compare dict to non-dict, a%s vs b%s. %s" %
                       (path_str, path_str, msg))
    if a_is_dict:
      self.assertItemsEqual(
          a.keys(),
          b.keys(),
          msg="mismatched keys: a%s has keys %s, but b%s has keys %s. %s" %
          (path_str, a.keys(), path_str, b.keys(), msg))
      for k in a:
        path.append(k)
        self._assertAllCloseRecursive(
            a[k], b[k], rtol=rtol, atol=atol, path=path, msg=msg)
        del path[-1]
    elif isinstance(a, (list, tuple)):
      # Try to directly compare a, b as ndarrays; if not work, then traverse
      # through the sequence, which is more expensive.
      try:
        (a, b) = self.evaluate_if_both_tensors(a, b)
        a_as_ndarray = self._GetNdArray(a)
        b_as_ndarray = self._GetNdArray(b)
        self._assertArrayLikeAllClose(
            a_as_ndarray,
            b_as_ndarray,
            rtol=rtol,
            atol=atol,
            msg="Mismatched value: a%s is different from b%s. %s" %
            (path_str, path_str, msg))
      except (ValueError, TypeError, NotImplementedError) as e:
        if len(a) != len(b):
          raise ValueError(
              "Mismatched length: a%s has %d items, but b%s has %d items. %s" %
              (path_str, len(a), path_str, len(b), msg))
        for idx, (a_ele, b_ele) in enumerate(zip(a, b)):
          path.append(str(idx))
          self._assertAllCloseRecursive(
              a_ele, b_ele, rtol=rtol, atol=atol, path=path, msg=msg)
          del path[-1]
    # a and b are ndarray like objects
    else:
      try:
        self._assertArrayLikeAllClose(
            a,
            b,
            rtol=rtol,
            atol=atol,
            msg=("Mismatched value: a%s is different from b%s. %s" %
                 (path_str, path_str, msg)))
      except TypeError as e:
        msg = ("Error: a%s has %s, but b%s has %s. %s" %
               (path_str, type(a), path_str, type(b), msg))
        e.args = ((e.args[0] + " : " + msg,) + e.args[1:])
        raise

  @py_func_if_in_function
  def assertAllClose(self, a, b, rtol=1e-6, atol=1e-6, msg=None):
    """Asserts that two structures of numpy arrays or Tensors, have near values.

    `a` and `b` can be arbitrarily nested structures. A layer of a nested
    structure can be a `dict`, `namedtuple`, `tuple` or `list`.

    Note: the implementation follows
    [`numpy.allclose`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html)
    (and numpy.testing.assert_allclose). It checks whether two arrays are
    element-wise equal within a tolerance. The relative difference
    (`rtol * abs(b)`) and the absolute difference `atol` are added together
    to compare against the absolute difference between `a` and `b`.

    Args:
      a: The expected numpy `ndarray`, or anything that can be converted into a
        numpy `ndarray` (including Tensor), or any arbitrarily nested of
        structure of these.
      b: The actual numpy `ndarray`, or anything that can be converted into a
        numpy `ndarray` (including Tensor), or any arbitrarily nested of
        structure of these.
      rtol: relative tolerance.
      atol: absolute tolerance.
      msg: Optional message to report on failure.

    Raises:
      ValueError: if only one of `a[p]` and `b[p]` is a dict or
          `a[p]` and `b[p]` have different length, where `[p]` denotes a path
          to the nested structure, e.g. given `a = [(1, 1), {'d': (6, 7)}]` and
          `[p] = [1]['d']`, then `a[p] = (6, 7)`.
    """
    self._assertAllCloseRecursive(a, b, rtol=rtol, atol=atol, msg=msg)

  @py_func_if_in_function
  def assertAllCloseAccordingToType(self,
                                    a,
                                    b,
                                    rtol=1e-6,
                                    atol=1e-6,
                                    float_rtol=1e-6,
                                    float_atol=1e-6,
                                    half_rtol=1e-3,
                                    half_atol=1e-3,
                                    bfloat16_rtol=1e-2,
                                    bfloat16_atol=1e-2,
                                    msg=None):
    """Like assertAllClose, but also suitable for comparing fp16 arrays.

    In particular, the tolerance is reduced to 1e-3 if at least
    one of the arguments is of type float16.

    Args:
      a: the expected numpy ndarray or anything can be converted to one.
      b: the actual numpy ndarray or anything can be converted to one.
      rtol: relative tolerance.
      atol: absolute tolerance.
      float_rtol: relative tolerance for float32.
      float_atol: absolute tolerance for float32.
      half_rtol: relative tolerance for float16.
      half_atol: absolute tolerance for float16.
      bfloat16_rtol: relative tolerance for bfloat16.
      bfloat16_atol: absolute tolerance for bfloat16.
      msg: Optional message to report on failure.
    """
    (a, b) = self.evaluate_if_both_tensors(a, b)
    a = self._GetNdArray(a)
    b = self._GetNdArray(b)
    # types with lower tol are put later to overwrite previous ones.
    if (a.dtype == np.float32 or b.dtype == np.float32 or
        a.dtype == np.complex64 or b.dtype == np.complex64):
      rtol = max(rtol, float_rtol)
      atol = max(atol, float_atol)
    if a.dtype == np.float16 or b.dtype == np.float16:
      rtol = max(rtol, half_rtol)
      atol = max(atol, half_atol)
    if (a.dtype == dtypes.bfloat16.as_numpy_dtype or
        b.dtype == dtypes.bfloat16.as_numpy_dtype):
      rtol = max(rtol, bfloat16_rtol)
      atol = max(atol, bfloat16_atol)

    self.assertAllClose(a, b, rtol=rtol, atol=atol, msg=msg)

  @py_func_if_in_function
  def assertNotAllClose(self, a, b, rtol=1e-6, atol=1e-6, msg=None):
    """Assert that two numpy arrays, or Tensors, do not have near values.

    Args:
      a: The expected numpy `ndarray`, or anything that can be converted into a
        numpy `ndarray` (including Tensor), or any arbitrarily nested of
        structure of these.
      b: The actual numpy `ndarray`, or anything that can be converted into a
        numpy `ndarray` (including Tensor), or any arbitrarily nested of
        structure of these.
      rtol: relative tolerance.
      atol: absolute tolerance.
      msg: Optional message to report on failure.

    Raises:
      AssertionError: If `a` and `b` are unexpectedly close at all elements.
    """
    try:
      self.assertAllClose(a, b, rtol=rtol, atol=atol, msg=msg)
    except AssertionError:
      return
    msg = msg or ""
    raise AssertionError("The two values are close at all elements. %s" % msg)

  @py_func_if_in_function
  def assertAllEqual(self, a, b, msg=None):
    """Asserts that two numpy arrays or Tensors have the same values.

    Args:
      a: the expected numpy ndarray or anything can be converted to one.
      b: the actual numpy ndarray or anything can be converted to one.
      msg: Optional message to report on failure.
    """
    if (ragged_tensor.is_ragged(a) or ragged_tensor.is_ragged(b)):
      return self._assertRaggedEqual(a, b, msg)
    msg = msg if msg else ""
    (a, b) = self.evaluate_if_both_tensors(a, b)
    a = self._GetNdArray(a)
    b = self._GetNdArray(b)
    # Arbitrary bounds so that we don't print giant tensors.
    if (b.ndim <= 3 or b.size < 500):
      self.assertEqual(
          a.shape, b.shape, "Shape mismatch: expected %s, got %s."
          " Contents: %r. \n%s." % (a.shape, b.shape, b, msg))
    else:
      self.assertEqual(
          a.shape, b.shape, "Shape mismatch: expected %s, got %s."
          " %s" % (a.shape, b.shape, msg))

    same = (a == b)

    if dtypes.as_dtype(a.dtype).is_floating:
      same = np.logical_or(same, np.logical_and(np.isnan(a), np.isnan(b)))
    msgs = [msg]
    if not np.all(same):
      # Adds more details to np.testing.assert_array_equal.
      diff = np.logical_not(same)
      if a.ndim:
        x = a[np.where(diff)]
        y = b[np.where(diff)]
        msgs.append("not equal where = {}".format(np.where(diff)))
      else:
        # np.where is broken for scalars
        x, y = a, b
      msgs.append("not equal lhs = %r" % x)
      msgs.append("not equal rhs = %r" % y)

      if (a.dtype.kind != b.dtype.kind and
          {a.dtype.kind, b.dtype.kind}.issubset({"U", "S", "O"})):
        a_list = []
        b_list = []
        # OK to flatten `a` and `b` because they are guaranteed to have the
        # same shape.
        for out_list, flat_arr in [(a_list, a.flat), (b_list, b.flat)]:
          for item in flat_arr:
            if isinstance(item, str):
              out_list.append(item.encode("utf-8"))
            else:
              out_list.append(item)
        a = np.array(a_list)
        b = np.array(b_list)

      np.testing.assert_array_equal(a, b, err_msg="\n".join(msgs))

  @py_func_if_in_function
  def assertNotAllEqual(self, a, b, msg=None):
    """Asserts that two numpy arrays or Tensors do not have the same values.

    Args:
      a: the expected numpy ndarray or anything can be converted to one.
      b: the actual numpy ndarray or anything can be converted to one.
      msg: Optional message to report on failure.
    """
    try:
      self.assertAllEqual(a, b)
    except AssertionError:
      return
    msg = msg or ""
    raise AssertionError("The two values are equal at all elements. %s" % msg)

  @py_func_if_in_function
  def assertAllGreater(self, a, comparison_target):
    """Assert element values are all greater than a target value.

    Args:
      a: The numpy `ndarray`, or anything that can be converted into a numpy
        `ndarray` (including Tensor).
      comparison_target: The target value of comparison.
    """
    (a, comparison_target) = self.evaluate_if_both_tensors(a, comparison_target)
    a = self._GetNdArray(a)
    self.assertGreater(np.min(a), comparison_target)

  @py_func_if_in_function
  def assertAllLess(self, a, comparison_target):
    """Assert element values are all less than a target value.

    Args:
      a: The numpy `ndarray`, or anything that can be converted into a numpy
        `ndarray` (including Tensor).
      comparison_target: The target value of comparison.
    """
    (a, comparison_target) = self.evaluate_if_both_tensors(a, comparison_target)
    a = self._GetNdArray(a)
    self.assertLess(np.max(a), comparison_target)

  @py_func_if_in_function
  def assertAllGreaterEqual(self, a, comparison_target):
    """Assert element values are all greater than or equal to a target value.

    Args:
      a: The numpy `ndarray`, or anything that can be converted into a numpy
        `ndarray` (including Tensor).
      comparison_target: The target value of comparison.
    """
    (a, comparison_target) = self.evaluate_if_both_tensors(a, comparison_target)
    a = self._GetNdArray(a)
    self.assertGreaterEqual(np.min(a), comparison_target)

  @py_func_if_in_function
  def assertAllLessEqual(self, a, comparison_target):
    """Assert element values are all less than or equal to a target value.

    Args:
      a: The numpy `ndarray`, or anything that can be converted into a numpy
        `ndarray` (including Tensor).
      comparison_target: The target value of comparison.
    """
    (a, comparison_target) = self.evaluate_if_both_tensors(a, comparison_target)
    a = self._GetNdArray(a)
    self.assertLessEqual(np.max(a), comparison_target)

  def _format_subscripts(self, subscripts, value, limit=10, indent=2):
    """Generate a summary of ndarray subscripts as a list of str.

    If limit == N, this method will print up to the first N subscripts on
    separate
    lines. A line of ellipses (...) will be appended at the end if the number of
    subscripts exceeds N.

    Args:
      subscripts: The tensor (np.ndarray) subscripts, of the same format as
        np.where()'s return value, i.e., a tuple of arrays with each array
        corresponding to a dimension. E.g., (array([1, 1]), array([0, 1])).
      value: (np.ndarray) value of the tensor.
      limit: (int) The maximum number of indices to print.
      indent: (int) Number of characters to indent at the beginning of each
        line.

    Returns:
      (list of str) the multi-line representation of the subscripts and values,
        potentially with omission at the end.
    """
    lines = []
    subscripts = np.transpose(subscripts)
    prefix = " " * indent
    if np.ndim(value) == 0:
      return [prefix + "[0] : " + str(value)]
    for subscript in itertools.islice(subscripts, limit):
      lines.append(prefix + str(subscript) + " : " +
                   str(value[tuple(subscript)]))
    if len(subscripts) > limit:
      lines.append(prefix + "...")
    return lines

  @py_func_if_in_function
  def assertAllInRange(self,
                       target,
                       lower_bound,
                       upper_bound,
                       open_lower_bound=False,
                       open_upper_bound=False):
    """Assert that elements in a Tensor are all in a given range.

    Args:
      target: The numpy `ndarray`, or anything that can be converted into a
        numpy `ndarray` (including Tensor).
      lower_bound: lower bound of the range
      upper_bound: upper bound of the range
      open_lower_bound: (`bool`) whether the lower bound is open (i.e., > rather
        than the default >=)
      open_upper_bound: (`bool`) whether the upper bound is open (i.e., < rather
        than the default <=)

    Raises:
      AssertionError:
        if the value tensor does not have an ordered numeric type (float* or
          int*), or
        if there are nan values, or
        if any of the elements do not fall in the specified range.
    """
    target = self._GetNdArray(target)
    if not (np.issubdtype(target.dtype, np.floating) or
            np.issubdtype(target.dtype, np.integer)):
      raise AssertionError(
          "The value of %s does not have an ordered numeric type, instead it "
          "has type: %s" % (target, target.dtype))

    nan_subscripts = np.where(np.isnan(target))
    if np.size(nan_subscripts):
      raise AssertionError(
          "%d of the %d element(s) are NaN. "
          "Subscripts(s) and value(s) of the NaN element(s):\n" %
          (len(nan_subscripts[0]), np.size(target)) +
          "\n".join(self._format_subscripts(nan_subscripts, target)))

    range_str = (("(" if open_lower_bound else "[") + str(lower_bound) + ", " +
                 str(upper_bound) + (")" if open_upper_bound else "]"))

    violations = (
        np.less_equal(target, lower_bound) if open_lower_bound else np.less(
            target, lower_bound))
    violations = np.logical_or(
        violations,
        np.greater_equal(target, upper_bound)
        if open_upper_bound else np.greater(target, upper_bound))
    violation_subscripts = np.where(violations)
    if np.size(violation_subscripts):
      raise AssertionError(
          "%d of the %d element(s) are outside the range %s. " %
          (len(violation_subscripts[0]), np.size(target), range_str) +
          "Subscript(s) and value(s) of the offending elements:\n" +
          "\n".join(self._format_subscripts(violation_subscripts, target)))

  @py_func_if_in_function
  def assertAllInSet(self, target, expected_set):
    """Assert that elements of a Tensor are all in a given closed set.

    Args:
      target: The numpy `ndarray`, or anything that can be converted into a
        numpy `ndarray` (including Tensor).
      expected_set: (`list`, `tuple` or `set`) The closed set that the elements
        of the value of `target` are expected to fall into.

    Raises:
      AssertionError:
        if any of the elements do not fall into `expected_set`.
    """
    target = self._GetNdArray(target)

    # Elements in target that are not in expected_set.
    diff = np.setdiff1d(target.flatten(), list(expected_set))
    if np.size(diff):
      raise AssertionError("%d unique element(s) are not in the set %s: %s" %
                           (np.size(diff), expected_set, diff))

  @py_func_if_in_function
  def assertDTypeEqual(self, target, expected_dtype):
    """Assert ndarray data type is equal to expected.

    Args:
      target: The numpy `ndarray`, or anything that can be converted into a
        numpy `ndarray` (including Tensor).
      expected_dtype: Expected data type.
    """
    target = self._GetNdArray(target)
    if not isinstance(target, list):
      arrays = [target]
    for arr in arrays:
      self.assertEqual(arr.dtype, expected_dtype)

  # pylint: disable=g-doc-return-or-yield
  @contextlib.contextmanager
  def assertRaisesWithPredicateMatch(self, exception_type,
                                     expected_err_re_or_predicate):
    """Returns a context manager to enclose code expected to raise an exception.

    If the exception is an OpError, the op stack is also included in the message
    predicate search.

    Args:
      exception_type: The expected type of exception that should be raised.
      expected_err_re_or_predicate: If this is callable, it should be a function
        of one argument that inspects the passed-in exception and returns True
        (success) or False (please fail the test). Otherwise, the error message
        is expected to match this regular expression partially.

    Returns:
      A context manager to surround code that is expected to raise an
      exception.
    """
    if callable(expected_err_re_or_predicate):
      predicate = expected_err_re_or_predicate
    else:

      def predicate(e):
        err_str = e.message if isinstance(e, errors.OpError) else str(e)
        op = e.op if isinstance(e, errors.OpError) else None
        while op is not None:
          err_str += "\nCaused by: " + op.name
          op = op._original_op  # pylint: disable=protected-access
        logging.info("Searching within error strings: '%s' within '%s'",
                     expected_err_re_or_predicate, err_str)
        return re.search(expected_err_re_or_predicate, err_str)

    try:
      yield
      self.fail(exception_type.__name__ + " not raised")
    except Exception as e:  # pylint: disable=broad-except
      if not isinstance(e, exception_type) or not predicate(e):
        raise AssertionError("Exception of type %s: %s" %
                             (str(type(e)), str(e)))

  # pylint: enable=g-doc-return-or-yield

  def assertRaisesOpError(self, expected_err_re_or_predicate):
    return self.assertRaisesWithPredicateMatch(errors.OpError,
                                               expected_err_re_or_predicate)

  def assertRaisesIncompatibleShapesError(
      self, exception_type=errors.InvalidArgumentError):
    return self.assertRaisesWithPredicateMatch(
        exception_type, r"Incompatible shapes|Dimensions must be equal|"
        r"required broadcastable shapes")

  def assertShapeEqual(self, input_a, input_b, msg=None):
    """Asserts that two Numpy or TensorFlow objects have the same shape.

    For Tensors, this compares statically known shapes at compile time, not
    dynamic shapes at runtime.

    Args:
      input_a: A Numpy ndarray, Numpy scalar, or a Tensor.
      input_b: A Numpy ndarray, Numpy scalar, or a Tensor.
      msg: Optional message to report on failure.

    Raises:
      TypeError: If the arguments have the wrong type.
    """
    if not isinstance(input_a, (np.ndarray, np.generic, tensor_lib.Tensor)):
      raise TypeError(
          "input_a must be a Numpy ndarray, Numpy scalar, or a Tensor."
          f"Instead received {type(input_a)}")
    if not isinstance(input_b, (np.ndarray, np.generic, tensor_lib.Tensor)):
      raise TypeError(
          "input_b must be a Numpy ndarray, Numpy scalar, or a Tensor."
          f"Instead received {type(input_b)}")
    shape_a = input_a.get_shape().as_list() if isinstance(
        input_a, tensor_lib.Tensor) else input_a.shape
    shape_b = input_b.get_shape().as_list() if isinstance(
        input_b, tensor_lib.Tensor) else input_b.shape
    self.assertAllEqual(shape_a, shape_b, msg=msg)

  def assertDeviceEqual(self, device1, device2, msg=None):
    """Asserts that the two given devices are the same.

    Args:
      device1: A string device name or TensorFlow `DeviceSpec` object.
      device2: A string device name or TensorFlow `DeviceSpec` object.
      msg: Optional message to report on failure.
    """
    device1 = pydev.canonical_name(device1)
    device2 = pydev.canonical_name(device2)
    self.assertEqual(
        device1, device2,
        "Devices %s and %s are not equal. %s" % (device1, device2, msg))

  @py_func_if_in_function
  def assertDictEqual(self, a, b, msg=None):
    """Assert that two given dictionary of tensors are the same.

    Args:
      a: Expected dictionary with numpy ndarray or anything else that can be
        converted to one as values.
      b: Actual dictionary with numpy ndarray or anything else that can be
        converted to one as values.
      msg: Optional message to report on failure.
    """
    # To keep backwards compatibility, we first try the base class
    # assertDictEqual. If that fails we try the tensorflow one.
    try:
      super().assertDictEqual(a, b, msg)
    except Exception:  # pylint: disable=broad-except
      self.assertSameElements(a.keys(), b.keys())  # pylint: disable=g-assert-in-except
      for k, v in a.items():
        (a_k, b_k) = self.evaluate_if_both_tensors(v, b[k])
        a_k = self._GetNdArray(a_k)
        b_k = self._GetNdArray(b_k)
        if np.issubdtype(a_k.dtype, np.floating):
          self.assertAllClose(v, b[k], msg=k)
        else:
          self.assertAllEqual(v, b[k], msg=k)

  def _GetPyList(self, a):
    """Converts `a` to a nested python list."""
    if isinstance(a, ragged_tensor.RaggedTensor):
      return self.evaluate(a).to_list()
    elif isinstance(a, tensor_lib.Tensor):
      a = self.evaluate(a)
      return a.tolist() if isinstance(a, np.ndarray) else a
    elif isinstance(a, np.ndarray):
      return a.tolist()
    elif isinstance(a, ragged_tensor_value.RaggedTensorValue):
      return a.to_list()
    else:
      return np.array(a, dtype=object).tolist()

  def _assertRaggedEqual(self, a, b, msg):
    """Asserts that two ragged tensors are equal."""
    a_list = self._GetPyList(a)
    b_list = self._GetPyList(b)
    self.assertEqual(a_list, b_list, msg)

    if not (isinstance(a, (list, tuple)) or isinstance(b, (list, tuple))):
      a_ragged_rank = a.ragged_rank if ragged_tensor.is_ragged(a) else 0
      b_ragged_rank = b.ragged_rank if ragged_tensor.is_ragged(b) else 0
      self.assertEqual(a_ragged_rank, b_ragged_rank, msg)

  def _assertRaggedClose(self, a, b, rtol, atol, msg=None):
    a_list = self._GetPyList(a)
    b_list = self._GetPyList(b)
    self._assertListCloseRecursive(a_list, b_list, rtol, atol, msg)

    if not (isinstance(a, (list, tuple)) or isinstance(b, (list, tuple))):
      a_ragged_rank = a.ragged_rank if ragged_tensor.is_ragged(a) else 0
      b_ragged_rank = b.ragged_rank if ragged_tensor.is_ragged(b) else 0
      self.assertEqual(a_ragged_rank, b_ragged_rank, msg)

  def _assertListCloseRecursive(self, a, b, rtol, atol, msg, path="value"):
    self.assertEqual(type(a), type(b))
    if isinstance(a, (list, tuple)):
      self.assertLen(a, len(b), "Length differs for %s" % path)
      for i in range(len(a)):
        self._assertListCloseRecursive(a[i], b[i], rtol, atol, msg,
                                       "%s[%s]" % (path, i))
    else:
      self._assertAllCloseRecursive(a, b, rtol, atol, path, msg)

  # Fix Python 3+ compatibility issues
  # pylint: disable=invalid-name

  # Silence a deprecation warning
  assertRaisesRegexp = googletest.TestCase.assertRaisesRegex

  # assertItemsEqual is assertCountEqual as of 3.2.
  assertItemsEqual = googletest.TestCase.assertCountEqual

  # pylint: enable=invalid-name

  @contextlib.contextmanager
  def _constrain_devices_and_set_default(self, sess, use_gpu, force_gpu):
    """Set the session and its graph to global default and constrain devices."""
    if context.executing_eagerly():
      yield None
    else:
      with sess.graph.as_default(), sess.as_default():
        if force_gpu:
          # Use the name of an actual device if one is detected, or
          # '/device:GPU:0' otherwise
          gpu_name = gpu_device_name()
          if not gpu_name:
            gpu_name = "/device:GPU:0"
          with sess.graph.device(gpu_name):
            yield sess
        elif use_gpu:
          yield sess
        else:
          with sess.graph.device("/device:CPU:0"):
            yield sess

  def _create_session(self, graph, config, force_gpu):
    """See session() for details."""

    def prepare_config(config):
      """Returns a config for sessions.

      Args:
        config: An optional config_pb2.ConfigProto to use to configure the
          session.

      Returns:
        A config_pb2.ConfigProto object.
      """
      # TODO(b/114333779): Enforce allow_soft_placement=False when
      # use_gpu=False. Currently many tests rely on the fact that any device
      # will be used even when a specific device is supposed to be used.
      allow_soft_placement = not force_gpu
      if config is None:
        config = context.context().config
        config.allow_soft_placement = allow_soft_placement
      elif not allow_soft_placement and config.allow_soft_placement:
        config_copy = context.context().config
        config = config_copy
        config.allow_soft_placement = False
      # Don't perform optimizations for tests so we don't inadvertently run
      # gpu ops on cpu
      config.graph_options.optimizer_options.opt_level = -1
      # Disable Grappler constant folding since some tests & benchmarks
      # use constant input and become meaningless after constant folding.
      # DO NOT DISABLE GRAPPLER OPTIMIZERS WITHOUT CONSULTING WITH THE
      # GRAPPLER TEAM.
      config.graph_options.rewrite_options.constant_folding = (
          rewriter_config_pb2.RewriterConfig.OFF)
      config.graph_options.rewrite_options.pin_to_host_optimization = (
          rewriter_config_pb2.RewriterConfig.OFF)
      return config

    return ErrorLoggingSession(graph=graph, config=prepare_config(config))

  def _get_cached_session(self,
                          graph=None,
                          config=None,
                          force_gpu=False,
                          crash_if_inconsistent_args=True):
    """See cached_session() for documentation."""
    if self._cached_session is None:
      sess = self._create_session(
          graph=graph, config=config, force_gpu=force_gpu)
      self._cached_session = sess
      self._cached_graph = graph
      self._cached_config = config
      self._cached_force_gpu = force_gpu
      return sess
    else:
      if crash_if_inconsistent_args and self._cached_graph is not graph:
        raise ValueError("The graph used to get the cached session is "
                         "different than the one that was used to create the "
                         "session. Maybe create a new session with "
                         "self.session()")
      if crash_if_inconsistent_args and self._cached_config is not config:
        raise ValueError("The config used to get the cached session is "
                         "different than the one that was used to create the "
                         "session. Maybe create a new session with "
                         "self.session()")
      if crash_if_inconsistent_args and (self._cached_force_gpu is
                                         not force_gpu):
        raise ValueError(
            "The force_gpu value used to get the cached session is "
            "different than the one that was used to create the "
            "session. Maybe create a new session with "
            "self.session()")
      return self._cached_session


ASSIGNED_PORTS = set()
lock = threading.Lock()


def pick_unused_port():
  """Returns an unused and unassigned local port."""
  import portpicker  # pylint: disable=g-import-not-at-top

  global ASSIGNED_PORTS
  with lock:
    while True:
      try:
        port = portpicker.pick_unused_port()
      except portpicker.NoFreePortFoundError as porterror:
        raise unittest.SkipTest("Flakes in portpicker library do not represent"
                                " TensorFlow errors.") from porterror
      if port > 10000 and port not in ASSIGNED_PORTS:
        ASSIGNED_PORTS.add(port)
        logging.info("Using local port %r", port)
        return port


@tf_export("test.create_local_cluster")
def create_local_cluster(num_workers,
                         num_ps,
                         protocol="grpc",
                         worker_config=None,
                         ps_config=None):
  """Create and start local servers and return the associated `Server` objects.

  "PS" stands for "parameter server": a task responsible for storing and
  updating the model's parameters. Other tasks send updates to these parameters
  as they work on optimizing the parameters. This particular division of labor
  between tasks is not required, but is common for distributed training.

  Read more at https://www.tensorflow.org/guide/extend/architecture

  ![components](https://www.tensorflow.org/images/diag1.svg "components")


  Figure illustrates the interaction of these components.
  "/job:worker/task:0" and "/job:ps/task:0" are both tasks with worker services.


  Example:
  ```python
  workers, _ = tf.test.create_local_cluster(num_workers=2, num_ps=2)

  worker_sessions = [tf.compat.v1.Session(w.target) for w in workers]

  with tf.device("/job:ps/task:0"):
    ...
  with tf.device("/job:ps/task:1"):
    ...
  with tf.device("/job:worker/task:0"):
    ...
  with tf.device("/job:worker/task:1"):
    ...

  worker_sessions[0].run(...)
  ```

  Args:
    num_workers: Number of worker servers to start.
    num_ps: Number of PS servers to start.
    protocol: Communication protocol. Allowed values are documented in the
      documentation of `tf.distribute.Server`.
    worker_config: (optional) `tf.ConfigProto` to initialize workers. Can be
      used to instantiate multiple devices etc.
    ps_config: (optional) `tf.ConfigProto` to initialize PS servers.

  Returns:
    A tuple `(worker_servers, ps_servers)`.  `worker_servers` is a list
    of `num_workers` objects of type `tf.distribute.Server` (all running
    locally);
    and `ps_servers` is a list of `num_ps` objects of similar type.

  Raises:
    ImportError: if portpicker module was not found at load time
  """
  worker_ports = [pick_unused_port() for _ in range(num_workers)]
  ps_ports = [pick_unused_port() for _ in range(num_ps)]
  cluster_dict = {
      "worker": ["localhost:%s" % port for port in worker_ports],
      "ps": ["localhost:%s" % port for port in ps_ports]
  }
  cs = server_lib.ClusterSpec(cluster_dict)

  workers = [
      server_lib.Server(
          cs,
          job_name="worker",
          protocol=protocol,
          task_index=ix,
          config=worker_config,
          start=True) for ix in range(num_workers)
  ]
  ps_servers = [
      server_lib.Server(
          cs,
          job_name="ps",
          protocol=protocol,
          task_index=ix,
          config=ps_config,
          start=True) for ix in range(num_ps)
  ]

  return workers, ps_servers


def get_node_def_from_graph(node_name, graph_def):
  """Returns the `NodeDef` instance for given node name in the graph def.

  This method explores only the NodeDefs in `graph_def.node`.

  Args:
    node_name: Name of the NodeDef to search for.
    graph_def: An instance of `GraphDef` proto.

  Returns:
    the `NodeDef` instance whose name field matches the given node_name or None.
  """
  for node_def in graph_def.node:
    if node_def.name == node_name:
      return node_def
  return None


def set_producer_version(graph, producer_version):
  """Sets graph.graph_def_versions.producer to `producer_version`."""
  # The C API doesn't expose altering GraphDefVersions. We can indirectly set
  # it via import_graph_def though.
  graph_def = graph_pb2.GraphDef()
  graph_def.versions.producer = producer_version
  with graph.as_default():
    importer.import_graph_def(graph_def)
  assert graph.graph_def_versions.producer, producer_version


@contextlib.contextmanager
def _fake_gradient_tape_context_manager():
  """tf.gradients(...) implemented as tf.GradientTape context manager interface.

  This is useful to test tf.gradients() in tests that uses tf.GradientTape().

  Yields:
    gradient tape instance that's implemented by tf.gradients() underneath.
  """
  try:
    class FakeGradientTape:

      def watch(self, x):
        pass

      def gradient(self, y, x, grad_ys=None):
        result = gradients_impl.gradients(y, x, grad_ys)

        # Unlike `tape.gradient()`, `tf.gradients()` returns a list for a single
        # element. So unpack if needed to match `tape.gradient()` behavior.
        if not isinstance(x, (list, tuple)):
          assert len(result) == 1
          return result[0]

        return result

    yield FakeGradientTape()
  finally:
    pass


class AbstractGradientTape:
  """Abstract GradientTape context manager that has multiple implementations.

  This is useful to test both tf.GradientTape() and tf.gradients() without
  duplicating tests.
  """

  def __init__(self, use_tape, persistent=False):
    self._use_tape = use_tape
    self._persistent = persistent

  def __enter__(self) -> backprop.GradientTape:
    if self._use_tape:
      self._tape_impl = backprop.GradientTape(persistent=self._persistent)
    else:
      self._tape_impl = _fake_gradient_tape_context_manager()
    return self._tape_impl.__enter__()

  def __exit__(self, exc_type, exc_val, exc_tb):
    self._tape_impl.__exit__(exc_type, exc_val, exc_tb)


@contextlib.contextmanager
def run_functions_eagerly(run_eagerly):
  """Runs functions eagerly if `run_eagerly` is true.

  WARNING: Setting `run_eagerly` to True in tests running in V1 graph mode
  *WILL NOT* make the tf.function to run eagerly because eager is disabled by
  default in V1. Instead, tf.function will run as a traced graph function.

  Ensures that the state (for running functions eagerly) is back to the initial
  `def_function.RUN_FUNCTIONS_EAGERLY` state.

  Args:
    run_eagerly: Boolean determining whether to run the function eagerly or not.

  Raises:
    ValueError if `run_eagerly` is not a boolean.

  Yields:
    Nothing.
  """
  if not isinstance(run_eagerly, bool):
    raise ValueError(
        "Expected bool for `run_eagerly` but got {}".format(run_eagerly))

  is_eager = context.executing_eagerly()
  if not is_eager and run_eagerly:
    logging.warning(
        "Running tf.function eagerly in V1 graph mode is not supported. "
        "tf.function will be run as a traced graph function.")

  initial_state = def_function.functions_run_eagerly()
  def_function.run_functions_eagerly(run_eagerly)
  try:
    yield
  finally:
    def_function.run_functions_eagerly(initial_state)


class TestDelta:
  """A utility class to track increments to test counters."""

  def __init__(self, name, label):
    self.name = name
    self.label = label
    self.Reset()

  def Reset(self):
    self.last_value = _test_metrics_util.test_counter_value(
        self.name, self.label)

  def Get(self):
    value = _test_metrics_util.test_counter_value(self.name, self.label)
    return value - self.last_value


@tf_export("test.experimental.sync_devices")
def sync_devices():
  """Synchronizes all devices.

  By default, GPUs run asynchronously. This means that when you run an op on the
  GPU, like `tf.linalg.matmul`, the op may still be running on the GPU when the
  function returns. Non-GPU devices can also be made to run asynchronously by
  calling `tf.config.experimental.set_synchronous_execution(False)`. Calling
  `sync_devices()` blocks until pending ops have finished executing. This is
  primarily useful for measuring performance during a benchmark.

  For example, here is how you can measure how long `tf.linalg.matmul` runs:

  >>> import time
  >>> x = tf.random.normal((4096, 4096))
  >>> tf.linalg.matmul(x, x)  # Warmup.
  >>> tf.test.experimental.sync_devices()  # Block until warmup has completed.
  >>>
  >>> start = time.time()
  >>> y = tf.linalg.matmul(x, x)
  >>> tf.test.experimental.sync_devices()  # Block until matmul has completed.
  >>> end = time.time()
  >>> print(f'Time taken: {end - start}')

  If the call to `sync_devices()` was omitted, the time printed could be too
  small. This is because the op could still be running asynchronously when
  the line `end = time.time()` is executed.

  Raises:
    RuntimeError: If run outside Eager mode. This must be called in Eager mode,
      outside any `tf.function`s.
  """
  if not context.executing_eagerly():
    raise RuntimeError(
        "sync_devices() must only be called in Eager mode, outside tf.functions"
    )

  # There are two sources of asynchrony in TensorFlow:
  #
  # 1. On GPUs, kernels are run on a CUDA stream, which is inherently
  #    asynchronous.
  # 2. Calling `tf.config.experimental.set_synchronous_execution(False)` makes
  #    all ops asynchronous, in which case TensorFlow maintains internal queues
  #    of pending ops.
  #
  # Calling SyncDevice addresses source (1). Calling async_await addresses
  # source (2). It is important that SyncDevice() is called before async_wait(),
  # otherwise the SyncDevice op itself may still be pending on an internal
  # TensorFlow queue when the sync_devices() Python function returns.
  devices = config.list_logical_devices()
  for dev in devices:
    with ops.device(dev.name):
      gen_sync_ops.SyncDevice()
  context.async_wait()
