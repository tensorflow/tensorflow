# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the experimental input pipeline ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import warnings

from absl.testing import parameterized
import numpy as np

from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.compat import compat as forward_compat
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.data.util import structure
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training import server_lib
from tensorflow.python.training.checkpointable import util as checkpointable_utils
from tensorflow.python.util import compat


class IteratorTest(test.TestCase, parameterized.TestCase):

  def testNoGradients(self):
    component = constant_op.constant([1.])
    side = constant_op.constant(0.)
    add = lambda x: x + side
    dataset = dataset_ops.Dataset.from_tensor_slices(component).map(add)
    value = dataset.make_one_shot_iterator().get_next()
    self.assertIsNone(gradients_impl.gradients(value, component)[0])
    self.assertIsNone(gradients_impl.gradients(value, side)[0])
    self.assertIsNone(gradients_impl.gradients(value, [component, side])[0])

  def testCapturingStateInOneShotRaisesException(self):
    var = variables.Variable(37.0, name="myvar")
    dataset = (
        dataset_ops.Dataset.from_tensor_slices([0.0, 1.0, 2.0])
        .map(lambda x: x + var))
    with self.assertRaisesRegexp(
        ValueError, r"`Dataset.make_one_shot_iterator\(\)` does not support "
        "datasets that capture stateful objects.+myvar"):
      dataset.make_one_shot_iterator()

  def testOneShotIterator(self):
    components = (np.arange(7),
                  np.array([[1, 2, 3]]) * np.arange(7)[:, np.newaxis],
                  np.array(37.0) * np.arange(7))

    def _map_fn(x, y, z):
      return math_ops.square(x), math_ops.square(y), math_ops.square(z)

    iterator = (
        dataset_ops.Dataset.from_tensor_slices(components).map(_map_fn)
        .repeat(14).make_one_shot_iterator())
    get_next = iterator.get_next()

    self.assertEqual([c.shape[1:] for c in components],
                     [t.shape for t in get_next])

    with self.cached_session() as sess:
      for _ in range(14):
        for i in range(7):
          result = self.evaluate(get_next)
          for component, result_component in zip(components, result):
            self.assertAllEqual(component[i]**2, result_component)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testOneShotIteratorCaptureByValue(self):
    components = (np.arange(7),
                  np.array([[1, 2, 3]]) * np.arange(7)[:, np.newaxis],
                  np.array(37.0) * np.arange(7))
    tensor_components = tuple([ops.convert_to_tensor(c) for c in components])

    def _map_fn(x, y, z):
      return math_ops.square(x), math_ops.square(y), math_ops.square(z)

    iterator = (
        dataset_ops.Dataset.from_tensor_slices(tensor_components)
        .map(_map_fn).repeat(14).make_one_shot_iterator())
    get_next = iterator.get_next()

    self.assertEqual([c.shape[1:] for c in components],
                     [t.shape for t in get_next])

    with self.cached_session() as sess:
      for _ in range(14):
        for i in range(7):
          result = self.evaluate(get_next)
          for component, result_component in zip(components, result):
            self.assertAllEqual(component[i]**2, result_component)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testOneShotIteratorInsideContainer(self):
    components = (np.arange(7),
                  np.array([[1, 2, 3]]) * np.arange(7)[:, np.newaxis],
                  np.array(37.0) * np.arange(7))

    def within_container():

      def _map_fn(x, y, z):
        return math_ops.square(x), math_ops.square(y), math_ops.square(z)

      iterator = (
          dataset_ops.Dataset.from_tensor_slices(components)
          .map(_map_fn).repeat(14).make_one_shot_iterator())
      return iterator.get_next()

    server = server_lib.Server.create_local_server()

    # Create two iterators within unique containers, and run them to
    # make sure that the resources aren't shared.
    #
    # The test below would fail if cname were the same across both
    # sessions.
    for j in range(2):
      with session.Session(server.target) as sess:
        cname = "iteration%d" % j
        with ops.container(cname):
          get_next = within_container()

        for _ in range(14):
          for i in range(7):
            result = self.evaluate(get_next)
            for component, result_component in zip(components, result):
              self.assertAllEqual(component[i]**2, result_component)
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

  def testOneShotIteratorNonBlocking(self):
    dataset = dataset_ops.Dataset.from_tensors([1, 2, 3]).map(lambda x: x * x)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    # Create a session with a single thread to ensure that the
    # one-shot iterator initializer does not deadlock.
    config = config_pb2.ConfigProto(
        inter_op_parallelism_threads=1, use_per_session_threads=True)
    with session.Session(config=config) as sess:
      self.assertAllEqual([1, 4, 9], self.evaluate(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

    # Test with multiple threads invoking the one-shot iterator concurrently.
    with session.Session(config=config) as sess:
      results = []

      def consumer_thread():
        try:
          results.append(sess.run(next_element))
        except errors.OutOfRangeError:
          results.append(None)

      num_threads = 8
      threads = [
          self.checkedThread(consumer_thread) for _ in range(num_threads)
      ]
      for t in threads:
        t.start()
      for t in threads:
        t.join()

      self.assertEqual(num_threads, len(results))
      self.assertEqual(num_threads - 1,
                       len([None for r in results if r is None]))
      self.assertAllEqual([[1, 4, 9]], [r for r in results if r is not None])

  def testOneShotIteratorInitializerFails(self):
    # Define a dataset whose initialization will always fail.
    dataset = dataset_ops.Dataset.from_tensors(
        array_ops.check_numerics(
            constant_op.constant(1.0) / constant_op.constant(0.0), "oops"))
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with self.cached_session() as sess:
      with self.assertRaisesRegexp(errors.InvalidArgumentError, "oops"):
        sess.run(next_element)

      # Test that subsequent attempts to use the iterator also fail.
      with self.assertRaisesRegexp(errors.InvalidArgumentError, "oops"):
        sess.run(next_element)

    with self.cached_session() as sess:

      def consumer_thread():
        with self.assertRaisesRegexp(errors.InvalidArgumentError, "oops"):
          sess.run(next_element)

      num_threads = 8
      threads = [
          self.checkedThread(consumer_thread) for _ in range(num_threads)
      ]
      for t in threads:
        t.start()
      for t in threads:
        t.join()

  def testSimpleSharedResource(self):
    components = (np.array(1, dtype=np.int64),
                  np.array([1, 2, 3], dtype=np.int64),
                  np.array(37.0, dtype=np.float64))

    server = server_lib.Server.create_local_server()

    # Create two non-overlapping sessions that share the same iterator
    # resource on the same server, and verify that an action of the
    # first session (initializing the iterator) is visible in the
    # second session.
    with ops.Graph().as_default():
      iterator = (
          dataset_ops.Dataset.from_tensors(components)
          .map(lambda x, y, z: (x, y, z)).make_initializable_iterator(
              shared_name="shared_iterator"))
      init_op = iterator.initializer
      get_next = iterator.get_next()

      with session.Session(server.target) as sess:
        self.evaluate(init_op)
        results = self.evaluate(get_next)
        for component, result_component in zip(components, results):
          self.assertAllEqual(component, result_component)
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

        # Re-initialize the iterator in the first session.
        self.evaluate(init_op)

    with ops.Graph().as_default():
      # Re-define the iterator manually, without defining any of the
      # functions in this graph, to ensure that we are not
      # accidentally redefining functions with the same names in the
      # new graph.
      iterator = iterator_ops.Iterator.from_structure(
          shared_name="shared_iterator",
          output_types=(dtypes.int64, dtypes.int64, dtypes.float64),
          output_shapes=([], [3], []))
      get_next = iterator.get_next()

      with session.Session(server.target) as sess:
        # Use the iterator without re-initializing in the second session.
        results = self.evaluate(get_next)
        for component, result_component in zip(components, results):
          self.assertAllEqual(component, result_component)
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

  def testNotInitializedError(self):
    components = (np.array(1), np.array([1, 2, 3]), np.array(37.0))
    iterator = (
        dataset_ops.Dataset.from_tensors(components)
        .make_initializable_iterator())
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      with self.assertRaisesRegexp(errors.FailedPreconditionError,
                                   "iterator has not been initialized"):
        sess.run(get_next)

  def testReinitializableIterator(self):
    dataset_3 = dataset_ops.Dataset.from_tensors(
        constant_op.constant([1, 2, 3]))
    dataset_4 = dataset_ops.Dataset.from_tensors(
        constant_op.constant([4, 5, 6, 7]))
    iterator = iterator_ops.Iterator.from_structure(dataset_3.output_types,
                                                    [None])

    dataset_3_init_op = iterator.make_initializer(dataset_3)
    dataset_4_init_op = iterator.make_initializer(dataset_4)
    get_next = iterator.get_next()

    self.assertEqual(dataset_3.output_types, iterator.output_types)
    self.assertEqual(dataset_4.output_types, iterator.output_types)
    self.assertEqual([None], iterator.output_shapes.as_list())

    with self.cached_session() as sess:
      # The iterator is initially uninitialized.
      with self.assertRaises(errors.FailedPreconditionError):
        sess.run(get_next)

      # Initialize with one dataset.
      self.evaluate(dataset_3_init_op)
      self.assertAllEqual([1, 2, 3], self.evaluate(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

      # Initialize with a different dataset.
      self.evaluate(dataset_4_init_op)
      self.assertAllEqual([4, 5, 6, 7], self.evaluate(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

      # Reinitialize with the first dataset.
      self.evaluate(dataset_3_init_op)
      self.assertAllEqual([1, 2, 3], self.evaluate(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testReinitializableIteratorWithFunctions(self):

    def g():
      for i in range(10):
        yield i

    iterator = iterator_ops.Iterator.from_structure(dtypes.int64, [])
    next_element = iterator.get_next()

    with self.cached_session() as sess:
      dataset_1 = dataset_ops.Dataset.from_generator(
          g, output_types=dtypes.int64)
      sess.run(iterator.make_initializer(dataset_1))
      for expected in range(10):
        self.assertEqual(expected, self.evaluate(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

      dataset_2 = dataset_ops.Dataset.from_generator(
          g, output_types=dtypes.int64)
      sess.run(iterator.make_initializer(dataset_2))
      for expected in range(10):
        self.assertEqual(expected, self.evaluate(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  def testReinitializableIteratorStaticErrors(self):
    # Non-matching structure for types and shapes.
    with self.assertRaises(TypeError):
      iterator = iterator_ops.Iterator.from_structure(
          (dtypes.int64, dtypes.float64), [None])

    # Test validation of dataset argument.
    iterator = iterator_ops.Iterator.from_structure((dtypes.int64,
                                                     dtypes.float64))

    # Incompatible structure.
    with self.assertRaises(ValueError):
      iterator.make_initializer(
          dataset_ops.Dataset.from_tensors(((constant_op.constant(
              [1, 2, 3], dtype=dtypes.int64),), (constant_op.constant(
                  [4., 5., 6., 7.], dtype=dtypes.float64),))))

    # Incompatible types.
    with self.assertRaises(TypeError):
      iterator.make_initializer(
          dataset_ops.Dataset.from_tensors(
              (constant_op.constant([1, 2, 3], dtype=dtypes.int32),
               constant_op.constant([4., 5., 6., 7.], dtype=dtypes.float32))))

    # Incompatible shapes.
    iterator = iterator_ops.Iterator.from_structure(
        (dtypes.int64, dtypes.float64), ([None], []))
    with self.assertRaises(TypeError):
      iterator.make_initializer(
          dataset_ops.Dataset.from_tensors(
              (constant_op.constant([1, 2, 3], dtype=dtypes.int64),
               constant_op.constant([4., 5., 6., 7.], dtype=dtypes.float64))))

  def testIteratorStringHandle(self):
    dataset_3 = dataset_ops.Dataset.from_tensor_slices([1, 2, 3])
    dataset_4 = dataset_ops.Dataset.from_tensor_slices([10, 20, 30, 40])

    iterator_3 = dataset_3.make_one_shot_iterator()
    iterator_4 = dataset_4.make_one_shot_iterator()

    handle_placeholder = array_ops.placeholder(dtypes.string, shape=[])
    feedable_iterator = iterator_ops.Iterator.from_string_handle(
        handle_placeholder, dataset_3.output_types, dataset_3.output_shapes)
    next_element = feedable_iterator.get_next()

    self.assertEqual(dataset_3.output_types, feedable_iterator.output_types)
    self.assertEqual(dataset_4.output_types, feedable_iterator.output_types)
    self.assertEqual([], feedable_iterator.output_shapes)

    with self.cached_session() as sess:
      iterator_3_handle = sess.run(iterator_3.string_handle())
      iterator_4_handle = sess.run(iterator_4.string_handle())

      self.assertEqual(10,
                       sess.run(
                           next_element,
                           feed_dict={handle_placeholder: iterator_4_handle}))
      self.assertEqual(1,
                       sess.run(
                           next_element,
                           feed_dict={handle_placeholder: iterator_3_handle}))
      self.assertEqual(20,
                       sess.run(
                           next_element,
                           feed_dict={handle_placeholder: iterator_4_handle}))
      self.assertEqual(2,
                       sess.run(
                           next_element,
                           feed_dict={handle_placeholder: iterator_3_handle}))
      self.assertEqual(30,
                       sess.run(
                           next_element,
                           feed_dict={handle_placeholder: iterator_4_handle}))
      self.assertEqual(3,
                       sess.run(
                           next_element,
                           feed_dict={handle_placeholder: iterator_3_handle}))
      self.assertEqual(40,
                       sess.run(
                           next_element,
                           feed_dict={handle_placeholder: iterator_4_handle}))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(
            next_element, feed_dict={handle_placeholder: iterator_3_handle})
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(
            next_element, feed_dict={handle_placeholder: iterator_4_handle})

  def testIteratorStringHandleFuture(self):
    with forward_compat.forward_compatibility_horizon(2018, 8, 4):
      dataset_3 = dataset_ops.Dataset.from_tensor_slices([1, 2, 3])
      dataset_4 = dataset_ops.Dataset.from_tensor_slices([10, 20, 30, 40])

      iterator_3 = dataset_3.make_one_shot_iterator()
      iterator_4 = dataset_4.make_one_shot_iterator()

      handle_placeholder = array_ops.placeholder(dtypes.string, shape=[])
      feedable_iterator = iterator_ops.Iterator.from_string_handle(
          handle_placeholder, dataset_3.output_types, dataset_3.output_shapes)
      next_element = feedable_iterator.get_next()

      self.assertEqual(dataset_3.output_types, feedable_iterator.output_types)
      self.assertEqual(dataset_4.output_types, feedable_iterator.output_types)
      self.assertEqual([], feedable_iterator.output_shapes)

      with self.cached_session() as sess:
        iterator_3_handle = sess.run(iterator_3.string_handle())
        iterator_4_handle = sess.run(iterator_4.string_handle())

        self.assertEqual(
            10,
            sess.run(
                next_element,
                feed_dict={handle_placeholder: iterator_4_handle}))
        self.assertEqual(
            1,
            sess.run(
                next_element,
                feed_dict={handle_placeholder: iterator_3_handle}))
        self.assertEqual(
            20,
            sess.run(
                next_element,
                feed_dict={handle_placeholder: iterator_4_handle}))
        self.assertEqual(
            2,
            sess.run(
                next_element,
                feed_dict={handle_placeholder: iterator_3_handle}))
        self.assertEqual(
            30,
            sess.run(
                next_element,
                feed_dict={handle_placeholder: iterator_4_handle}))
        self.assertEqual(
            3,
            sess.run(
                next_element,
                feed_dict={handle_placeholder: iterator_3_handle}))
        self.assertEqual(
            40,
            sess.run(
                next_element,
                feed_dict={handle_placeholder: iterator_4_handle}))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(
              next_element, feed_dict={handle_placeholder: iterator_3_handle})
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(
              next_element, feed_dict={handle_placeholder: iterator_4_handle})

  def testIteratorStringHandleReuseTensorObject(self):
    dataset = dataset_ops.Dataset.from_tensor_slices([1, 2, 3])
    one_shot_iterator = dataset.make_one_shot_iterator()
    initializable_iterator = dataset.make_initializable_iterator()
    structure_iterator = iterator_ops.Iterator.from_structure(
        dataset.output_types)

    created_ops = len(ops.get_default_graph().get_operations())

    self.assertIs(one_shot_iterator.string_handle(),
                  one_shot_iterator.string_handle())
    self.assertIs(initializable_iterator.string_handle(),
                  initializable_iterator.string_handle())
    self.assertIs(structure_iterator.string_handle(),
                  structure_iterator.string_handle())

    # Assert that getting the (default) string handle creates no ops.
    self.assertEqual(created_ops, len(ops.get_default_graph().get_operations()))

    # Specifying an explicit name will create a new op.
    handle_with_name = one_shot_iterator.string_handle(name="foo")
    self.assertEqual("foo", handle_with_name.op.name)
    self.assertIsNot(one_shot_iterator.string_handle(), handle_with_name)

    handle_with_same_name = one_shot_iterator.string_handle(name="foo")
    self.assertEqual("foo_1", handle_with_same_name.op.name)
    self.assertIsNot(handle_with_name, handle_with_same_name)

  def testIteratorStringHandleError(self):
    dataset_int_scalar = (
        dataset_ops.Dataset.from_tensor_slices([1, 2, 3]).repeat())
    dataset_float_vector = (dataset_ops.Dataset.from_tensors([1.0, 2.0, 3.0]))

    handle_placeholder = array_ops.placeholder(dtypes.string, shape=[])

    feedable_int_scalar = iterator_ops.Iterator.from_string_handle(
        handle_placeholder, dtypes.int32, [])
    feedable_int_vector = iterator_ops.Iterator.from_string_handle(
        handle_placeholder, dtypes.int32, [None])
    feedable_int_any = iterator_ops.Iterator.from_string_handle(
        handle_placeholder, dtypes.int32)

    with self.cached_session() as sess:
      handle_int_scalar = sess.run(
          dataset_int_scalar.make_one_shot_iterator().string_handle())
      handle_float_vector = sess.run(
          dataset_float_vector.make_one_shot_iterator().string_handle())

      self.assertEqual(1,
                       sess.run(
                           feedable_int_scalar.get_next(),
                           feed_dict={handle_placeholder: handle_int_scalar}))

      self.assertEqual(2,
                       sess.run(
                           feedable_int_any.get_next(),
                           feed_dict={handle_placeholder: handle_int_scalar}))

      with self.assertRaises(errors.InvalidArgumentError):
        print(sess.run(
            feedable_int_vector.get_next(),
            feed_dict={handle_placeholder: handle_int_scalar}))

      with self.assertRaises(errors.InvalidArgumentError):
        print(sess.run(
            feedable_int_vector.get_next(),
            feed_dict={handle_placeholder: handle_float_vector}))

  def testRemoteIteratorUsingRemoteCallOpDirectSession(self):
    worker_config = config_pb2.ConfigProto()
    worker_config.device_count["CPU"] = 3

    with ops.device("/job:localhost/replica:0/task:0/cpu:1"):
      dataset_3 = dataset_ops.Dataset.from_tensor_slices([1, 2, 3])
      iterator_3 = dataset_3.make_one_shot_iterator()
      iterator_3_handle = iterator_3.string_handle()

    @function.Defun(dtypes.string)
    def _remote_fn(h):
      remote_iterator = iterator_ops.Iterator.from_string_handle(
          h, dataset_3.output_types, dataset_3.output_shapes)
      return remote_iterator.get_next()

    with ops.device("/job:localhost/replica:0/task:0/cpu:0"):
      target_placeholder = array_ops.placeholder(dtypes.string, shape=[])
      remote_op = functional_ops.remote_call(
          args=[iterator_3_handle],
          Tout=[dtypes.int32],
          f=_remote_fn,
          target=target_placeholder)

    with self.session(config=worker_config) as sess:
      elem = sess.run(
          remote_op,
          feed_dict={
              target_placeholder: "/job:localhost/replica:0/task:0/cpu:1"
          })
      self.assertEqual(elem, [1])
      # Fails when target is cpu:2 where the resource is not located.
      with self.assertRaises(errors.InvalidArgumentError):
        sess.run(
            remote_op,
            feed_dict={
                target_placeholder: "/job:localhost/replica:0/task:0/cpu:2"
            })
      elem = sess.run(
          remote_op,
          feed_dict={
              target_placeholder: "/job:localhost/replica:0/task:0/cpu:1"
          })
      self.assertEqual(elem, [2])
      elem = sess.run(
          remote_op,
          feed_dict={
              target_placeholder: "/job:localhost/replica:0/task:0/cpu:1"
          })
      self.assertEqual(elem, [3])
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(
            remote_op,
            feed_dict={
                target_placeholder: "/job:localhost/replica:0/task:0/cpu:1"
            })

  def testRemoteIteratorUsingRemoteCallOpMultiWorkers(self):
    s1 = server_lib.Server.create_local_server()
    s2 = server_lib.Server.create_local_server()
    s3 = server_lib.Server.create_local_server()

    cluster_def = cluster_pb2.ClusterDef()
    workers = cluster_def.job.add()
    workers.name = "worker"
    workers.tasks[0] = s1.target[len("grpc://"):]
    workers.tasks[1] = s2.target[len("grpc://"):]
    client = cluster_def.job.add()
    client.name = "client"
    client.tasks[0] = s3.target[len("grpc://"):]
    config = config_pb2.ConfigProto(cluster_def=cluster_def)

    worker_devices = [
        "/job:worker/replica:0/task:%d/cpu:0" % i for i in range(2)
    ]
    itr_handles = []
    for device in worker_devices:
      with ops.device(device):
        src = dataset_ops.Dataset.from_tensor_slices([device])
        itr = src.make_one_shot_iterator()
        itr_handles.append(itr.string_handle())

    targets = dataset_ops.Dataset.from_tensor_slices(worker_devices)
    handles = dataset_ops.Dataset.from_tensor_slices(itr_handles)

    @function.Defun(dtypes.string)
    def loading_func(h):
      remote_itr = iterator_ops.Iterator.from_string_handle(
          h, itr.output_types, itr.output_shapes)
      return remote_itr.get_next()

    def map_fn(target, handle):
      return functional_ops.remote_call(
          args=[handle], Tout=[dtypes.string], f=loading_func, target=target)

    with ops.device("/job:client"):
      client_dataset = dataset_ops.Dataset.zip((targets, handles)).map(map_fn)
      itr = client_dataset.make_initializable_iterator()
      n = itr.get_next()

    with session.Session(s3.target, config=config) as sess:
      self.evaluate(itr.initializer)
      expected_values = worker_devices
      for expected in expected_values:
        self.assertEqual((compat.as_bytes(expected),), self.evaluate(n))

      with self.assertRaises(errors.OutOfRangeError):
        sess.run(n)

  def testRemoteIteratorUsingRemoteCallOpDirectSessionGPUCPU(self):
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    with ops.device("/job:localhost/replica:0/task:0/cpu:0"):
      dataset_3 = dataset_ops.Dataset.from_tensor_slices([1, 2, 3])
      iterator_3 = dataset_3.make_one_shot_iterator()
      iterator_3_handle = iterator_3.string_handle()

    def _encode_raw(byte_array):
      return bytes(bytearray(byte_array))

    @function.Defun(dtypes.uint8)
    def _remote_fn(h):
      handle = script_ops.py_func(_encode_raw, [h], dtypes.string)
      remote_iterator = iterator_ops.Iterator.from_string_handle(
          handle, dataset_3.output_types, dataset_3.output_shapes)
      return remote_iterator.get_next()

    with ops.device("/job:localhost/replica:0/task:0/device:GPU:0"):
      target_placeholder = array_ops.placeholder(dtypes.string, shape=[])
      iterator_3_handle_uint8 = parsing_ops.decode_raw(
          bytes=iterator_3_handle, out_type=dtypes.uint8)
      remote_op = functional_ops.remote_call(
          args=[iterator_3_handle_uint8],
          Tout=[dtypes.int32],
          f=_remote_fn,
          target=target_placeholder)

    with self.cached_session() as sess:
      elem = sess.run(
          remote_op,
          feed_dict={
              target_placeholder: "/job:localhost/replica:0/task:0/cpu:0"
          })
      self.assertEqual(elem, [1])
      elem = sess.run(
          remote_op,
          feed_dict={
              target_placeholder: "/job:localhost/replica:0/task:0/cpu:0"
          })
      self.assertEqual(elem, [2])
      elem = sess.run(
          remote_op,
          feed_dict={
              target_placeholder: "/job:localhost/replica:0/task:0/cpu:0"
          })
      self.assertEqual(elem, [3])
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(
            remote_op,
            feed_dict={
                target_placeholder: "/job:localhost/replica:0/task:0/cpu:0"
            })

  def testIncorrectIteratorRestore(self):

    def _path():
      return os.path.join(self.get_temp_dir(), "iterator")

    def _save_op(iterator_resource):
      iterator_state_variant = gen_dataset_ops.serialize_iterator(
          iterator_resource)
      save_op = io_ops.write_file(
          _path(), parsing_ops.serialize_tensor(iterator_state_variant))
      return save_op

    def _restore_op(iterator_resource):
      iterator_state_variant = parsing_ops.parse_tensor(
          io_ops.read_file(_path()), dtypes.variant)
      restore_op = gen_dataset_ops.deserialize_iterator(iterator_resource,
                                                        iterator_state_variant)
      return restore_op

    def _build_range_dataset_graph():
      start = 1
      stop = 10
      iterator = dataset_ops.Dataset.range(start,
                                           stop).make_initializable_iterator()
      init_op = iterator.initializer
      get_next = iterator.get_next()
      save_op = _save_op(iterator._iterator_resource)
      restore_op = _restore_op(iterator._iterator_resource)
      return init_op, get_next, save_op, restore_op

    def _build_reader_dataset_graph():
      filenames = ["test"]  # Does not exist but we don't care in this test.
      iterator = readers.FixedLengthRecordDataset(
          filenames, 1, 0, 0).make_initializable_iterator()
      init_op = iterator.initializer
      get_next_op = iterator.get_next()
      save_op = _save_op(iterator._iterator_resource)
      restore_op = _restore_op(iterator._iterator_resource)
      return init_op, get_next_op, save_op, restore_op

    # Saving iterator for RangeDataset graph.
    with ops.Graph().as_default() as g:
      init_op, _, save_op, _ = _build_range_dataset_graph()
      with self.session(graph=g) as sess:
        self.evaluate(init_op)
        self.evaluate(save_op)

    # Attempt to restore the saved iterator into an IteratorResource of
    # incompatible type. An iterator of RangeDataset has output type int64,
    # while an iterator of FixedLengthRecordDataset has output type string.
    # So an InvalidArgumentError should be raised by
    # IteratorResource::set_iterator.
    with ops.Graph().as_default() as g:
      _, _, _, restore_op = _build_reader_dataset_graph()
      with self.session(graph=g) as sess:
        with self.assertRaises(errors.InvalidArgumentError):
          self.evaluate(restore_op)

  def testRepeatedGetNextWarning(self):
    iterator = dataset_ops.Dataset.range(10).make_one_shot_iterator()
    warnings.simplefilter("always")
    with warnings.catch_warnings(record=True) as w:
      for _ in range(100):
        iterator.get_next()
    self.assertEqual(100 - iterator_ops.GET_NEXT_CALL_WARNING_THRESHOLD, len(w))
    for warning in w:
      self.assertIn(
          iterator_ops.GET_NEXT_CALL_WARNING_MESSAGE, str(warning.message))

  def testEagerIteratorAsync(self):
    with context.eager_mode(), context.execution_mode(context.ASYNC):
      val = 0
      dataset = dataset_ops.Dataset.range(10)
      for foo in dataset:
        self.assertEqual(val, foo.numpy())
        val += 1

  # pylint: disable=g-long-lambda
  @parameterized.named_parameters(
      ("Tensor", lambda: constant_op.constant(37.0),
       structure.TensorStructure(dtypes.float32, []),
       ops.Tensor, dtypes.float32, []),
      ("SparseTensor", lambda: sparse_tensor.SparseTensor(
          indices=[[0]], values=constant_op.constant([0], dtype=dtypes.int32),
          dense_shape=[1]),
       structure.SparseTensorStructure(dtypes.int32, [1]),
       sparse_tensor.SparseTensor, dtypes.int32, [1]),
      ("Nest", lambda: {
          "a": constant_op.constant(37.0),
          "b": (constant_op.constant(["Foo"]), constant_op.constant("Bar"))},
       structure.NestedStructure({
           "a": structure.TensorStructure(dtypes.float32, []),
           "b": (structure.TensorStructure(dtypes.string, [1]),
                 structure.TensorStructure(dtypes.string, []))}),
       {"a": ops.Tensor, "b": (ops.Tensor, ops.Tensor)},
       {"a": dtypes.float32, "b": (dtypes.string, dtypes.string)},
       {"a": [], "b": ([1], [])}),
  )
  def testIteratorStructure(self, tf_value_fn, expected_element_structure,
                            expected_output_classes, expected_output_types,
                            expected_output_shapes):
    tf_value = tf_value_fn()
    iterator = dataset_ops.Dataset.from_tensors(
        tf_value).make_one_shot_iterator()

    self.assertTrue(expected_element_structure.is_compatible_with(
        iterator._element_structure))
    self.assertTrue(iterator._element_structure.is_compatible_with(
        expected_element_structure))

    self.assertEqual(expected_output_classes, iterator.output_classes)
    self.assertEqual(expected_output_types, iterator.output_types)
    self.assertEqual(expected_output_shapes, iterator.output_shapes)

  def testIteratorGetNextName(self):
    with ops.Graph().as_default():
      iterator = dataset_ops.Dataset.from_tensors(37.0).make_one_shot_iterator()
      next_element = iterator.get_next(name="overridden_name")
      self.assertEqual("overridden_name", next_element.op.name)


class IteratorCheckpointingTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testSaveRestoreOneShotIterator(self):
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    dataset = dataset_ops.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6]).map(
        math_ops.square).batch(2)
    iterator = dataset.make_one_shot_iterator()
    get_next = iterator.get_next if context.executing_eagerly(
    ) else functools.partial(self.evaluate, iterator.get_next())
    checkpoint = checkpointable_utils.Checkpoint(iterator=iterator)
    self.assertAllEqual([1, 4], get_next())
    save_path = checkpoint.save(checkpoint_prefix)
    self.assertAllEqual([9, 16], get_next())
    self.assertAllEqual([25, 36], get_next())
    checkpoint.restore(save_path).run_restore_ops()
    self.assertAllEqual([9, 16], get_next())
    self.assertAllEqual([25, 36], get_next())
    with self.assertRaises(errors.OutOfRangeError):
      get_next()

  @test_util.run_in_graph_and_eager_modes
  def testSaveRestoreMultipleIterator(self):
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    dataset = dataset_ops.Dataset.from_tensor_slices(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    dataset = dataset.map(math_ops.square).batch(2)
    iterator_1 = dataset.make_one_shot_iterator()
    get_next_1 = iterator_1.get_next if context.executing_eagerly(
    ) else functools.partial(self.evaluate, iterator_1.get_next())
    iterator_2 = dataset.make_one_shot_iterator()
    get_next_2 = iterator_2.get_next if context.executing_eagerly(
    ) else functools.partial(self.evaluate, iterator_2.get_next())
    dataset_2 = dataset_ops.Dataset.range(10)
    iterator_3 = dataset_2.make_one_shot_iterator()
    get_next_3 = iterator_3.get_next if context.executing_eagerly(
    ) else functools.partial(self.evaluate, iterator_3.get_next())
    checkpoint = checkpointable_utils.Checkpoint(
        iterator_1=iterator_1, iterator_2=iterator_2, iterator_3=iterator_3)
    self.assertAllEqual([1, 4], get_next_1())
    self.assertAllEqual(0, get_next_3())
    self.assertAllEqual(1, get_next_3())
    self.assertAllEqual(2, get_next_3())
    save_path = checkpoint.save(checkpoint_prefix)
    self.assertAllEqual([1, 4], get_next_2())
    self.assertAllEqual([9, 16], get_next_2())
    self.assertAllEqual(3, get_next_3())
    checkpoint.restore(save_path).run_restore_ops()
    self.assertAllEqual([9, 16], get_next_1())
    self.assertAllEqual([1, 4], get_next_2())
    self.assertAllEqual(3, get_next_3())

  @test_util.run_in_graph_and_eager_modes
  def testRestoreExhaustedIterator(self):
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    dataset = dataset_ops.Dataset.range(3)
    iterator = dataset.make_one_shot_iterator()
    get_next = iterator.get_next if context.executing_eagerly(
    ) else functools.partial(self.evaluate, iterator.get_next())
    checkpoint = checkpointable_utils.Checkpoint(iterator=iterator)
    self.assertAllEqual(0, get_next())
    self.assertAllEqual(1, get_next())
    save_path = checkpoint.save(checkpoint_prefix)
    self.assertAllEqual(2, get_next())
    checkpoint.restore(save_path).run_restore_ops()
    self.assertAllEqual(2, get_next())
    save_path = checkpoint.save(checkpoint_prefix)
    checkpoint.restore(save_path).run_restore_ops()
    with self.assertRaises(errors.OutOfRangeError):
      get_next()

  def testRestoreInReconstructedIteratorInitializable(self):
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    dataset = dataset_ops.Dataset.range(10)
    iterator = dataset.make_initializable_iterator()
    get_next = iterator.get_next()
    checkpoint = checkpointable_utils.Checkpoint(iterator=iterator)
    for i in range(5):
      with self.cached_session() as sess:
        checkpoint.restore(checkpoint_management.latest_checkpoint(
            checkpoint_directory)).initialize_or_restore(sess)
        for j in range(2):
          self.assertEqual(i * 2 + j, self.evaluate(get_next))
        checkpoint.save(file_prefix=checkpoint_prefix)


if __name__ == "__main__":
  test.main()
