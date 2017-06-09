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

"""Tests for tensorflow.python.client.session.Session's partial run APIs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.client import session
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest
from tensorflow.python.training import server_lib

ops._USE_C_API = True

# NOTE(mrry): Dummy shape registration for ops used in the tests, since they
# don't have C++ op registrations on which to attach C++ shape fns.
ops.RegisterShape('ConstructionFails')(common_shapes.unknown_shape)


class PartialRunTest(test_util.TensorFlowTestCase):

  def setUp(self):
    # Partial runs don't work with C API
    ops._USE_C_API = False
    super(PartialRunTest, self).setUp()

  def tearDown(self):
    ops._USE_C_API = True
    super(PartialRunTest, self).tearDown()

  def runTestPartialRun(self, sess):
    a = array_ops.placeholder(dtypes.float32, shape=[])
    b = array_ops.placeholder(dtypes.float32, shape=[])
    c = array_ops.placeholder(dtypes.float32, shape=[])
    r1 = math_ops.add(a, b)
    r2 = math_ops.multiply(r1, c)

    h = sess.partial_run_setup([r1, r2], [a, b, c])
    res = sess.partial_run(h, r1, feed_dict={a: 1, b: 2})
    self.assertEqual(3, res)
    temp = res * 17
    res = sess.partial_run(h, r2, feed_dict={c: temp})
    self.assertEqual(153, res)

    # Call again on the same graph.
    h2 = sess.partial_run_setup([r1, r2], [a, b, c])
    res = sess.partial_run(h2, r1, feed_dict={a: 1, b: 2})
    self.assertEqual(3, res)
    temp = res * 18
    res = sess.partial_run(h2, r2, feed_dict={c: temp})
    self.assertEqual(162, res)

  def runTestPartialRunIncomplete(self, sess):
    a = array_ops.placeholder(dtypes.float32, shape=[])
    b = array_ops.placeholder(dtypes.float32, shape=[])
    c = array_ops.placeholder(dtypes.float32, shape=[])
    r1 = math_ops.add(a, b)
    r2 = math_ops.multiply(r1, c)

    h = sess.partial_run_setup([r1, r2], [a, b, c])
    res = sess.partial_run(h, r1, feed_dict={a: 1, b: 2})
    self.assertEqual(3, res)

  def runTestConcurrentPartialRun(self, sess):
    a = array_ops.placeholder(dtypes.float32, shape=[])
    b = array_ops.placeholder(dtypes.float32, shape=[])
    c = array_ops.placeholder(dtypes.float32, shape=[])
    r1 = math_ops.add(a, b)
    r2 = math_ops.multiply(r1, c)

    h1 = sess.partial_run_setup([r1], [a, b, c])
    h2 = sess.partial_run_setup([r1, r2], [a, b, c])
    res = sess.partial_run(h1, r1, feed_dict={a: 1, b: 2})
    self.assertEqual(3, res)
    temp = res * 19
    res = sess.partial_run(h2, r1, feed_dict={a: temp, b: 9})
    self.assertEqual(66, res)
    res = sess.partial_run(h2, r2, feed_dict={c: 7})
    self.assertEqual(462, res)

  def runTestManyPartialRun(self, sess):
    steps = 200
    inputs = []
    outputs = []
    a = constant_op.constant(2.0, dtypes.float32)
    for i in xrange(steps):
      inputs.append(array_ops.placeholder(dtypes.float32, shape=[]))
      a = math_ops.multiply(a, inputs[i])
      outputs.append(a)

    h = sess.partial_run_setup(outputs, inputs)
    for i in xrange(steps):
      res = sess.partial_run(h, outputs[i], feed_dict={inputs[i]: 1.0})
    self.assertEqual(2.0, res)

    feed_dict = {}
    for i in xrange(steps):
      feed_dict[inputs[i]] = 1.0
    res = sess.run(outputs, feed_dict)
    self.assertEqual(steps, len(res))
    self.assertEqual(2.0, res[-1])

  def runTestRunAndPartialRun(self, sess):
    a = constant_op.constant(2.0, dtypes.float32)
    b = a * 2
    c = b * 3
    r1 = sess.run([b, c])
    h = sess.partial_run_setup([b, c], [])
    r2 = sess.partial_run(h, [b, c])
    self.assertEqual(r1, r2)

  def runTestPartialRunMissingPlaceholderFeedException(self, sess):
    x = array_ops.placeholder(dtypes.float32, shape=())
    fetches = [x * 2, x * 3]
    handle = sess.partial_run_setup(fetches=fetches, feeds=[])
    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 'You must feed a value for placeholder'):
      sess.partial_run(handle, fetches[0])

  def runTestPartialRunUnspecifiedFeed(self, sess):
    a = array_ops.placeholder(dtypes.float32, shape=[])
    b = array_ops.placeholder(dtypes.float32, shape=[])
    c = array_ops.placeholder(dtypes.float32, shape=[])
    r1 = math_ops.add(a, b)

    h = sess.partial_run_setup([r1], [a, b])
    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 'was not specified in partial_run_setup.$'):
      sess.partial_run(h, r1, feed_dict={a: 1, b: 2, c: 3})

  def runTestPartialRunUnspecifiedFetch(self, sess):
    a = array_ops.placeholder(dtypes.float32, shape=[])
    b = array_ops.placeholder(dtypes.float32, shape=[])
    c = array_ops.placeholder(dtypes.float32, shape=[])
    r1 = math_ops.add(a, b)
    r2 = math_ops.multiply(a, c)

    h = sess.partial_run_setup([r1], [a, b, c])
    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 'was not specified in partial_run_setup.$'):
      sess.partial_run(h, r2, feed_dict={a: 1, c: 3})

  def runTestPartialRunAlreadyFed(self, sess):
    a = array_ops.placeholder(dtypes.float32, shape=[])
    b = array_ops.placeholder(dtypes.float32, shape=[])
    c = array_ops.placeholder(dtypes.float32, shape=[])
    r1 = math_ops.add(a, b)
    r2 = math_ops.multiply(a, c)

    h = sess.partial_run_setup([r1, r2], [a, b, c])
    sess.partial_run(h, r1, feed_dict={a: 1, b: 2})
    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 'has already been fed.$'):
      sess.partial_run(h, r2, feed_dict={a: 1, c: 3})

  def runTestPartialRunAlreadyFetched(self, sess):
    a = array_ops.placeholder(dtypes.float32, shape=[])
    b = array_ops.placeholder(dtypes.float32, shape=[])
    c = array_ops.placeholder(dtypes.float32, shape=[])
    r1 = math_ops.add(a, b)
    r2 = math_ops.multiply(a, c)

    h = sess.partial_run_setup([r1, r2], [a, b, c])
    sess.partial_run(h, r1, feed_dict={a: 1, b: 2})
    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 'has already been fetched.$'):
      sess.partial_run(h, r1, feed_dict={c: 3})

  def runTestPartialRunEmptyFetches(self, sess):
    a = array_ops.placeholder(dtypes.float32)
    b = a * 2.0

    h = sess.partial_run_setup(fetches=[b], feeds=[a])
    sess.partial_run(h, [], {a: 3.0})
    r = sess.partial_run(h, [b], {})
    self.assertEqual([6.0], r)

  def testInvalidPartialRunSetup(self):
    sess = session.Session()
    x = array_ops.placeholder(dtypes.float32, shape=[])
    with self.assertRaisesRegexp(
        errors.InvalidArgumentError,
        'specify at least one target to fetch or execute.'):
      sess.partial_run_setup(fetches=[], feeds=[x])

  def testPartialRunDirect(self):
    self.runTestPartialRun(session.Session())

  def testPartialRunIncompleteDirect(self):
    self.runTestPartialRunIncomplete(session.Session())

  def testConcurrentPartialRunDirect(self):
    self.runTestConcurrentPartialRun(session.Session())

  def testManyPartialRunDirect(self):
    self.runTestManyPartialRun(session.Session())

  def testRunAndPartialRunDirect(self):
    self.runTestRunAndPartialRun(session.Session())

  def testPartialRunMissingPlaceholderFeedExceptionDirect(self):
    self.runTestPartialRunMissingPlaceholderFeedException(session.Session())

  def testPartialRunUnspecifiedFeedDirect(self):
    self.runTestPartialRunUnspecifiedFeed(session.Session())

  def testPartialRunUnspecifiedFetchDirect(self):
    self.runTestPartialRunUnspecifiedFetch(session.Session())

  def testPartialRunAlreadyFedDirect(self):
    self.runTestPartialRunAlreadyFed(session.Session())

  def testPartialRunAlreadyFetchedDirect(self):
    self.runTestPartialRunAlreadyFetched(session.Session())

  def testPartialRunEmptyFetchesDirect(self):
    self.runTestPartialRunEmptyFetches(session.Session())

  def testPartialRunDist(self):
    server = server_lib.Server.create_local_server()
    self.runTestPartialRun(session.Session(server.target))

  def testPartialRunIncompleteDist(self):
    server = server_lib.Server.create_local_server()
    self.runTestPartialRunIncomplete(session.Session(server.target))

  def testConcurrentPartialRunDist(self):
    server = server_lib.Server.create_local_server()
    self.runTestConcurrentPartialRun(session.Session(server.target))

  def testManyPartialRunDist(self):
    server = server_lib.Server.create_local_server()
    self.runTestManyPartialRun(session.Session(server.target))

  def testRunAndPartialRunDist(self):
    server = server_lib.Server.create_local_server()
    self.runTestRunAndPartialRun(session.Session(server.target))

  def testPartialRunMissingPlaceholderFeedExceptionDist(self):
    server = server_lib.Server.create_local_server()
    self.runTestPartialRunMissingPlaceholderFeedException(
        session.Session(server.target))

  def testPartialRunUnspecifiedFeedDist(self):
    server = server_lib.Server.create_local_server()
    self.runTestPartialRunUnspecifiedFeed(session.Session(server.target))

  def testPartialRunUnspecifiedFetchDist(self):
    server = server_lib.Server.create_local_server()
    self.runTestPartialRunUnspecifiedFetch(session.Session(server.target))

  def testPartialRunAlreadyFedDist(self):
    server = server_lib.Server.create_local_server()
    self.runTestPartialRunAlreadyFed(session.Session(server.target))

  def testPartialRunAlreadyFetchedDist(self):
    server = server_lib.Server.create_local_server()
    self.runTestPartialRunAlreadyFetched(session.Session(server.target))

  def testPartialRunEmptyFetchesDist(self):
    server = server_lib.Server.create_local_server()
    self.runTestPartialRunEmptyFetches(session.Session(server.target))


if __name__ == '__main__':
  googletest.main()
