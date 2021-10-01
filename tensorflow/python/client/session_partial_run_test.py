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

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest
from tensorflow.python.training import server_lib


class PartialRunTest(test_util.TensorFlowTestCase):

  def RunTestPartialRun(self, sess):
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

  def RunTestPartialRunIncomplete(self, sess):
    a = array_ops.placeholder(dtypes.float32, shape=[])
    b = array_ops.placeholder(dtypes.float32, shape=[])
    c = array_ops.placeholder(dtypes.float32, shape=[])
    r1 = math_ops.add(a, b)
    r2 = math_ops.multiply(r1, c)

    h = sess.partial_run_setup([r1, r2], [a, b, c])
    res = sess.partial_run(h, r1, feed_dict={a: 1, b: 2})
    self.assertEqual(3, res)

  def RunTestConcurrentPartialRun(self, sess):
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

  def RunTestManyPartialRun(self, sess):
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

  def RunTestRunAndPartialRun(self, sess):
    a = constant_op.constant(2.0, dtypes.float32)
    b = a * 2
    c = b * 3
    r1 = self.evaluate([b, c])
    h = sess.partial_run_setup([b, c], [])
    r2 = sess.partial_run(h, [b, c])
    self.assertEqual(r1, r2)

  def RunTestPartialRunMissingPlaceholderFeedException(self, sess):
    x = array_ops.placeholder(dtypes.float32, shape=())
    fetches = [x * 2, x * 3]
    handle = sess.partial_run_setup(fetches=fetches, feeds=[])
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                'You must feed a value for placeholder'):
      sess.partial_run(handle, fetches[0])

  def RunTestPartialRunUnspecifiedFeed(self, sess):
    a = array_ops.placeholder(dtypes.float32, shape=[])
    b = array_ops.placeholder(dtypes.float32, shape=[])
    c = array_ops.placeholder(dtypes.float32, shape=[])
    r1 = math_ops.add(a, b)

    h = sess.partial_run_setup([r1], [a, b])
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                'was not specified in partial_run_setup.$'):
      sess.partial_run(h, r1, feed_dict={a: 1, b: 2, c: 3})

  def RunTestPartialRunUnspecifiedFetch(self, sess):
    a = array_ops.placeholder(dtypes.float32, shape=[])
    b = array_ops.placeholder(dtypes.float32, shape=[])
    c = array_ops.placeholder(dtypes.float32, shape=[])
    r1 = math_ops.add(a, b)
    r2 = math_ops.multiply(a, c)

    h = sess.partial_run_setup([r1], [a, b, c])
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                'was not specified in partial_run_setup.$'):
      sess.partial_run(h, r2, feed_dict={a: 1, c: 3})

  def RunTestPartialRunAlreadyFed(self, sess):
    a = array_ops.placeholder(dtypes.float32, shape=[])
    b = array_ops.placeholder(dtypes.float32, shape=[])
    c = array_ops.placeholder(dtypes.float32, shape=[])
    r1 = math_ops.add(a, b)
    r2 = math_ops.multiply(a, c)

    h = sess.partial_run_setup([r1, r2], [a, b, c])
    sess.partial_run(h, r1, feed_dict={a: 1, b: 2})
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                'has already been fed.$'):
      sess.partial_run(h, r2, feed_dict={a: 1, c: 3})

  def RunTestPartialRunAlreadyFetched(self, sess):
    a = array_ops.placeholder(dtypes.float32, shape=[])
    b = array_ops.placeholder(dtypes.float32, shape=[])
    c = array_ops.placeholder(dtypes.float32, shape=[])
    r1 = math_ops.add(a, b)
    r2 = math_ops.multiply(a, c)

    h = sess.partial_run_setup([r1, r2], [a, b, c])
    sess.partial_run(h, r1, feed_dict={a: 1, b: 2})
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                'has already been fetched.$'):
      sess.partial_run(h, r1, feed_dict={c: 3})

  def RunTestPartialRunEmptyFetches(self, sess):
    a = array_ops.placeholder(dtypes.float32)
    b = a * 2.0

    h = sess.partial_run_setup(fetches=[b], feeds=[a])
    sess.partial_run(h, [], {a: 3.0})
    r = sess.partial_run(h, [b], {})
    self.assertEqual([6.0], r)

  @test_util.run_deprecated_v1
  def testInvalidPartialRunSetup(self):
    sess = session.Session()
    x = array_ops.placeholder(dtypes.float32, shape=[])
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        'specify at least one target to fetch or execute.'):
      sess.partial_run_setup(fetches=[], feeds=[x])

  @test_util.run_deprecated_v1
  def testPartialRunSetupNoFeedsPassed(self):
    sess = session.Session()
    r1 = constant_op.constant([6.0])

    h = sess.partial_run_setup([r1])
    result1 = sess.partial_run(h, r1)
    self.assertEqual([6.0], result1)

  @test_util.run_deprecated_v1
  def testPartialRunDirect(self):
    self.RunTestPartialRun(session.Session())

  @test_util.run_deprecated_v1
  def testPartialRunIncompleteDirect(self):
    self.RunTestPartialRunIncomplete(session.Session())

  @test_util.run_deprecated_v1
  def testConcurrentPartialRunDirect(self):
    self.RunTestConcurrentPartialRun(session.Session())

  @test_util.run_deprecated_v1
  def testManyPartialRunDirect(self):
    self.RunTestManyPartialRun(session.Session())

  @test_util.run_deprecated_v1
  def testRunAndPartialRunDirect(self):
    self.RunTestRunAndPartialRun(session.Session())

  @test_util.run_deprecated_v1
  def testPartialRunMissingPlaceholderFeedExceptionDirect(self):
    self.RunTestPartialRunMissingPlaceholderFeedException(session.Session())

  @test_util.run_deprecated_v1
  def testPartialRunUnspecifiedFeedDirect(self):
    self.RunTestPartialRunUnspecifiedFeed(session.Session())

  @test_util.run_deprecated_v1
  def testPartialRunUnspecifiedFetchDirect(self):
    self.RunTestPartialRunUnspecifiedFetch(session.Session())

  @test_util.run_deprecated_v1
  def testPartialRunAlreadyFedDirect(self):
    self.RunTestPartialRunAlreadyFed(session.Session())

  @test_util.run_deprecated_v1
  def testPartialRunAlreadyFetchedDirect(self):
    self.RunTestPartialRunAlreadyFetched(session.Session())

  @test_util.run_deprecated_v1
  def testPartialRunEmptyFetchesDirect(self):
    self.RunTestPartialRunEmptyFetches(session.Session())

  @test_util.run_deprecated_v1
  def testPartialRunDist(self):
    server = server_lib.Server.create_local_server()
    self.RunTestPartialRun(session.Session(server.target))

  @test_util.run_deprecated_v1
  def testPartialRunIncompleteDist(self):
    server = server_lib.Server.create_local_server()
    self.RunTestPartialRunIncomplete(session.Session(server.target))

  @test_util.run_deprecated_v1
  def testConcurrentPartialRunDist(self):
    server = server_lib.Server.create_local_server()
    self.RunTestConcurrentPartialRun(session.Session(server.target))

  @test_util.run_deprecated_v1
  def testManyPartialRunDist(self):
    server = server_lib.Server.create_local_server()
    self.RunTestManyPartialRun(session.Session(server.target))

  @test_util.run_deprecated_v1
  def testRunAndPartialRunDist(self):
    server = server_lib.Server.create_local_server()
    self.RunTestRunAndPartialRun(session.Session(server.target))

  @test_util.run_deprecated_v1
  def testPartialRunMissingPlaceholderFeedExceptionDist(self):
    server = server_lib.Server.create_local_server()
    self.RunTestPartialRunMissingPlaceholderFeedException(
        session.Session(server.target))

  @test_util.run_deprecated_v1
  def testPartialRunUnspecifiedFeedDist(self):
    server = server_lib.Server.create_local_server()
    self.RunTestPartialRunUnspecifiedFeed(session.Session(server.target))

  @test_util.run_deprecated_v1
  def testPartialRunUnspecifiedFetchDist(self):
    server = server_lib.Server.create_local_server()
    self.RunTestPartialRunUnspecifiedFetch(session.Session(server.target))

  @test_util.run_deprecated_v1
  def testPartialRunAlreadyFedDist(self):
    server = server_lib.Server.create_local_server()
    self.RunTestPartialRunAlreadyFed(session.Session(server.target))

  @test_util.run_deprecated_v1
  def testPartialRunAlreadyFetchedDist(self):
    server = server_lib.Server.create_local_server()
    self.RunTestPartialRunAlreadyFetched(session.Session(server.target))

  @test_util.run_deprecated_v1
  def testPartialRunEmptyFetchesDist(self):
    server = server_lib.Server.create_local_server()
    self.RunTestPartialRunEmptyFetches(session.Session(server.target))


if __name__ == '__main__':
  googletest.main()
