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
"""Tests for scalar strictness and scalar leniency."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sparse_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


# TODO(rmlarsen) : Remove this test completely after we stop supporting GraphDef
# version 5 and remove support of legacy scalars from Concat, Fill, Range,
# and Reshape.
class ScalarTest(test.TestCase):

  def check(self, op, args, error, correct=None, lenient=None, strict=[5, 6]):
    if lenient is None:
      lenient = []
    # Use placeholders to bypass shape inference, since only the C++
    # G raphDef level is ever scalar lenient.
    def placeholders(args, feed):
      if isinstance(args, tuple):
        return [placeholders(x, feed) for x in args]
      else:
        x = ops.convert_to_tensor(args).eval()
        fake = array_ops.placeholder(np.asarray(x).dtype)
        feed[fake] = x
        return fake

    # Test various GraphDef versions
    for version in strict + lenient:
      with ops.Graph().as_default() as g:
        test_util.set_producer_version(g, version)
        with self.session(graph=g) as sess:
          feed = {}
          xs = placeholders(args, feed)
          x = op(*xs)
          if version in strict:
            with self.assertRaisesOpError(error):
              sess.run(x, feed_dict=feed)
          else:
            r = sess.run(x, feed_dict=feed)
            if correct is not None:
              self.assertAllEqual(r, correct)

  def testConcat(self):
    for data in (2, [3], 7), ([2], 3, 7), ([2], [3], 7):
      self.check(array_ops.concat, (data, 0),
                 r'Ranks of all input tensors should match', [2, 3, 7])

  def testFill(self):
    self.check(
        array_ops.fill, (2, 3),
        'dims must be a vector', [3, 3],
        lenient=[5, 6],
        strict=[])
    self.check(
        array_ops.fill, ([2], [3]),
        'value must be a scalar', [3, 3],
        lenient=[5, 6],
        strict=[])

  def testPad(self):
    self.check(array_ops.pad, (7, [[1, 2]]),
               'The first dimension of paddings must be the rank of inputs',
               [0, 7, 0, 0])

  def testRandom(self):
    self.check(random_ops.random_uniform, (3,), 'shape must be a vector')

  def testReshape(self):
    self.check(
        array_ops.reshape, (7, 1),
        'sizes input must be 1-D', [7],
        lenient=[5, 6],
        strict=[])

  def testShardedFilename(self):
    self.check(gen_io_ops.sharded_filename, ('foo', 4, [100]),
               'must be a scalar', b'foo-00004-of-00100')

  def testShardedFilespec(self):
    self.check(gen_io_ops.sharded_filespec, ('foo', [100]), 'must be a scalar',
               b'foo-?????-of-00100')

  def testUnsortedSegmentSum(self):
    self.check(math_ops.unsorted_segment_sum, (7, 1, [4]),
               'num_segments should be a scalar', [0, 7, 0, 0])

  def testRange(self):
    self.check(
        math_ops.range, ([0], 3, 2),
        'start must be a scalar', [0, 2],
        lenient=[5, 6],
        strict=[])
    self.check(
        math_ops.range, (0, [3], 2),
        'limit must be a scalar', [0, 2],
        lenient=[5, 6],
        strict=[])
    self.check(
        math_ops.range, (0, 3, [2]),
        'delta must be a scalar', [0, 2],
        lenient=[5, 6],
        strict=[])

  def testSlice(self):
    data = np.arange(10)
    error = 'Expected begin and size arguments to be 1-D tensors'
    self.check(array_ops.slice, (data, 2, 3), error, [2, 3, 4])
    self.check(array_ops.slice, (data, [2], 3), error, [2, 3, 4])
    self.check(array_ops.slice, (data, 2, [3]), error, [2, 3, 4])

  def testSparseToDense(self):
    self.check(sparse_ops.sparse_to_dense, (1, 4, 7),
               'output_shape must be rank 1', [0, 7, 0, 0])

  def testTile(self):
    self.check(array_ops.tile, ([7], 2), 'Expected multiples to be 1-D', [7, 7])


if __name__ == '__main__':
  test.main()
