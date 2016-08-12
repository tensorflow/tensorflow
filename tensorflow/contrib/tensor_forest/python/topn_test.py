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
"""Tests for topn.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.tensor_forest.python import topn
from tensorflow.contrib.tensor_forest.python.ops import topn_ops

from tensorflow.python.client import session
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class TopNOpsTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self.ops = topn_ops.Load()

  def testInsertOpIntoEmptyShortlist(self):
    with self.test_session():
      shortlist_ids, new_ids, new_scores = self.ops.top_n_insert(
          [0, -1, -1, -1, -1, -1],  # sl_ids
          [-999, -999, -999, -999, -999, -999],  # sl_scores
          [5],
          [33.0]  # new id and score
      )
      self.assertAllEqual([1, 0], shortlist_ids.eval())
      self.assertAllEqual([5, 1], new_ids.eval())
      self.assertAllEqual([33.0, -999], new_scores.eval())

  def testInsertOpIntoAlmostFullShortlist(self):
    with self.test_session():
      shortlist_ids, new_ids, new_scores = self.ops.top_n_insert(
          [4, 13, -1, 27, 99, 15],  # sl_ids
          [60.0, 87.0, -999, 65.0, 1000.0, 256.0],  # sl_scores
          [5],
          [93.0]  # new id and score
      )
      self.assertAllEqual([2, 0], shortlist_ids.eval())
      self.assertAllEqual([5, 5], new_ids.eval())
      # Shortlist still contains all known scores > 60.0
      self.assertAllEqual([93.0, 60.0], new_scores.eval())

  def testInsertOpIntoFullShortlist(self):
    with self.test_session():
      shortlist_ids, new_ids, new_scores = self.ops.top_n_insert(
          [5, 13, 44, 27, 99, 15],  # sl_ids
          [60.0, 87.0, 111.0, 65.0, 1000.0, 256.0],  # sl_scores
          [5],
          [93.0]  # new id and score
      )
      self.assertAllEqual([3, 0], shortlist_ids.eval())
      self.assertAllEqual([5, 5], new_ids.eval())
      # We removed a 65.0 from the list, so now we can only claim that
      # it holds all scores > 65.0.
      self.assertAllEqual([93.0, 65.0], new_scores.eval())

  def testInsertOpHard(self):
    with self.test_session():
      shortlist_ids, new_ids, new_scores = self.ops.top_n_insert(
          [4, 13, -1, 27, 99, 15],  # sl_ids
          [60.0, 87.0, -999, 65.0, 1000.0, 256.0],  # sl_scores
          [5, 6, 7, 8, 9],
          [61.0, 66.0, 90.0, 100.0, 2000.0]  # new id and score
      )
      # Top 5 scores are: 2000.0, 1000.0, 256.0, 100.0, 90.0
      self.assertAllEqual([2, 3, 1, 0], shortlist_ids.eval())
      self.assertAllEqual([9, 8, 7, 5], new_ids.eval())
      # 87.0 is the highest score we overwrote or didn't insert.
      self.assertAllEqual([2000.0, 100.0, 90.0, 87.0], new_scores.eval())

  def testRemoveSimple(self):
    with self.test_session():
      shortlist_ids, new_length = self.ops.top_n_remove(
          [5, 100, 200, 300, 400, 500], [200, 400, 600])
      self.assertAllEqual([2, 4], shortlist_ids.eval())
      self.assertAllEqual([3], new_length.eval())

  def testRemoveAllMissing(self):
    with self.test_session():
      shortlist_ids, new_length = self.ops.top_n_remove(
          [5, 100, 200, 300, 400, 500], [1200, 1400, 600])
      self.assertAllEqual([], shortlist_ids.eval())
      self.assertAllEqual([5], new_length.eval())

  def testRemoveAll(self):
    with self.test_session():
      shortlist_ids, new_length = self.ops.top_n_remove(
          [5, 100, 200, 300, 400, 500],
          [100, 200, 300, 400, 500],)
      self.assertAllEqual([1, 2, 3, 4, 5], shortlist_ids.eval())
      self.assertAllEqual([0], new_length.eval())


class TopNTest(test_util.TensorFlowTestCase):

  def testSimple(self):
    t = topn.TopN(1000, shortlist_size=10)
    t.insert([1, 2, 3, 4, 5], [1.0, 2.0, 3.0, 4.0, 5.0])
    t.remove([4, 5])
    ids, vals = t.get_best(2)
    with session.Session() as sess:
      sess.run(tf.initialize_all_variables())
      ids_v, vals_v = sess.run([ids, vals])
      self.assertItemsEqual([2, 3], list(ids_v))
      self.assertItemsEqual([2.0, 3.0], list(vals_v))

  def testSimpler(self):
    t = topn.TopN(1000, shortlist_size=10)
    t.insert([1], [33.0])
    ids, vals = t.get_best(1)
    with session.Session() as sess:
      sess.run(tf.initialize_all_variables())
      ids_v, vals_v = sess.run([ids, vals])
      self.assertListEqual([1], list(ids_v))
      self.assertListEqual([33.0], list(vals_v))

  def testLotsOfInsertsAscending(self):
    t = topn.TopN(1000, shortlist_size=10)
    for i in range(100):
      t.insert([i], [float(i)])
    ids, vals = t.get_best(5)
    with session.Session() as sess:
      sess.run(tf.initialize_all_variables())
      ids_v, vals_v = sess.run([ids, vals])
      self.assertItemsEqual([95, 96, 97, 98, 99], list(ids_v))
      self.assertItemsEqual([95.0, 96.0, 97.0, 98.0, 99.0], list(vals_v))

  def testLotsOfInsertsDescending(self):
    t = topn.TopN(1000, shortlist_size=10)
    for i in range(99, 1, -1):
      t.insert([i], [float(i)])
    ids, vals = t.get_best(5)
    with session.Session() as sess:
      sess.run(tf.initialize_all_variables())
      ids_v, vals_v = sess.run([ids, vals])
      self.assertItemsEqual([95, 96, 97, 98, 99], list(ids_v))
      self.assertItemsEqual([95.0, 96.0, 97.0, 98.0, 99.0], list(vals_v))

  def testRemoveNotInShortlist(self):
    t = topn.TopN(1000, shortlist_size=10)
    for i in range(20):
      t.insert([i], [float(i)])
    t.remove([4, 5])
    ids, vals = t.get_best(2)
    with session.Session() as sess:
      sess.run(tf.initialize_all_variables())
      ids_v, vals_v = sess.run([ids, vals])
      self.assertItemsEqual([18.0, 19.0], list(vals_v))
      self.assertItemsEqual([18, 19], list(ids_v))

  def testNeedToRefreshShortlistInGetBest(self):
    t = topn.TopN(1000, shortlist_size=10)
    for i in range(20):
      t.insert([i], [float(i)])
    # Shortlist now has 10 .. 19
    t.remove([11, 12, 13, 14, 15, 16, 17, 18, 19])
    ids, vals = t.get_best(2)
    with session.Session() as sess:
      sess.run(tf.initialize_all_variables())
      ids_v, vals_v = sess.run([ids, vals])
      self.assertItemsEqual([9, 10], list(ids_v))
      self.assertItemsEqual([9.0, 10.0], list(vals_v))


if __name__ == '__main__':
  googletest.main()
