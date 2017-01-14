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

"""Tests for barrier ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import data_flow_ops


class BarrierTest(tf.test.TestCase):

  def testConstructorWithShapes(self):
    with tf.Graph().as_default():
      b = data_flow_ops.Barrier(
          (tf.float32, tf.float32),
          shapes=((1, 2, 3), (8,)),
          shared_name="B",
          name="B")
    self.assertTrue(isinstance(b.barrier_ref, tf.Tensor))
    self.assertEquals(tf.string_ref, b.barrier_ref.dtype)
    self.assertProtoEquals("""
      name:'B' op:'Barrier'
      attr {
        key: "capacity"
        value {
          i: -1
        }
      }
      attr { key: 'component_types'
             value { list { type: DT_FLOAT type: DT_FLOAT } } }
      attr {
        key: 'shapes'
        value {
          list {
            shape {
              dim { size: 1 } dim { size: 2 } dim { size: 3 }
            }
            shape {
              dim { size: 8 }
            }
          }
        }
      }
      attr { key: 'container' value { s: "" } }
      attr { key: 'shared_name' value: { s: 'B' } }
      """, b.barrier_ref.op.node_def)

  def testInsertMany(self):
    with self.test_session():
      b = data_flow_ops.Barrier(
          (tf.float32, tf.float32),
          shapes=((), ()),
          name="B")
      size_t = b.ready_size()
      self.assertEqual([], size_t.get_shape())
      keys = [b"a", b"b", b"c"]
      insert_0_op = b.insert_many(0, keys, [10.0, 20.0, 30.0])
      insert_1_op = b.insert_many(1, keys, [100.0, 200.0, 300.0])

      self.assertEquals(size_t.eval(), [0])
      insert_0_op.run()
      self.assertEquals(size_t.eval(), [0])
      insert_1_op.run()
      self.assertEquals(size_t.eval(), [3])

  def testInsertManyEmptyTensor(self):
    with self.test_session():
      error_message = ("Empty tensors are not supported, but received shape "
                       r"\'\(0,\)\' at index 1")
      with self.assertRaisesRegexp(ValueError, error_message):
        data_flow_ops.Barrier((tf.float32, tf.float32),
                              shapes=((1,), (0,)),
                              name="B")

  def testInsertManyEmptyTensorUnknown(self):
    with self.test_session():
      b = data_flow_ops.Barrier(
          (tf.float32, tf.float32),
          name="B")
      size_t = b.ready_size()
      self.assertEqual([], size_t.get_shape())
      keys = [b"a", b"b", b"c"]
      insert_0_op = b.insert_many(0, keys, np.array([[], [], []], np.float32))
      self.assertEquals(size_t.eval(), [0])
      with self.assertRaisesOpError(
          ".*Tensors with no elements are not supported.*"):
        insert_0_op.run()

  def testTakeMany(self):
    with self.test_session() as sess:
      b = data_flow_ops.Barrier(
          (tf.float32, tf.float32),
          shapes=((), ()),
          name="B")
      size_t = b.ready_size()
      keys = [b"a", b"b", b"c"]
      values_0 = [10.0, 20.0, 30.0]
      values_1 = [100.0, 200.0, 300.0]
      insert_0_op = b.insert_many(0, keys, values_0)
      insert_1_op = b.insert_many(1, keys, values_1)
      take_t = b.take_many(3)

      insert_0_op.run()
      insert_1_op.run()
      self.assertEquals(size_t.eval(), [3])

      indices_val, keys_val, values_0_val, values_1_val = sess.run([
          take_t[0], take_t[1], take_t[2][0], take_t[2][1]])

    self.assertAllEqual(indices_val, [-2**63] * 3)
    for k, v0, v1 in zip(keys, values_0, values_1):
      idx = keys_val.tolist().index(k)
      self.assertEqual(values_0_val[idx], v0)
      self.assertEqual(values_1_val[idx], v1)

  def testTakeManySmallBatch(self):
    with self.test_session() as sess:
      b = data_flow_ops.Barrier(
          (tf.float32, tf.float32),
          shapes=((), ()),
          name="B")
      size_t = b.ready_size()
      size_i = b.incomplete_size()
      keys = [b"a", b"b", b"c", b"d"]
      values_0 = [10.0, 20.0, 30.0, 40.0]
      values_1 = [100.0, 200.0, 300.0, 400.0]
      insert_0_op = b.insert_many(0, keys, values_0)
      # Split adding of the second component into two independent operations.
      # After insert_1_1_op, we'll have two ready elements in the barrier,
      # 2 will still be incomplete.
      insert_1_1_op = b.insert_many(1, keys[0:2], values_1[0:2])  # add "a", "b"
      insert_1_2_op = b.insert_many(1, keys[2:3], values_1[2:3])  # add "c"
      insert_1_3_op = b.insert_many(1, keys[3:], values_1[3:])  # add "d"
      insert_empty_op = b.insert_many(0, [], [])
      close_op = b.close()
      close_op_final = b.close(cancel_pending_enqueues=True)
      index_t, key_t, value_list_t = b.take_many(3, allow_small_batch=True)
      insert_0_op.run()
      insert_1_1_op.run()
      close_op.run()
      # Now we have a closed barrier with 2 ready elements. Running take_t
      # should return a reduced batch with 2 elements only.
      self.assertEquals(size_i.eval(), [2])  # assert that incomplete size = 2
      self.assertEquals(size_t.eval(), [2])  # assert that ready size = 2
      _, keys_val, values_0_val, values_1_val = sess.run([
          index_t, key_t, value_list_t[0], value_list_t[1]
      ])
      # Check that correct values have been returned.
      for k, v0, v1 in zip(keys[0:2], values_0[0:2], values_1[0:2]):
        idx = keys_val.tolist().index(k)
        self.assertEqual(values_0_val[idx], v0)
        self.assertEqual(values_1_val[idx], v1)

      # The next insert completes the element with key "c". The next take_t
      # should return a batch with just 1 element.
      insert_1_2_op.run()
      self.assertEquals(size_i.eval(), [1])  # assert that incomplete size = 1
      self.assertEquals(size_t.eval(), [1])  # assert that ready size = 1
      _, keys_val, values_0_val, values_1_val = sess.run([
          index_t, key_t, value_list_t[0], value_list_t[1]
      ])
      # Check that correct values have been returned.
      for k, v0, v1 in zip(keys[2:3], values_0[2:3], values_1[2:3]):
        idx = keys_val.tolist().index(k)
        self.assertEqual(values_0_val[idx], v0)
        self.assertEqual(values_1_val[idx], v1)

      # Adding nothing ought to work, even if the barrier is closed.
      insert_empty_op.run()

      # currently keys "a" and "b" are not in the barrier, adding them
      # again after it has been closed, ought to cause failure.
      with self.assertRaisesOpError("is closed"):
        insert_1_1_op.run()
      close_op_final.run()

      # These ops should fail because the barrier has now been closed with
      # cancel_pending_enqueues = True.
      with self.assertRaisesOpError("is closed"):
        insert_empty_op.run()
      with self.assertRaisesOpError("is closed"):
        insert_1_3_op.run()

  def testUseBarrierWithShape(self):
    with self.test_session() as sess:
      b = data_flow_ops.Barrier(
          (tf.float32, tf.float32),
          shapes=((2, 2), (8,)),
          name="B")
      size_t = b.ready_size()
      keys = [b"a", b"b", b"c"]
      values_0 = np.array(
          [[[10.0] * 2] * 2, [[20.0] * 2] * 2, [[30.0] * 2] * 2], np.float32)
      values_1 = np.array([[100.0] * 8, [200.0] * 8, [300.0] * 8],
                          np.float32)
      insert_0_op = b.insert_many(0, keys, values_0)
      insert_1_op = b.insert_many(1, keys, values_1)
      take_t = b.take_many(3)

      insert_0_op.run()
      insert_1_op.run()
      self.assertEquals(size_t.eval(), [3])

      indices_val, keys_val, values_0_val, values_1_val = sess.run([
          take_t[0], take_t[1], take_t[2][0], take_t[2][1]])
      self.assertAllEqual(indices_val, [-2**63] * 3)
      self.assertShapeEqual(keys_val, take_t[1])
      self.assertShapeEqual(values_0_val, take_t[2][0])
      self.assertShapeEqual(values_1_val, take_t[2][1])

    for k, v0, v1 in zip(keys, values_0, values_1):
      idx = keys_val.tolist().index(k)
      self.assertAllEqual(values_0_val[idx], v0)
      self.assertAllEqual(values_1_val[idx], v1)

  def testParallelInsertMany(self):
    with self.test_session() as sess:
      b = data_flow_ops.Barrier(tf.float32, shapes=())
      size_t = b.ready_size()
      keys = [str(x).encode("ascii") for x in range(10)]
      values = [float(x) for x in range(10)]
      insert_ops = [b.insert_many(0, [k], [v]) for k, v in zip(keys, values)]
      take_t = b.take_many(10)

      sess.run(insert_ops)
      self.assertEquals(size_t.eval(), [10])

      indices_val, keys_val, values_val = sess.run(
          [take_t[0], take_t[1], take_t[2][0]])

    self.assertAllEqual(indices_val, [-2**63 + x for x in range(10)])
    for k, v in zip(keys, values):
      idx = keys_val.tolist().index(k)
      self.assertEqual(values_val[idx], v)

  def testParallelTakeMany(self):
    with self.test_session() as sess:
      b = data_flow_ops.Barrier(tf.float32, shapes=())
      size_t = b.ready_size()
      keys = [str(x).encode("ascii") for x in range(10)]
      values = [float(x) for x in range(10)]
      insert_op = b.insert_many(0, keys, values)
      take_t = [b.take_many(1) for _ in keys]

      insert_op.run()
      self.assertEquals(size_t.eval(), [10])

      index_fetches = []
      key_fetches = []
      value_fetches = []
      for ix_t, k_t, v_t in take_t:
        index_fetches.append(ix_t)
        key_fetches.append(k_t)
        value_fetches.append(v_t[0])
      vals = sess.run(index_fetches + key_fetches + value_fetches)

    index_vals = vals[:len(keys)]
    key_vals = vals[len(keys):2 * len(keys)]
    value_vals = vals[2 * len(keys):]

    taken_elems = []
    for k, v in zip(key_vals, value_vals):
      taken_elems.append((k[0], v[0]))

    self.assertAllEqual(np.hstack(index_vals), [-2**63] * 10)

    self.assertItemsEqual(
        zip(keys, values),
        [(k[0], v[0]) for k, v in zip(key_vals, value_vals)])

  def testBlockingTakeMany(self):
    with self.test_session() as sess:
      b = data_flow_ops.Barrier(tf.float32, shapes=())
      keys = [str(x).encode("ascii") for x in range(10)]
      values = [float(x) for x in range(10)]
      insert_ops = [b.insert_many(0, [k], [v]) for k, v in zip(keys, values)]
      take_t = b.take_many(10)

      def take():
        indices_val, keys_val, values_val = sess.run(
            [take_t[0], take_t[1], take_t[2][0]])
        self.assertAllEqual(
            indices_val, [int(x.decode("ascii")) - 2**63 for x in keys_val])
        self.assertItemsEqual(zip(keys, values), zip(keys_val, values_val))

      t = self.checkedThread(target=take)
      t.start()
      time.sleep(0.1)
      for insert_op in insert_ops:
        insert_op.run()
      t.join()

  def testParallelInsertManyTakeMany(self):
    with self.test_session() as sess:
      b = data_flow_ops.Barrier(
          (tf.float32, tf.int64), shapes=((), (2,)))
      num_iterations = 100
      keys = [str(x) for x in range(10)]
      values_0 = np.asarray(range(10), dtype=np.float32)
      values_1 = np.asarray([[x+1, x + 2] for x in range(10)], dtype=np.int64)
      keys_i = lambda i: [("%d:%s" % (i, k)).encode("ascii") for k in keys]
      insert_0_ops = [
          b.insert_many(0, keys_i(i), values_0 + i)
          for i in range(num_iterations)]
      insert_1_ops = [
          b.insert_many(1, keys_i(i), values_1 + i)
          for i in range(num_iterations)]
      take_ops = [b.take_many(10) for _ in range(num_iterations)]

      def take(sess, i, taken):
        indices_val, keys_val, values_0_val, values_1_val = sess.run(
            [take_ops[i][0], take_ops[i][1],
             take_ops[i][2][0], take_ops[i][2][1]])
        taken.append({"indices": indices_val,
                      "keys": keys_val,
                      "values_0": values_0_val,
                      "values_1": values_1_val})

      def insert(sess, i):
        sess.run([insert_0_ops[i], insert_1_ops[i]])

      taken = []

      take_threads = [
          self.checkedThread(target=take, args=(sess, i, taken))
          for i in range(num_iterations)]
      insert_threads = [
          self.checkedThread(target=insert, args=(sess, i))
          for i in range(num_iterations)]

      for t in take_threads:
        t.start()
      time.sleep(0.1)
      for t in insert_threads:
        t.start()
      for t in take_threads:
        t.join()
      for t in insert_threads:
        t.join()

      self.assertEquals(len(taken), num_iterations)
      flatten = lambda l: [item for sublist in l for item in sublist]
      all_indices = sorted(flatten([t_i["indices"] for t_i in taken]))
      all_keys = sorted(flatten([t_i["keys"] for t_i in taken]))

      expected_keys = sorted(flatten(
          [keys_i(i) for i in range(num_iterations)]))
      expected_indices = sorted(flatten(
          [-2**63 + j] * 10 for j in range(num_iterations)))

      self.assertAllEqual(all_indices, expected_indices)
      self.assertAllEqual(all_keys, expected_keys)

      for taken_i in taken:
        outer_indices_from_keys = np.array(
            [int(k.decode("ascii").split(":")[0]) for k in taken_i["keys"]])
        inner_indices_from_keys = np.array(
            [int(k.decode("ascii").split(":")[1]) for k in taken_i["keys"]])
        self.assertAllEqual(taken_i["values_0"],
                            outer_indices_from_keys + inner_indices_from_keys)
        expected_values_1 = np.vstack(
            (1 + outer_indices_from_keys + inner_indices_from_keys,
             2 + outer_indices_from_keys + inner_indices_from_keys)).T
        self.assertAllEqual(taken_i["values_1"], expected_values_1)

  def testClose(self):
    with self.test_session() as sess:
      b = data_flow_ops.Barrier(
          (tf.float32, tf.float32),
          shapes=((), ()),
          name="B")
      size_t = b.ready_size()
      incomplete_t = b.incomplete_size()
      keys = [b"a", b"b", b"c"]
      values_0 = [10.0, 20.0, 30.0]
      values_1 = [100.0, 200.0, 300.0]
      insert_0_op = b.insert_many(0, keys, values_0)
      insert_1_op = b.insert_many(1, keys, values_1)
      close_op = b.close()
      fail_insert_op = b.insert_many(0, ["f"], [60.0])
      take_t = b.take_many(3)
      take_too_many_t = b.take_many(4)

      self.assertEquals(size_t.eval(), [0])
      self.assertEquals(incomplete_t.eval(), [0])
      insert_0_op.run()
      self.assertEquals(size_t.eval(), [0])
      self.assertEquals(incomplete_t.eval(), [3])
      close_op.run()

      # This op should fail because the barrier is closed.
      with self.assertRaisesOpError("is closed"):
        fail_insert_op.run()

      # This op should succeed because the barrier has not cancelled
      # pending enqueues
      insert_1_op.run()
      self.assertEquals(size_t.eval(), [3])
      self.assertEquals(incomplete_t.eval(), [0])

      # This op should fail because the barrier is closed.
      with self.assertRaisesOpError("is closed"):
        fail_insert_op.run()

      # This op should fail because we requested more elements than are
      # available in incomplete + ready queue.
      with self.assertRaisesOpError(
          r"is closed and has insufficient elements "
          r"\(requested 4, total size 3\)"):
        sess.run(take_too_many_t[0])  # Sufficient to request just the indices

      # This op should succeed because there are still completed elements
      # to process.
      indices_val, keys_val, values_0_val, values_1_val = sess.run(
          [take_t[0], take_t[1], take_t[2][0], take_t[2][1]])
      self.assertAllEqual(indices_val, [-2**63] * 3)
      for k, v0, v1 in zip(keys, values_0, values_1):
        idx = keys_val.tolist().index(k)
        self.assertEqual(values_0_val[idx], v0)
        self.assertEqual(values_1_val[idx], v1)

      # This op should fail because there are no more completed elements and
      # the queue is closed.
      with self.assertRaisesOpError("is closed and has insufficient elements"):
        sess.run(take_t[0])

  def testCancel(self):
    with self.test_session() as sess:
      b = data_flow_ops.Barrier(
          (tf.float32, tf.float32),
          shapes=((), ()),
          name="B")
      size_t = b.ready_size()
      incomplete_t = b.incomplete_size()
      keys = [b"a", b"b", b"c"]
      values_0 = [10.0, 20.0, 30.0]
      values_1 = [100.0, 200.0, 300.0]
      insert_0_op = b.insert_many(0, keys, values_0)
      insert_1_op = b.insert_many(1, keys[0:2], values_1[0:2])
      insert_2_op = b.insert_many(1, keys[2:], values_1[2:])
      cancel_op = b.close(cancel_pending_enqueues=True)
      fail_insert_op = b.insert_many(0, ["f"], [60.0])
      take_t = b.take_many(2)
      take_too_many_t = b.take_many(3)

      self.assertEquals(size_t.eval(), [0])
      insert_0_op.run()
      insert_1_op.run()
      self.assertEquals(size_t.eval(), [2])
      self.assertEquals(incomplete_t.eval(), [1])
      cancel_op.run()

      # This op should fail because the queue is closed.
      with self.assertRaisesOpError("is closed"):
        fail_insert_op.run()

      # This op should fail because the queue is cancelled.
      with self.assertRaisesOpError("is closed"):
        insert_2_op.run()

      # This op should fail because we requested more elements than are
      # available in incomplete + ready queue.
      with self.assertRaisesOpError(
          r"is closed and has insufficient elements "
          r"\(requested 3, total size 2\)"):
        sess.run(take_too_many_t[0])  # Sufficient to request just the indices

      # This op should succeed because there are still completed elements
      # to process.
      indices_val, keys_val, values_0_val, values_1_val = sess.run(
          [take_t[0], take_t[1], take_t[2][0], take_t[2][1]])
      self.assertAllEqual(indices_val, [-2**63] * 2)
      for k, v0, v1 in zip(keys[0:2], values_0[0:2], values_1[0:2]):
        idx = keys_val.tolist().index(k)
        self.assertEqual(values_0_val[idx], v0)
        self.assertEqual(values_1_val[idx], v1)

      # This op should fail because there are no more completed elements and
      # the queue is closed.
      with self.assertRaisesOpError("is closed and has insufficient elements"):
        sess.run(take_t[0])

  def _testParallelInsertManyTakeManyCloseHalfwayThrough(self, cancel):
    with self.test_session() as sess:
      b = data_flow_ops.Barrier(
          (tf.float32, tf.int64), shapes=((), (2,)))
      num_iterations = 50
      keys = [str(x) for x in range(10)]
      values_0 = np.asarray(range(10), dtype=np.float32)
      values_1 = np.asarray([[x + 1, x + 2] for x in range(10)], dtype=np.int64)
      keys_i = lambda i: [("%d:%s" % (i, k)).encode("ascii") for k in keys]
      insert_0_ops = [
          b.insert_many(0, keys_i(i), values_0 + i)
          for i in range(num_iterations)]
      insert_1_ops = [
          b.insert_many(1, keys_i(i), values_1 + i)
          for i in range(num_iterations)]
      take_ops = [b.take_many(10) for _ in range(num_iterations)]
      close_op = b.close(cancel_pending_enqueues=cancel)

      def take(sess, i, taken):
        try:
          indices_val, unused_keys_val, unused_val_0, unused_val_1 = sess.run(
              [take_ops[i][0], take_ops[i][1],
               take_ops[i][2][0], take_ops[i][2][1]])
          taken.append(len(indices_val))
        except tf.errors.OutOfRangeError:
          taken.append(0)

      def insert(sess, i):
        try:
          sess.run([insert_0_ops[i], insert_1_ops[i]])
        except tf.errors.AbortedError:
          pass

      taken = []

      take_threads = [
          self.checkedThread(target=take, args=(sess, i, taken))
          for i in range(num_iterations)]
      insert_threads = [
          self.checkedThread(target=insert, args=(sess, i))
          for i in range(num_iterations)]

      first_half_insert_threads = insert_threads[:num_iterations//2]
      second_half_insert_threads = insert_threads[num_iterations//2:]

      for t in take_threads:
        t.start()
      for t in first_half_insert_threads:
        t.start()
      for t in first_half_insert_threads:
        t.join()

      close_op.run()

      for t in second_half_insert_threads:
        t.start()
      for t in take_threads:
        t.join()
      for t in second_half_insert_threads:
        t.join()

      self.assertEqual(
          sorted(taken), [0] * (num_iterations//2) + [10] * (num_iterations//2))

  def testParallelInsertManyTakeManyCloseHalfwayThrough(self):
    self._testParallelInsertManyTakeManyCloseHalfwayThrough(cancel=False)

  def testParallelInsertManyTakeManyCancelHalfwayThrough(self):
    self._testParallelInsertManyTakeManyCloseHalfwayThrough(cancel=True)

  def _testParallelPartialInsertManyTakeManyCloseHalfwayThrough(self, cancel):
    with self.test_session() as sess:
      b = data_flow_ops.Barrier(
          (tf.float32, tf.int64), shapes=((), (2,)))
      num_iterations = 100
      keys = [str(x) for x in range(10)]
      values_0 = np.asarray(range(10), dtype=np.float32)
      values_1 = np.asarray([[x + 1, x + 2] for x in range(10)], dtype=np.int64)
      keys_i = lambda i: [("%d:%s" % (i, k)).encode("ascii") for k in keys]
      insert_0_ops = [
          b.insert_many(0, keys_i(i), values_0 + i, name="insert_0_%d" % i)
          for i in range(num_iterations)]

      close_op = b.close(cancel_pending_enqueues=cancel)

      take_ops = [b.take_many(10, name="take_%d" % i)
                  for i in range(num_iterations)]
      # insert_1_ops will only run after closure
      insert_1_ops = [
          b.insert_many(1, keys_i(i), values_1 + i, name="insert_1_%d" % i)
          for i in range(num_iterations)]

      def take(sess, i, taken):
        if cancel:
          try:
            indices_val, unused_keys_val, unused_val_0, unused_val_1 = sess.run(
                [take_ops[i][0], take_ops[i][1],
                 take_ops[i][2][0], take_ops[i][2][1]])
            taken.append(len(indices_val))
          except tf.errors.OutOfRangeError:
            taken.append(0)
        else:
          indices_val, unused_keys_val, unused_val_0, unused_val_1 = sess.run(
              [take_ops[i][0], take_ops[i][1],
               take_ops[i][2][0], take_ops[i][2][1]])
          taken.append(len(indices_val))

      def insert_0(sess, i):
        insert_0_ops[i].run(session=sess)

      def insert_1(sess, i):
        if cancel:
          try:
            insert_1_ops[i].run(session=sess)
          except tf.errors.AbortedError:
            pass
        else:
          insert_1_ops[i].run(session=sess)

      taken = []

      take_threads = [
          self.checkedThread(target=take, args=(sess, i, taken))
          for i in range(num_iterations)]
      insert_0_threads = [
          self.checkedThread(target=insert_0, args=(sess, i))
          for i in range(num_iterations)]
      insert_1_threads = [
          self.checkedThread(target=insert_1, args=(sess, i))
          for i in range(num_iterations)]

      for t in insert_0_threads:
        t.start()
      for t in insert_0_threads:
        t.join()
      for t in take_threads:
        t.start()

      close_op.run()

      for t in insert_1_threads:
        t.start()
      for t in take_threads:
        t.join()
      for t in insert_1_threads:
        t.join()

      if cancel:
        self.assertEqual(taken, [0] * num_iterations)
      else:
        self.assertEqual(taken, [10] * num_iterations)

  def testParallelPartialInsertManyTakeManyCloseHalfwayThrough(self):
    self._testParallelPartialInsertManyTakeManyCloseHalfwayThrough(cancel=False)

  def testParallelPartialInsertManyTakeManyCancelHalfwayThrough(self):
    self._testParallelPartialInsertManyTakeManyCloseHalfwayThrough(cancel=True)

  def testIncompatibleSharedBarrierErrors(self):
    with self.test_session():
      # Do component types and shapes.
      b_a_1 = data_flow_ops.Barrier((tf.float32,), shapes=(()),
                                    shared_name="b_a")
      b_a_2 = data_flow_ops.Barrier((tf.int32,), shapes=(()),
                                    shared_name="b_a")
      b_a_1.barrier_ref.eval()
      with self.assertRaisesOpError("component types"):
        b_a_2.barrier_ref.eval()

      b_b_1 = data_flow_ops.Barrier((tf.float32,), shapes=(()),
                                    shared_name="b_b")
      b_b_2 = data_flow_ops.Barrier(
          (tf.float32, tf.int32),
          shapes=((), ()),
          shared_name="b_b")
      b_b_1.barrier_ref.eval()
      with self.assertRaisesOpError("component types"):
        b_b_2.barrier_ref.eval()

      b_c_1 = data_flow_ops.Barrier(
          (tf.float32, tf.float32),
          shapes=((2, 2), (8,)),
          shared_name="b_c")
      b_c_2 = data_flow_ops.Barrier(
          (tf.float32, tf.float32), shared_name="b_c")
      b_c_1.barrier_ref.eval()
      with self.assertRaisesOpError("component shapes"):
        b_c_2.barrier_ref.eval()

      b_d_1 = data_flow_ops.Barrier(
          (tf.float32, tf.float32), shapes=((), ()),
          shared_name="b_d")
      b_d_2 = data_flow_ops.Barrier(
          (tf.float32, tf.float32),
          shapes=((2, 2), (8,)),
          shared_name="b_d")
      b_d_1.barrier_ref.eval()
      with self.assertRaisesOpError("component shapes"):
        b_d_2.barrier_ref.eval()

      b_e_1 = data_flow_ops.Barrier(
          (tf.float32, tf.float32),
          shapes=((2, 2), (8,)),
          shared_name="b_e")
      b_e_2 = data_flow_ops.Barrier(
          (tf.float32, tf.float32),
          shapes=((2, 5), (8,)),
          shared_name="b_e")
      b_e_1.barrier_ref.eval()
      with self.assertRaisesOpError("component shapes"):
        b_e_2.barrier_ref.eval()


if __name__ == "__main__":
  tf.test.main()
