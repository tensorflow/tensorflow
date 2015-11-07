"""Tests for state updating ops that may have benign race conditions."""
import tensorflow.python.platform

import numpy as np
import tensorflow as tf


class AssignOpTest(tf.test.TestCase):

  # NOTE(mrry): We exclude thess tests from the TSAN TAP target, because they
  #   contain benign and deliberate data races when multiple threads update
  #   the same parameters without a lock.
  def testParallelUpdateWithoutLocking(self):
    with self.test_session() as sess:
      ones_t = tf.fill([1024, 1024], 1.0)
      p = tf.Variable(tf.zeros([1024, 1024]))
      adds = [tf.assign_add(p, ones_t, use_locking=False)
              for _ in range(20)]
      tf.initialize_all_variables().run()

      def run_add(add_op):
        sess.run(add_op)
      threads = [self.checkedThread(target=run_add, args=(add_op,))
                 for add_op in adds]
      for t in threads:
        t.start()
      for t in threads:
        t.join()

      vals = p.eval()
      ones = np.ones((1024, 1024)).astype(np.float32)
      self.assertTrue((vals >= ones).all())
      self.assertTrue((vals <= ones * 20).all())

  def testParallelAssignWithoutLocking(self):
    with self.test_session() as sess:
      ones_t = tf.fill([1024, 1024], float(1))
      p = tf.Variable(tf.zeros([1024, 1024]))
      assigns = [tf.assign(p, tf.mul(ones_t, float(i)), False)
                 for i in range(1, 21)]
      tf.initialize_all_variables().run()

      def run_assign(assign_op):
        sess.run(assign_op)
      threads = [self.checkedThread(target=run_assign, args=(assign_op,))
                 for assign_op in assigns]
      for t in threads:
        t.start()
      for t in threads:
        t.join()

      vals = p.eval()

      # Assert every element is taken from one of the assignments.
      self.assertTrue((vals > 0).all())
      self.assertTrue((vals <= 20).all())


if __name__ == "__main__":
  tf.test.main()
