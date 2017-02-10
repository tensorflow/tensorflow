"""Test for fact op"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import tensorflow as tf
from tensorflow.core.user_ops import fact_op

class FactTest(tf.test.TestCase):
  def testFact(self):
    with self.test_session():
      self.assertAllEqual(fact_op.fact().eval(), "0! == 1")

if __name__ == "__main__":
  tf.test.main()
