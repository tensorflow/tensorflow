# Tests for immediate.Env

import tensorflow as tf
import numpy as np
import tensorflow.contrib.immediate as immediate

import wrapping_util

class WrappingUtilTest(tf.test.TestCase):
  def testToposort(self):
    l = wrapping_util.python_op_module_list_sorted()
    self.assertTrue(l.index('clip_ops')>l.index('array_ops'))
  

if __name__ == "__main__":
  tf.test.main()
