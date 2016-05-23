# Tests for immediate.Env

import tensorflow as tf
import numpy as np
import tensorflow.contrib.immediate as immediate

import wrapping_util

class WrappingUtilTest(tf.test.TestCase):
  def testToposort(self):
    l = wrapping_util.python_op_module_list_sorted()
    self.assertTrue(l.index('clip_ops')>l.index('array_ops'))

  def testGetArgInputs(self):
    print('hi')
    d = wrapping_util.get_arg_inputs()
    print(d)
    print(len(d))

if __name__ == "__main__":
  tf.test.main()
