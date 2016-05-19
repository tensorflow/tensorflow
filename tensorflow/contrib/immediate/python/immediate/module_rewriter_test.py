# Tests for module rewriter

import tensorflow as tf
import numpy as np
import tensorflow.contrib.immediate as immediate

import types

class ModuleRewriterTest(tf.test.TestCase):

  def testInit(self):
    rewriter = immediate.ModuleRewriter()
    self.assertTrue(True)

  def testSubstitute(self):

    class WrapperMaker(object):
      def __call__(self, symbol):
        return Wrapper(symbol)

    class Wrapper(object):
      def __init__(self, symbol):
        self.symbol = symbol

      def __call__(self, *args, **kwargs):
        if self.symbol.__name__ == "add":
          return args[0] + args[1]

    rewriter = immediate.ModuleRewriter()
    rewriter.substitute("gen_math_ops.py$", "add", [types.FunctionType],
                        WrapperMaker())
    from tensorflow.python.ops import gen_math_ops
    new_math_ops = rewriter.apply(gen_math_ops)
    self.assertEqual(new_math_ops.add(1, 2), 3)



if __name__ == "__main__":
  tf.test.main()
