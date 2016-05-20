# Tests for module rewriter

import tensorflow as tf
import numpy as np
import tensorflow.contrib.immediate as immediate

import inspect
import sys
import types


class ModuleRewriterTest(tf.test.TestCase):

  def atestInit(self):
    rewriter = immediate.ModuleRewriter(None)
    self.assertTrue(True)

  def atestDirectReplacement(self):
    """Replace a single function in the package."""

    def get_source(symbol):
      try:
        return inspect.getsourcefile(symbol)
      except:
        return ""

    def get_name(symbol):
      try:
        return symbol.__name__
      except:
        return ""

    def symbol_rewriter(symbol):
      def funky_add(a, b): return 11*(a+b)
      if (get_name(symbol) == "add" and
          "gen_math_ops.py" in get_source(symbol)):
        return funky_add


    rewriter = immediate.ModuleRewriter(symbol_rewriter, "immediate.")
    from tensorflow.python.ops import gen_math_ops
    new_math_ops = rewriter(gen_math_ops)
    self.assertEqual(new_math_ops.add(1, 2), 11*3)

  def testSumSubstitute(self):
    def get_source(symbol):
      try:
        return inspect.getsourcefile(symbol)
      except:
        return ""

    def get_name(symbol):
      try:
        return symbol.__name__
      except:
        return ""

    def symbol_rewriter(symbol):
      def funky_sum(*args, **kwargs):
        print 'doing funky sum'
        return 43

      if (get_name(symbol) == "_sum" and
          "gen_math_ops.py" in get_source(symbol)):
        return funky_sum


    rewriter = immediate.ModuleRewriter(symbol_rewriter, "immediate.")
    from tensorflow.python.ops import math_ops
    new_math_ops = rewriter(math_ops)
    tensor = tf.constant([1, 2, 3])
    self.assertEqual(new_math_ops.reduce_sum(tensor), 43)

  # def testSubstitute(self):

  #   class WrapperMaker(object):
  #     def __call__(self, symbol):
  #       return Wrapper(symbol)

  #   class Wrapper(object):
  #     def __init__(self, symbol):
  #       self.symbol = symbol

  #     def __call__(self, *args, **kwargs):
  #       if self.symbol.__name__ == "add":
  #         return args[0] + args[1]

  #   rewriter = immediate.ModuleRewriter()
  #   rewriter.substitute("gen_math_ops.py$", "add", [types.FunctionType],
  #                       WrapperMaker())
  #   from tensorflow.python.ops import gen_math_ops
  #   new_math_ops = rewriter.apply(gen_math_ops)
  #   self.assertEqual(new_math_ops.add(1, 2), 3)



if __name__ == "__main__":
  tf.test.main()
