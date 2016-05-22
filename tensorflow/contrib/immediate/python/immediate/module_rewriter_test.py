# Tests for module rewriter
# pylint: disable=missing-docstring,invalid-name

import tensorflow as tf
import tensorflow.contrib.immediate as immediate
import tensorflow.contrib.immediate.python.immediate.module_rewriter as util


class ModuleRewriterTest(tf.test.TestCase):

  def testInit(self):
    rewriter = immediate.ModuleRewriter(None)
    self.assertTrue(rewriter)

  def testDirectReplacement(self):
    """Replace a single function in the package."""


    def symbol_rewriter(symbol):
      def funky_add(arg1, arg2):
        return 11*(arg1+arg2)

      if (util.get_symbol_name(symbol) == "add" and
          "gen_math_ops.py" in util.get_symbol_file(symbol)):
        return funky_add

    rewriter = immediate.ModuleRewriter(symbol_rewriter, "immediate.")
    from tensorflow.python.ops import gen_math_ops
    new_math_ops = rewriter(gen_math_ops)
    self.assertEqual(new_math_ops.add(1, 2), 11*3)

  def testSumSubstitute(self):

    def symbol_rewriter(symbol):
      def funky_sum(*_unused_args, **_unused_kwargs):
        return 43

      if (util.get_symbol_name(symbol) == "_sum" and
          "gen_math_ops.py" in util.get_symbol_file(symbol)):
        return funky_sum

    from tensorflow.python.ops import math_ops
    rewriter = immediate.ModuleRewriter(symbol_rewriter, "immediate.")
    new_math_ops = rewriter(math_ops)
    tensor = tf.constant([1, 2, 3])
    self.assertEqual(new_math_ops.reduce_sum(tensor), 43)


if __name__ == "__main__":
  tf.test.main()
