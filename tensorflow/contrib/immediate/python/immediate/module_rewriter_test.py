# Tests for module rewriter
# pylint: disable=missing-docstring,invalid-name

import tensorflow as tf
import tensorflow.contrib.immediate as immediate
import tensorflow.contrib.immediate.python.immediate.module_rewriter as module_rewriter


class ModuleRewriterTest(tf.test.TestCase):

  def atestInit(self):
    rewriter = immediate.ModuleRewriter(None)
    self.assertTrue(rewriter)

  def atestDirectReplacement(self):
    """Replace a single function in the package."""


    def symbol_rewriter(symbol):
      def funky_add(arg1, arg2):
        return 11*(arg1+arg2)

      if (module_rewriter.get_symbol_name(symbol) == "add" and
          "gen_math_ops.py" in module_rewriter.get_symbol_file(symbol)):
        return funky_add

    rewriter = immediate.ModuleRewriter(symbol_rewriter, "immediate.")
    from tensorflow.python.ops import gen_math_ops
    new_math_ops = rewriter(gen_math_ops)
    self.assertEqual(new_math_ops.add(1, 2), 11*3)

  def atestAddRewriter(self):
    """Test simple rewriter."""

    symbol_rewriter = module_rewriter.AddSymbolRewriter(42)
    rewriter = immediate.ModuleRewriter(symbol_rewriter, "immediate.")
    from tensorflow.python.ops import gen_math_ops
    new_math_ops = rewriter(gen_math_ops)
    self.assertEqual(new_math_ops.add(1, 2), 42)

  def atestGenopsRewriter(self):

    import pdb, traceback, sys

    try:

      env = immediate.Env(tf)
      symbol_rewriter = module_rewriter.GenopsRewriter(env)
      rewriter = immediate.ModuleRewriter(symbol_rewriter, "immediate.")
      new_tf = rewriter(tf)
      val1 = env.numpy_to_tensor(2)
      val2 = env.numpy_to_tensor(3)
      val3 = env.numpy_to_tensor([1,2,3])
      print new_tf.add(val1, val2)
      print new_tf.reduce_sum(val3)
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)


  def testOpDefLibRewriter(self):
    import pdb, traceback, sys


    try:
      env = immediate.Env(tf)
      #      symbol_rewriter = module_rewriter.OpDefLibRewriter(env)
      symbol_rewriter = module_rewriter.ImmediateRewriter(env)
      rewriter = immediate.ModuleRewriter(symbol_rewriter, "immediate.")
      new_tf = rewriter(tf)
      env.__dict__['tf'] = new_tf

      val1 = env.numpy_to_tensor(2)
      val2 = env.numpy_to_tensor(3)
      val3 = env.numpy_to_tensor([1,2,3, 4])
      print new_tf.concat(0, [val3, val3])

      print new_tf.add(val1, val2)
      print new_tf.reduce_sum(val3)
      print new_tf.ones((3, 3))
      print new_tf.constant([1,2,3])
      print new_tf.random_uniform([2,2], 0, 10)
      val4 = new_tf.reshape(val3, [2, 2])
      print val4
      print new_tf.nn.relu(val1)
      print new_tf.matmul(val4, val4)
      print new_tf.equal(val1, val2)
      
      print new_tf.image.random_brightness(val4, 1.)
      tensor1 = env.numpy_to_tensor([0, 0, 0, 0])
      tensor2 = env.numpy_to_tensor([1, 1, 1, 1])
      bool_tensor = env.numpy_to_tensor([True, False, True, False])
      print new_tf.select(bool_tensor, tensor1, tensor2)
      print new_tf.mod(5, 2)
      print new_tf.cast(5, tf.float32)
      print new_tf.transpose(new_tf.ones((2,2)))

      # TODO(yaroslavvb) remove numpy string restriction

      print new_tf.string_to_number("123")
      print new_tf.nn.top_k(val3)
      print new_tf.range(10)
      print new_tf.nn.softmax([[1.,2.],[3., 4.]])
      print new_tf.div(5,2)
      print new_tf.squeeze([[1,2,3]])
      print new_tf.split(0, 2, tensor2)

      # decode_csv missing
      # try Print
      
    except:
      exc_type, value, tb = sys.exc_info()
      traceback.print_exc()
      pdb.post_mortem(tb)


  def atestSumSubstitute(self):

    def symbol_rewriter(symbol):
      def funky_sum(*_unused_args, **_unused_kwargs):
        return 43

      if (module_rewriter.get_symbol_name(symbol) == "_sum" and
          "gen_math_ops.py" in module_rewriter.get_symbol_file(symbol)):
        return funky_sum

    from tensorflow.python.ops import math_ops
    rewriter = immediate.ModuleRewriter(symbol_rewriter, "immediate.")
    new_math_ops = rewriter(math_ops)
    tensor = tf.constant([1, 2, 3])
    self.assertEqual(new_math_ops.reduce_sum(tensor), 43)


if __name__ == "__main__":
  tf.test.main()
