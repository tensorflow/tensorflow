""" Tests for module rewriter."""

import tensorflow as tf
import tensorflow.contrib.immediate as immediate
from tensorflow.contrib.immediate.python.immediate import module_rewriter
from tensorflow.contrib.immediate.python.immediate import test_util


class ModuleRewriterTest(test_util.TensorFlowTestCase):

  def testOpDefLibRewriter(self):
    """Try running functions to catch Python symbol linking errors."""

    with self.test_env(tf) as env:
      symbol_rewriter = module_rewriter.ImmediateRewriter(env)
      rewriter = immediate.ModuleRewriter(symbol_rewriter, "immediate.")
      new_tf = rewriter(tf)
      env.__dict__['tf'] = new_tf

      val1 = env.numpy_to_tensor(2)
      val2 = env.numpy_to_tensor(3)
      val3 = env.numpy_to_tensor([1, 2, 3, 4])
      new_tf.concat(0, [val3, val3])

      new_tf.add(val1, val2)
      new_tf.reduce_sum(val3)
      new_tf.ones((3, 3))
      new_tf.constant([1, 2, 3])
      new_tf.random_uniform([2, 2], 0, 10)
      val4 = new_tf.reshape(val3, [2, 2])
      new_tf.nn.relu(val1)
      new_tf.matmul(val4, val4)
      new_tf.equal(val1, val2)

      new_tf.image.random_brightness(val4, 1.)
      tensor1 = env.numpy_to_tensor([0, 0, 0, 0])
      tensor2 = env.numpy_to_tensor([1, 1, 1, 1])
      bool_tensor = env.numpy_to_tensor([True, False, True, False])
      new_tf.select(bool_tensor, tensor1, tensor2)
      new_tf.mod(5, 2)
      new_tf.cast(5, tf.float32)
      new_tf.transpose(new_tf.ones((2, 2)))

      new_tf.string_to_number("123")
      new_tf.nn.top_k(val3)
      new_tf.range(10)
      new_tf.nn.softmax([[1., 2.], [3., 4.]])
      new_tf.div(5, 2)
      new_tf.squeeze([[1, 2, 3]])
      new_tf.split(0, 2, tensor2)



if __name__ == "__main__":
  tf.test.main()
