import tensorflow as tf
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

class SparseApplyAdadeltaTest(test.TestCase):
  @test_util.run_in_graph_and_eager_modes
  def testInvalidDimensions(self):
    var = tf.Variable(tf.random.uniform([10, 10], dtype=tf.float32))
    accum = tf.Variable(tf.zeros([10, 10], dtype=tf.float32))
    accum_update = tf.Variable(tf.zeros([10, 10], dtype=tf.float32))

    lr = tf.constant(0.1, dtype=tf.float32)
    rho = tf.constant(0.95, dtype=tf.float32)
    epsilon = tf.constant(1e-7, dtype=tf.float32)

    grad = tf.constant([[0.1], [0.2], [0.3]], dtype=tf.float32)  # shape [3, 1]
    indices = tf.constant([0, 2, 4], dtype=tf.int32)

    with self.assertRaisesRegex(tf.errors.InvalidArgumentError, "var and grad must have the same number of dimensions"):
      tf.raw_ops.ResourceSparseApplyAdadelta(
          var=var.handle,
          accum=accum.handle,
          accum_update=accum_update.handle,
          lr=lr,
          rho=rho,
          epsilon=epsilon,
          grad=grad,
          indices=indices,
          use_locking=False,
      )

if __name__ == "__main__":
  test.main()
