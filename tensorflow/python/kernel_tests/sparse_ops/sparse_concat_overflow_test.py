import tensorflow as tf
from tensorflow.python.framework import errors
from tensorflow.python.platform import test


class SparseConcatOverflowTest(test.TestCase):

    def test_sparse_concat_shape_overflow(self):
        indices1 = tf.constant([[0, 0, 0]], dtype=tf.int64)
        values1 = tf.constant([1], dtype=tf.int32)
        shape1 = tf.constant([5, 2, 2147483647], dtype=tf.int64)

        indices2 = tf.constant([[0, 1, 0]], dtype=tf.int64)
        values2 = tf.constant([2], dtype=tf.int32)
        shape2 = tf.constant([5, 1879048192, 536870912], dtype=tf.int64)

        with self.assertRaises(errors.InvalidArgumentError):
            tf.raw_ops.SparseConcat(
                indices=[indices1, indices2],
                values=[values1, values2],
                shapes=[shape1, shape2],
                concat_dim=1,
            )


if __name__ == "__main__":
    test.main()

