import numpy as np
import tensorflow as tf


class ShuffleTest(tf.test.TestCase):
    def testShuffle(self):
        shuffle_module = tf.load_op_library('shuffle.so')
        shuffle = shuffle_module.shuffle

        input_tensor = np.arange(12).reshape((3, 4))
        desired_shape = np.array([6, -1])
        output_tensor = input_tensor.reshape((6, 2))
        with self.test_session():
            result = shuffle(input_tensor, desired_shape)
            self.assertAllEqual(result.eval(), output_tensor)

        input_tensor = np.arange(12).reshape((3, 4))
        desired_shape = np.array([5, -1])
        output_tensor = input_tensor.reshape((6, 2))[:-1]
        with self.test_session():
            result = shuffle(input_tensor, desired_shape)
            self.assertAllEqual(result.eval(), output_tensor)


if __name__ == "__main__":
    tf.test.main()
