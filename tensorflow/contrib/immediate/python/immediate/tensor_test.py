# Tests for immediate.Tensor

import tensorflow as tf
import numpy as np
import tensorflow.contrib.immediate as immediate


class TensorTest(tf.test.TestCase):

  def testInit(self):
    tensor = immediate.Tensor(None, None)
    self.assertTrue(True)

  def testNumpyInit(self):
    env = immediate.Env()
    a = np.array([[1,2],[3,4]], dtype=np.float32)
    tensor1 = immediate.Tensor.numpy_to_tensor(env, a)
    tensor2 = immediate.Tensor.numpy_to_tensor(env, a)
    print tensor1
    print tensor2
    
    array1 = tensor1.as_numpy()
    array2 = tensor2.as_numpy()
    self.assertAllEqual(array1, array2)


if __name__ == "__main__":
  tf.test.main()
