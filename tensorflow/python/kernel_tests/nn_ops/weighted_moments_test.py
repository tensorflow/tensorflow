from tensorflow.python.framework import constant_op
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test


class WeightedMomentsTest(test.TestCase):

  def test_tensor_axes_keepdims_false_shape_mismatch(self):
    x = constant_op.constant([[1., 2., 3., 4., 5.],
                              [6., 7., 8., 9., 10.]])
    weights = constant_op.constant([[1.], [1.]])
    axes = constant_op.constant([0], dtype=constant_op.dtypes.int32)
    mean, var = nn_ops.weighted_moments(
        x, axes=axes, frequency_weights=weights, keepdims=False)
    self.assertEqual(mean.shape.as_list(), [5])
    self.assertEqual(var.shape.as_list(), [5])


if __name__ == "__main__":
  test.main()

