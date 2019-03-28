
from tensorflow.python.platform import test
from tensorflow.python.ops.image_ops_impl import median_filter_2D
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops


import numpy as np


class Median_2d_test(test.TestCase):


  def _validateMedian_2d(self, inputs, expected_values, filter_shape = (3, 3)):
    np_expected_values = np.array(expected_values)
    values_op = median_filter_2D(inputs)
    with self.test_session(use_gpu=False) as sess:
      values = values_op.eval()
      self.assertShapeEqual(values, inputs)
      self.assertShapeEqual(np_expected_values, values_op)
      self.assertAllClose(np_expected_values, values)


  def testfiltertuple(self):
    tf_img = np.zeros((3, 4, 3))
    tf_img = ops.convert_to_tensor(tf_img)

    with self.assertRaisesRegexp(TypeError, 'Filter shape must be a tuple'):
      median_filter_2D(tf_img, 3)
      median_filter_2D(tf_img, 3.5)
      median_filter_2D(tf_img, 'dt')
      median_filter_2D(tf_img, None)

    filter_shape = (3, 3, 3)
    with self.assertRaisesRegexp(ValueError, 'Filter shape must be a tuple of 2 integers .'
                                             'Got %s values in tuple' % len(filter_shape)):
      median_filter_2D(tf_img, filter_shape)

    with self.assertRaisesRegexp(TypeError, 'Size of the filter must be Integers'):
      median_filter_2D(tf_img, (3.5, 3))
      median_filter_2D(tf_img, (None, 3))


  def testfiltervalue(self):
    tf_img = np.zeros((6, 4, 3))
    tf_img = ops.convert_to_tensor(tf_img)
    with self.assertRaises(ValueError):
      median_filter_2D(tf_img, (4, 3))


  def testDimension(self) :
    tf_img = array_ops.placeholder(dtypes.int32,shape=[3, 4, None])
    tf_img1 = array_ops.placeholder(dtypes.int32,shape=[3, None, 4])
    tf_img2 = array_ops.placeholder(dtypes.int32,shape=[None, 3, 4])

    with self.assertRaises (TypeError) :
      median_filter_2D(tf_img)
      median_filter_2D(tf_img1)
      median_filter_2D(tf_img2)



  def test_imagevsfilter(self):
    tf_img = np.zeros((3, 4, 3))
    tf_img = ops.convert_to_tensor(tf_img)
    m = tf_img.shape[0].value
    no = tf_img.shape[1].value
    ch = tf_img.shape[2].value
    filter_shape = (3,5)
    with self.assertRaises( ValueError) :
      median_filter_2D(tf_img,filter_shape)


  def testcase(self):
    tf_img = [[[0.32801723, 0.08863795, 0.79119259],
              [0.35526001, 0.79388736, 0.55435993],
              [0.11607035, 0.55673079, 0.99473371]],

             [[0.53240645, 0.74684819, 0.33700031],
              [0.01760473, 0.28181609, 0.9751476 ],
              [0.01605137, 0.8292904,  0.56405609]],

             [[0.57215374, 0.10155051, 0.64836128],
              [0.36533048, 0.91401874, 0.02524159],
              [0.56379134, 0.9028874,  0.19505117]]]

    tf_img = np.array(tf_img)
    tf_img = ops.convert_to_tensor(tf_img)
    expt = [[[  0,   0,   0],
            [  4,  71, 141],
            [  0,   0,  0]],

           [[ 83,  25,  85],
            [ 90, 190, 143],
            [  4, 141,  49]],

           [[  0,   0,   0],
            [  4,  71,  49],
            [  0,   0,   0]]]
    expt = np.array(expt)
    self._validateMedian_2d(tf_img,expt)

if __name__ == "__main__":
  test.main()
