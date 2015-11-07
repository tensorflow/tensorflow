"""Tests for summary image op."""
import tensorflow.python.platform

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import image_ops


class SummaryImageOpTest(tf.test.TestCase):

  def _AsSummary(self, s):
    summ = tf.Summary()
    summ.ParseFromString(s)
    return summ

  def testImageSummary(self):
    np.random.seed(7)
    with self.test_session() as sess:
      for depth in 1, 3, 4:
        shape = (4, 5, 7) + (depth,)
        bad_color = [255, 0, 0, 255][:depth]
        for positive in False, True:
          # Build a mostly random image with one nan
          const = np.random.randn(*shape)
          const[0, 1, 2] = 0  # Make the nan entry not the max
          if positive:
            const = 1 + np.maximum(const, 0)
            scale = 255 / const.reshape(4, -1).max(axis=1)
            offset = 0
          else:
            scale = 127 / np.abs(const.reshape(4, -1)).max(axis=1)
            offset = 128
          adjusted = np.floor(scale[:, None, None, None] * const + offset)
          const[0, 1, 2, depth / 2] = np.nan

          # Summarize
          summ = tf.image_summary("img", const)
          value = sess.run(summ)
          self.assertEqual([], summ.get_shape())
          image_summ = self._AsSummary(value)

          # Decode the first image and check consistency
          image = image_ops.decode_png(
              image_summ.value[0].image.encoded_image_string).eval()
          self.assertAllEqual(image[1, 2], bad_color)
          image[1, 2] = adjusted[0, 1, 2]
          self.assertAllClose(image, adjusted[0])

          # Check the rest of the proto
          # Only the first 3 images are returned.
          for v in image_summ.value:
            v.image.ClearField("encoded_image_string")
          expected = '\n'.join("""
              value {
                tag: "img/image/%d"
                image { height: %d width: %d colorspace: %d }
              }""" % ((i,) + shape[1:]) for i in xrange(3))
          self.assertProtoEquals(expected, image_summ)


if __name__ == "__main__":
  tf.test.main()
