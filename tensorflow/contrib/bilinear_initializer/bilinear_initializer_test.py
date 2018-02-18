import numpy as np
import skimage.transform as sktr
import tensorflow as tf
from tensorflow.contrib.bilinear_initializer.bilinear_initializer_op import bilinear_initializer

class BilinearInitializerTest(tf.test.TestCase):
    def testBilinearInitializer(self):
        with self.test_session() as sess:
            # Define parameters
            height, width, channel = 3, 3, 3
            factor = 2
            kernel_width = 2 * factor - factor % 2
            num_output = 3

            # Generate input image
            img = self.generate_imput_img(height, width, channel)

            # Generate benchmark image (ski)
            img_benchmark = self.ski_upsample(factor, img)

            # Generate test image (tensorflow)
            with tf.Session() as sess:
                dims = [kernel_width, kernel_width, channel, num_output]
                tf_filter = bilinear_initializer(dims).eval()
                new_h = factor * height
                new_w = factor * width
                img_test = self.tf_upsample(sess, tf_filter, new_h, new_w, channel, factor, img)
                self.assertTrue(np.allclose(img_benchmark, img_test))

    def generate_imput_img(self, height, width, channel):
        """
        Generate input image with a given height, width, and number of channels
        """
        x, y = np.ogrid[:height, :width]
        return np.repeat((x + y)[..., np.newaxis], channel, 2) / float(height * width)

    def ski_upsample(self, factor, input_img):
        """
        Benchmark for testing. Use skikit learn library
        order = 1 means bilinear initializer
        """
        return sktr.rescale(input_img, factor, 
            mode='constant', cval=0, order=1)

    def tf_upsample(self, sess, tf_filter, new_h, new_w, channel, factor, input_img):
        expanded_img = np.expand_dims(input_img, axis=0)
        filt = tf.placeholder(tf.float32)
        img_in = tf.placeholder(tf.float32)
        res = tf.nn.conv2d_transpose(img_in, filt,
            output_shape=[1, new_h, new_w, channel],
            strides=[1, factor, factor, 1])
        final_result = sess.run(res,
                        feed_dict={filt: tf_filter,
                                   img_in: expanded_img})
        return final_result.squeeze()


if __name__ == "__main__":
    tf.test.main()
