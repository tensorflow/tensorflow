from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as otf   # original TensorFlow namespace

from tensorflow.contrib.immediate.python.immediate import test_util
import tensorflow.contrib.immediate as immediate
import tensorflow.models.image.mnist.convolutional as convolutional

def fake_data(num_images):
  """Generate a fake dataset that matches the dimensions of MNIST."""
  data = np.ndarray(
      shape=(num_images, 28, 28, 1),
      dtype=np.float32)
  labels = np.zeros(shape=(num_images,), dtype=np.int64)
  for image in xrange(num_images):
    label = image % 2
    data[image, :, :, 0] = label - 0.5
    labels[image] = label
  return data, labels

class MnistInferenceTest(otf.test.TestCase):

  def testMnistInference(self):
    prefix = 'tensorflow/contrib/immediate/python/immediate/testdata'

    # create immediate environment
    env = immediate.Env(otf)
    tf = env.tf

    # Load images
    test_data_filename = prefix+"/t10k-images-idx3-ubyte.gz"
    test_labels_filename = prefix+"/t10k-labels-idx1-ubyte.gz"
    meta_checkpoint_filename = prefix+"/convolutional-0.meta"
    checkpoint_filename = prefix+"/convolutional-0"

    # work-around for Jenkins setup issue
    # https://github.com/tensorflow/tensorflow/issues/2855
    if not (os.path.exists(test_data_filename) and
            os.path.exists(test_labels_filename) and
            os.path.exists(meta_checkpoint_filename) and
            os.path.exists(checkpoint_filename)):
      print("Couldn't find data dependency, aborting.")
      return True

    test_data = convolutional.extract_data(test_data_filename, 10000)
    test_labels = convolutional.extract_labels(test_labels_filename, 10000)


    # Load 99.25% accuracy checkpoint
    # TODO(yaroslavvb): add Variable/ITensor integration to obviate this
    new_saver = otf.train.import_meta_graph(meta_checkpoint_filename)
    sess = otf.Session()
    new_saver.restore(sess, checkpoint_filename)
    conv1_weights = env.numpy_to_itensor(sess.run(otf.trainable_variables()[0]))
    conv1_biases = env.numpy_to_itensor(sess.run(otf.trainable_variables()[1]))
    conv2_weights = env.numpy_to_itensor(sess.run(otf.trainable_variables()[2]))
    conv2_biases = env.numpy_to_itensor(sess.run(otf.trainable_variables()[3]))
    fc1_weights = env.numpy_to_itensor(sess.run(otf.trainable_variables()[4]))
    fc1_biases = env.numpy_to_itensor(sess.run(otf.trainable_variables()[5]))
    fc2_weights = env.numpy_to_itensor(sess.run(otf.trainable_variables()[6]))
    fc2_biases = env.numpy_to_itensor(sess.run(otf.trainable_variables()[7]))

    # run model in immedate environment to compute accuracy

    def predict(data):
      conv = tf.nn.conv2d(data, conv1_weights, strides=[1, 1, 1, 1],
                          padding='SAME')
      relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
      pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                            padding='SAME')
      conv = tf.nn.conv2d(pool, conv2_weights, strides=[1, 1, 1, 1],
                          padding='SAME')
      relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
      pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                            padding='SAME')
      pool_shape = pool.get_shape().as_list()
      reshape = tf.reshape(pool, [pool_shape[0], pool_shape[1] *
                                  pool_shape[2] * pool_shape[3]])
      hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
      logits = tf.matmul(hidden, fc2_weights) + fc2_biases
      eval_prediction = tf.nn.softmax(logits)
      return eval_prediction

    start_pos = 0
    BATCH_SIZE = 64
    total_errors = 0
    while start_pos < len(test_labels):
      end_pos = start_pos + BATCH_SIZE
      prediction = predict(test_data[start_pos:end_pos])
      true_labels = test_labels[start_pos:end_pos]
      errors = tf.argmax(prediction, 1) != true_labels
      num_errors = tf.reduce_sum(tf.cast(errors, tf.int32))
      total_errors += num_errors
      start_pos += BATCH_SIZE

    self.assertTrue(total_errors == 75)

if __name__ == "__main__":
  otf.test.main()

