"""Tests for cifar10 input."""

import os

import tensorflow.python.platform

import tensorflow as tf

from tensorflow.models.image.cifar10 import cifar10_input


class CIFAR10InputTest(tf.test.TestCase):

  def _record(self, label, red, green, blue):
    image_size = 32 * 32
    record = "%s%s%s%s" % (chr(label), chr(red) * image_size,
                           chr(green) * image_size, chr(blue) * image_size)
    expected = [[[red, green, blue]] * 32] * 32
    return record, expected

  def testSimple(self):
    labels = [9, 3, 0]
    records = [self._record(labels[0], 0, 128, 255),
               self._record(labels[1], 255, 0, 1),
               self._record(labels[2], 254, 255, 0)]
    contents = "".join([record for record, _ in records])
    expected = [expected for _, expected in records]
    filename = os.path.join(self.get_temp_dir(), "cifar")
    open(filename, "w").write(contents)

    with self.test_session() as sess:
      q = tf.FIFOQueue(99, [tf.string], shapes=())
      q.enqueue([filename]).run()
      q.close().run()
      result = cifar10_input.read_cifar10(q)

      for i in range(3):
        key, label, uint8image = sess.run([
            result.key, result.label, result.uint8image])
        self.assertEqual("%s:%d" % (filename, i), key)
        self.assertEqual(labels[i], label)
        self.assertAllEqual(expected[i], uint8image)

      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run([result.key, result.uint8image])


if __name__ == "__main__":
  tf.test.main()
