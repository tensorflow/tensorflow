"""Tests for tensorflow.ops.io_ops."""
import tensorflow.python.platform

import tensorflow as tf
from tensorflow.python.ops import gen_io_ops


class ShardedFileOpsTest(tf.test.TestCase):

  def testShardedFileName(self):
    with tf.Session(
        target="",
        config=tf.ConfigProto(device_count={"CPU": 2})):
      self.assertEqual(gen_io_ops._sharded_filename("foo", 4, 100).eval(),
                       "foo-00004-of-00100")
      self.assertEqual(gen_io_ops._sharded_filespec("foo", 100).eval(),
                       "foo-?????-of-00100")


if __name__ == "__main__":
  tf.test.main()
