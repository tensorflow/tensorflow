from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import attention_mask_ops


class AttentionMaskTest(tf.test.TestCase):

  def _testMask(self, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      sequence_len = np.asarray([2, 3], dtype=np.int64)
      energies = np.random.randn(2, 10).astype(np.float32)

      masked_energies = attention_mask_ops.attention_mask(sequence_len, energies).eval()

      for b in range(energies.shape[0]):
        sequence_len_b = sequence_len[b]
        self.assertAllEqual(masked_energies[b, :sequence_len_b], energies[b, :sequence_len_b])
        self.assertAllEqual(masked_energies[b, sequence_len_b:], np.full([energies.shape[1] - sequence_len_b], -np.finfo(np.float32).max))

  def _testMaskMedian(self, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      sequence_len = np.asarray([10, 8], dtype=np.int64)
      energies = np.random.randn(2, 10).astype(np.float32)

      # array([[ 0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1],
      #        [ 0.2,  0.2,  0.2,  0.2,  0.2,  0. ,  0. ,  0. ,  0. ,  0. ]])
      prev = np.asarray([[0.1] * 10, [0.2] * 5 + [0.0] * 5], dtype=np.float32)

      window_l = 1
      window_r = 1
      masked_energies = attention_mask_ops.attention_mask_median(
          sequence_len, energies, prev, window_l=window_l, window_r=window_r)
      masked_energies = masked_energies.eval()

      medians = [4, 2]
      for b in range(energies.shape[0]):
        index_l = medians[b] - window_l
        index_r = medians[b] + window_r + 1
        self.assertAllEqual(masked_energies[b, index_l:index_r], energies[b, index_l:index_r])
        self.assertAllEqual(masked_energies[b, :index_l], np.full([index_l], -np.finfo(np.float32).max))
        self.assertAllEqual(masked_energies[b, index_r:], np.full([energies.shape[1] - index_r], -np.finfo(np.float32).max))

  def testMask(self):
    self._testMask(False)
    self._testMask(True)

  def testMaskMedian(self):
    self._testMaskMedian(False)
    self._testMaskMedian(True)


if __name__ == '__main__':
  tf.test.main()
