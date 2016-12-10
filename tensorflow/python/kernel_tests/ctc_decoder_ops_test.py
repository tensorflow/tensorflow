# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for tensorflow.ctc_ops.ctc_loss_op."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
from six.moves import zip_longest

import tensorflow as tf


def grouper(iterable, n, fillvalue=None):
  """Collect data into fixed-length chunks or blocks."""
  # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
  args = [iter(iterable)] * n
  return zip_longest(fillvalue=fillvalue, *args)


def flatten(list_of_lists):
  """Flatten one level of nesting."""
  return itertools.chain.from_iterable(list_of_lists)


class CTCGreedyDecoderTest(tf.test.TestCase):

  def _testCTCDecoder(self, decoder, inputs, seq_lens, log_prob_truth,
                      decode_truth, expected_err_re=None, **decoder_args):
    inputs_t = [tf.convert_to_tensor(x) for x in inputs]
    # convert inputs_t into a [max_time x batch_size x depth] tensor
    # from a len time python list of [batch_size x depth] tensors
    inputs_t = tf.stack(inputs_t)

    with self.test_session(use_gpu=False) as sess:
      decoded_list, log_probability = decoder(
          inputs_t,
          sequence_length=seq_lens, **decoder_args)
      decoded_unwrapped = list(flatten([
          (st.indices, st.values, st.dense_shape) for st in decoded_list]))

      if expected_err_re is None:
        outputs = sess.run(
            decoded_unwrapped + [log_probability])

        # Group outputs into (ix, vals, shape) tuples
        output_sparse_tensors = list(grouper(outputs[:-1], 3))

        output_log_probability = outputs[-1]

        # Check the number of decoded outputs (top_paths) match
        self.assertEqual(len(output_sparse_tensors), len(decode_truth))

        # For each SparseTensor tuple, compare (ix, vals, shape)
        for out_st, truth_st, tf_st in zip(
            output_sparse_tensors, decode_truth, decoded_list):
          self.assertAllEqual(out_st[0], truth_st[0])  # ix
          self.assertAllEqual(out_st[1], truth_st[1])  # vals
          self.assertAllEqual(out_st[2], truth_st[2])  # shape
          # Compare the shapes of the components with the truth. The
          # `None` elements are not known statically.
          self.assertEqual([None, truth_st[0].shape[1]],
                           tf_st.indices.get_shape().as_list())
          self.assertEqual([None], tf_st.values.get_shape().as_list())
          self.assertShapeEqual(truth_st[2], tf_st.dense_shape)

        # Make sure decoded probabilities match
        self.assertAllClose(output_log_probability, log_prob_truth, atol=1e-6)
      else:
        with self.assertRaisesOpError(expected_err_re):
          sess.run(decoded_unwrapped + [log_probability])

  def testCTCGreedyDecoder(self):
    """Test two batch entries - best path decoder."""
    max_time_steps = 6
    # depth == 4

    seq_len_0 = 4
    input_prob_matrix_0 = np.asarray(
        [[1.0, 0.0, 0.0, 0.0],  # t=0
         [0.0, 0.0, 0.4, 0.6],  # t=1
         [0.0, 0.0, 0.4, 0.6],  # t=2
         [0.0, 0.9, 0.1, 0.0],  # t=3
         [0.0, 0.0, 0.0, 0.0],  # t=4 (ignored)
         [0.0, 0.0, 0.0, 0.0]],  # t=5 (ignored)
        dtype=np.float32)
    input_log_prob_matrix_0 = np.log(input_prob_matrix_0)

    seq_len_1 = 5
    # dimensions are time x depth

    input_prob_matrix_1 = np.asarray(
        [[0.1, 0.9, 0.0, 0.0],  # t=0
         [0.0, 0.9, 0.1, 0.0],  # t=1
         [0.0, 0.0, 0.1, 0.9],  # t=2
         [0.0, 0.9, 0.1, 0.1],  # t=3
         [0.9, 0.1, 0.0, 0.0],  # t=4
         [0.0, 0.0, 0.0, 0.0]],  # t=5 (ignored)
        dtype=np.float32)
    input_log_prob_matrix_1 = np.log(input_prob_matrix_1)

    # len max_time_steps array of batch_size x depth matrices
    inputs = [np.vstack([input_log_prob_matrix_0[t, :],
                         input_log_prob_matrix_1[t, :]])
              for t in range(max_time_steps)]

    # batch_size length vector of sequence_lengths
    seq_lens = np.array([seq_len_0, seq_len_1], dtype=np.int32)

    # batch_size length vector of negative log probabilities
    log_prob_truth = np.array([
        np.sum(-np.log([1.0, 0.6, 0.6, 0.9])),
        np.sum(-np.log([0.9, 0.9, 0.9, 0.9, 0.9]))
    ], np.float32)[:, np.newaxis]

    # decode_truth: one SparseTensor (ix, vals, shape)
    decode_truth = [
        (np.array([[0, 0],  # batch 0, 2 outputs
                   [0, 1],
                   [1, 0],  # batch 1, 3 outputs
                   [1, 1],
                   [1, 2]], dtype=np.int64),
         np.array([0, 1,      # batch 0
                   1, 1, 0],  # batch 1
                  dtype=np.int64),
         # shape is batch x max_decoded_length
         np.array([2, 3], dtype=np.int64)),
    ]

    self._testCTCDecoder(
        tf.nn.ctc_greedy_decoder,
        inputs, seq_lens, log_prob_truth, decode_truth)

  def testCTCDecoderBeamSearch(self):
    """Test one batch, two beams - hibernating beam search."""
    # max_time_steps == 8
    depth = 6

    seq_len_0 = 5
    input_prob_matrix_0 = np.asarray(
        [[0.30999, 0.309938, 0.0679938, 0.0673362, 0.0708352, 0.173908],
         [0.215136, 0.439699, 0.0370931, 0.0393967, 0.0381581, 0.230517],
         [0.199959, 0.489485, 0.0233221, 0.0251417, 0.0233289, 0.238763],
         [0.279611, 0.452966, 0.0204795, 0.0209126, 0.0194803, 0.20655],
         [0.51286, 0.288951, 0.0243026, 0.0220788, 0.0219297, 0.129878],
         # Random entry added in at time=5
         [0.155251, 0.164444, 0.173517, 0.176138, 0.169979, 0.160671]],
        dtype=np.float32)
    # Add arbitrary offset - this is fine
    input_log_prob_matrix_0 = np.log(input_prob_matrix_0) + 2.0

    # len max_time_steps array of batch_size x depth matrices
    inputs = ([input_log_prob_matrix_0[t, :][np.newaxis, :]
               for t in range(seq_len_0)]  # Pad to max_time_steps = 8
              + 2 * [np.zeros((1, depth), dtype=np.float32)])

    # batch_size length vector of sequence_lengths
    seq_lens = np.array([seq_len_0], dtype=np.int32)

    # batch_size length vector of negative log probabilities
    log_prob_truth = np.array([
        0.584855,  # output beam 0
        0.389139  # output beam 1
    ], np.float32)[np.newaxis, :]

    # decode_truth: two SparseTensors, (ix, values, shape)
    decode_truth = [
        # beam 0, batch 0, two outputs decoded
        (np.array([[0, 0], [0, 1]], dtype=np.int64),
         np.array([1, 0], dtype=np.int64),
         np.array([1, 2], dtype=np.int64)),
        # beam 1, batch 0, three outputs decoded
        (np.array([[0, 0], [0, 1], [0, 2]], dtype=np.int64),
         np.array([0, 1, 0], dtype=np.int64),
         np.array([1, 3], dtype=np.int64)),
    ]

    self._testCTCDecoder(
        tf.nn.ctc_beam_search_decoder,
        inputs, seq_lens, log_prob_truth,
        decode_truth,
        beam_width=2,
        top_paths=2)


if __name__ == "__main__":
  tf.test.main()
