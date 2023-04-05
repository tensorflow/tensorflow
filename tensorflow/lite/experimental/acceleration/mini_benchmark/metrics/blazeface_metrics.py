# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Metrics model generator for Blazeface.

The produced model is to be used as part of the mini-benchmark, combined into
the same flatbuffer with the main model.

The blazeface model is described in
https://tfhub.dev/tensorflow/tfjs-model/blazeface/1/default/1

The metrics are roughly equivalent to the training time loss function for SSD
(https://arxiv.org/abs/1512.02325): localization loss and classification loss.

The localization loss is MSE (L2-norm) of box encodings over high-probability
boxes. A box encoding contains the size and location difference between the
prediction and the prototype box (see section 2 in the linked paper).

The classification loss is symmetric KL-divergence over classification scores
squashed to 0..1.

This follows the general rationale of the mini-benchmark: use as much of the
model outputs as possible for metrics, so that less example data is needed.
"""
import argparse
import sys
# TODO(b/152872335): (re-)port to tf v2 after output names are kept during
# conversion in v2.
import tensorflow.compat.v1 as tf
from tensorflow.lite.experimental.acceleration.mini_benchmark.metrics import kl_divergence
from tensorflow.lite.tools import flatbuffer_utils

parser = argparse.ArgumentParser(
    description='Script to generate a metrics model for the Blazeface.')
parser.add_argument('output', help='Output filepath')


@tf.function
def metrics(expected_box_encodings, expected_scores, actual_box_encodings,
            actual_scores):
  """Calculate metrics from expected and actual blazeface outputs.

  Args:
    expected_box_encodings: box encodings from model
    expected_scores: classifications from model
    actual_box_encodings: golden box encodings
    actual_scores: golden classifications

  Returns:
    two-item list with classification error and localization error
  """
  squashed_expected_scores = tf.math.divide(1.0,
                                            1.0 + tf.math.exp(-expected_scores))
  squashed_actual_scores = tf.math.divide(1.0,
                                          1.0 + tf.math.exp(-actual_scores))
  kld_metric = kl_divergence.symmetric_kl_divergence(expected_scores,
                                                     actual_scores)
  # ML Kit uses 0.5 as the threshold. We use
  # 0.1 to use more possible boxes based on experimentation with the model.
  high_scoring_indices = tf.math.logical_or(
      tf.math.greater(squashed_expected_scores, 0.1),
      tf.math.greater(squashed_actual_scores, 0.1))

  high_scoring_actual_boxes = tf.where(
      condition=tf.broadcast_to(
          input=high_scoring_indices, shape=tf.shape(actual_box_encodings)),
      x=actual_box_encodings,
      y=expected_box_encodings)
  box_diff = high_scoring_actual_boxes - expected_box_encodings
  box_squared_diff = tf.math.pow(box_diff, 2)
  # MSE is calculated over the high-scoring boxes.
  box_mse = tf.divide(
      tf.math.reduce_sum(box_squared_diff),
      tf.math.maximum(
          tf.math.count_nonzero(high_scoring_indices, dtype=tf.float32), 1.0))
  # Thresholds were determined experimentally by running validation on a variety
  # of devices. Known good devices give KLD ~10-e7 and MSE ~10-e12. A buggy
  # NNAPI implementation gives KLD > 200 and MSE > 100.
  ok = tf.logical_and(kld_metric < 0.1, box_mse < 0.01)

  return [kld_metric, box_mse, ok]


def main(output_path):
  tf.reset_default_graph()
  with tf.Graph().as_default():
    expected_box_encodings = tf.placeholder(
        dtype=tf.float32, shape=[1, 564, 16])
    expected_scores = tf.placeholder(dtype=tf.float32, shape=[1, 564, 1])
    actual_box_encodings = tf.placeholder(dtype=tf.float32, shape=[1, 564, 16])
    actual_scores = tf.placeholder(dtype=tf.float32, shape=[1, 564, 1])

    [kld_metric, box_mse, ok] = metrics(expected_box_encodings, expected_scores,
                                        actual_box_encodings, actual_scores)
    ok = tf.reshape(ok, [1], name='ok')
    kld_metric = tf.reshape(kld_metric, [1], name='symmetric_kl_divergence')
    box_mse = tf.reshape(box_mse, [1], name='box_mse')
    sess = tf.compat.v1.Session()
    converter = tf.lite.TFLiteConverter.from_session(sess, [
        expected_box_encodings, expected_scores, actual_box_encodings,
        actual_scores
    ], [kld_metric, box_mse, ok])
    converter.experimental_new_converter = True
    tflite_model = converter.convert()
    if sys.byteorder == 'big':
      tflite_model = flatbuffer_utils.byte_swap_tflite_buffer(
          tflite_model, 'big', 'little'
      )
    open(output_path, 'wb').write(tflite_model)


if __name__ == '__main__':
  flags, unparsed = parser.parse_known_args()

  if unparsed:
    parser.print_usage()
    sys.stderr.write('\nGot the following unparsed args, %r please fix.\n' %
                     unparsed)
    exit(1)
  else:
    main(flags.output)
    exit(0)
