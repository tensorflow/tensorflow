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
"""Metrics model generator for mobilenet v1.

The produced model is to be used as part of the mini-benchmark, combined into
the same flatbuffer with the main model.

Mobilenet v1 is described in
https://tfhub.dev/tensorflow/coral-model/mobilenet_v1_1.0_224_quantized/1/default/1

The metrics used are symmetric KL-divergence and MSE.
"""
import argparse
import sys
# TODO(b/152872335): (re-)port to tf v2 after output names are kept during
# conversion in v2.
import tensorflow.compat.v1 as tf
from tensorflow.lite.experimental.acceleration.mini_benchmark.metrics import kl_divergence
from tensorflow.lite.python import lite
from tensorflow.lite.tools import flatbuffer_utils
parser = argparse.ArgumentParser(
    description='Script to generate a metrics model for mobilenet v1.')
parser.add_argument('output', help='Output filepath')


def main(output_path):
  tf.reset_default_graph()
  with tf.Graph().as_default():
    expected_scores = tf.placeholder(dtype=tf.float32, shape=[1, 1001])
    actual_scores = tf.placeholder(dtype=tf.float32, shape=[1, 1001])
    mse = tf.reshape(
        tf.math.reduce_mean((expected_scores - actual_scores)**2), [1],
        name='mse')
    kld_metric = kl_divergence.symmetric_kl_divergence(expected_scores,
                                                       actual_scores)
    kld_metric = tf.reshape(kld_metric, [1], name='symmetric_kl_divergence')
    # Thresholds chosen by comparing NNAPI top-k accuracy on MLTS on devices
    # with top-k accuracy within 1%-p of tflite CPU and with a 5-%p drop.
    ok = tf.reshape(
        tf.logical_and(kld_metric < 5.5, mse < 0.003), [1], name='ok')
    sess = tf.compat.v1.Session()
    converter = lite.TFLiteConverter.from_session(sess, [
        expected_scores,
        actual_scores,
    ], [kld_metric, mse, ok])
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
