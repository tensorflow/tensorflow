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
"""Demo of the tfdbg curses UI: A TF network computing Fibonacci sequence."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python import debug as tf_debug

FLAGS = None


def main(_):
  sess = tf.Session()

  # Construct the TensorFlow network.
  n0 = tf.Variable(
      np.ones([FLAGS.tensor_size] * 2), dtype=tf.int32, name="node_00")
  n1 = tf.Variable(
      np.ones([FLAGS.tensor_size] * 2), dtype=tf.int32, name="node_01")

  for i in xrange(2, FLAGS.length):
    n0, n1 = n1, tf.add(n0, n1, name="node_%.2d" % i)

  sess.run(tf.global_variables_initializer())

  # Wrap the TensorFlow Session object for debugging.
  if FLAGS.debug and FLAGS.tensorboard_debug_address:
    raise ValueError(
        "The --debug and --tensorboard_debug_address flags are mutually "
        "exclusive.")
  if FLAGS.debug:
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    def has_negative(_, tensor):
      return np.any(tensor < 0)

    sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    sess.add_tensor_filter("has_negative", has_negative)
  elif FLAGS.tensorboard_debug_address:
    sess = tf_debug.TensorBoardDebugWrapperSession(
        sess, FLAGS.tensorboard_debug_address)

  print("Fibonacci number at position %d:\n%s" %
        (FLAGS.length, sess.run(n1)))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--tensor_size",
      type=int,
      default=1,
      help="""\
      Size of tensor. E.g., if the value is 30, the tensors will have shape
      [30, 30].\
      """)
  parser.add_argument(
      "--length",
      type=int,
      default=20,
      help="Length of the fibonacci sequence to compute.")
  parser.add_argument(
      "--ui_type",
      type=str,
      default="curses",
      help="Command-line user interface type (curses | readline)")
  parser.add_argument(
      "--debug",
      dest="debug",
      action="store_true",
      help="Use TensorFlow Debugger (tfdbg). Mutually exclusive with the "
      "--tensorboard_debug_address flag.")
  parser.add_argument(
      "--tensorboard_debug_address",
      type=str,
      default=None,
      help="Connect to the TensorBoard Debugger Plugin backend specified by "
      "the gRPC address (e.g., localhost:1234). Mutually exclusive with the "
      "--debug flag.")

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
