# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Export an RNN cell in SavedModel format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import numpy as np

import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS

flags.DEFINE_string("export_dir", None, "Directory to export SavedModel.")


def main(argv):
  del argv

  root = tf.train.Checkpoint()
  # Create a cell and attach to our trackable.
  root.rnn_cell = tf.keras.layers.LSTMCell(units=10, recurrent_initializer=None)

  # Wrap the rnn_cell.__call__ function and assign to next_state.
  root.next_state = tf.function(root.rnn_cell.__call__, autograph=False)

  # Wrap the rnn_cell.get_initial_function using a decorator and assign to an
  # attribute with the same name.
  @tf.function(input_signature=[tf.TensorSpec([None, None], tf.float32)])
  def get_initial_state(tensor):
    return root.rnn_cell.get_initial_state(tensor, None, None)

  root.get_initial_state = get_initial_state

  # Construct an initial_state, then call next_state explicitly to trigger a
  # trace for serialization (we need an explicit call, because next_state has
  # not been annotated with an input_signature).
  initial_state = root.get_initial_state(
      tf.constant(np.random.uniform(size=[3, 10]).astype(np.float32)))
  root.next_state(
      tf.constant(np.random.uniform(size=[3, 19]).astype(np.float32)),
      initial_state)

  tf.saved_model.save(root, FLAGS.export_dir)


if __name__ == "__main__":
  tf.enable_v2_behavior()
  app.run(main)
