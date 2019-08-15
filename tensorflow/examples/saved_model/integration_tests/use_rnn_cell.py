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
"""Load and use an RNN cell stored as a SavedModel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import numpy as np
import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", None, "Directory to load SavedModel from.")


def main(argv):
  del argv
  cell = tf.saved_model.load(FLAGS.model_dir)

  initial_state = cell.get_initial_state(
      tf.constant(np.random.uniform(size=[3, 10]).astype(np.float32)))

  cell.next_state(
      tf.constant(np.random.uniform(size=[3, 19]).astype(np.float32)),
      initial_state)


if __name__ == "__main__":
  app.run(main)
