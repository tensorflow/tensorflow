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
"""Load and use RNN model stored as a SavedModel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
from absl import app
from absl import flags
import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", None, "Directory to load SavedModel from.")


def main(argv):
  del argv

  sentences = [
      "<S> sentence <E>", "<S> second sentence <E>", "<S> third sentence<E>"
  ]

  model = tf.saved_model.load(FLAGS.model_dir)
  model.train(tf.constant(sentences))
  decoded = model.decode_greedy(
      sequence_length=10, first_word=tf.constant("<S>"))
  _ = [d.numpy() for d in decoded]

  # This is testing that a model using a SavedModel can be re-exported again,
  # e.g. to catch issues such as b/142231881.
  tf.saved_model.save(model, tempfile.mkdtemp())

if __name__ == "__main__":
  app.run(main)
