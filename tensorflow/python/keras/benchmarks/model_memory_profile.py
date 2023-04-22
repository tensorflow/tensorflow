# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Memory profile on Keras model.

To add a new model for memory profile:
1. Create the model.
2. Decorate it with `@memory_profiler.profile`.
3. Add the model function to the dict `models`.
"""

from absl import app
from absl import flags

from absl import logging
import numpy as np

import tensorflow as tf

try:
  import memory_profiler  # pylint:disable=g-import-not-at-top
except ImportError:
  memory_profiler = None


FLAGS = flags.FLAGS
flags.DEFINE_string('model', None,
                    'The model to run memory profiler.')


@memory_profiler.profile
def _imdb_lstm_model():
  """LSTM model."""
  x_train = np.random.randint(0, 1999, size=(2500, 100))
  y_train = np.random.random((2500, 1))

  # IMDB LSTM model.
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Embedding(20000, 128))
  model.add(tf.keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2))
  model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

  model.compile('sgd', 'mse')
  # Warm up the model with one epoch.
  model.fit(x_train, y_train, batch_size=512, epochs=3)


def main(_):
  # Add the model for memory profile.
  models = {
      'lstm': _imdb_lstm_model,
  }

  if FLAGS.model in models:
    logging.info('Run memory profile on %s.', FLAGS.model)
    run_model = models[FLAGS.model]
    run_model()
  else:
    logging.info('The model does not exist. Please verify the model name.')


if __name__ == '__main__':
  flags.mark_flags_as_required(['model'])
  if memory_profiler:
    app.run(main)

