# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""An example tf.keras model that is trained using MirroredStrategy."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sys import argv
import numpy as np
import tensorflow as tf


def input_fn():
  x = np.random.random((1024, 10))
  y = np.random.randint(2, size=(1024, 1))
  x = tf.cast(x, tf.float32)
  dataset = tf.data.Dataset.from_tensor_slices((x, y))
  dataset = dataset.repeat(10)
  dataset = dataset.batch(32)
  return dataset


def main(args):
  if len(args) < 2:
    print('You must specify  model_dir for checkpoints such as'
          ' /tmp/tfkeras_example./')
    return

  print('Using %s to store checkpoints.' % args[1])

  strategy = tf.contrib.distribute.MirroredStrategy(
      ['/device:GPU:0', '/device:GPU:1'])
  config = tf.estimator.RunConfig(train_distribute=strategy)
  optimizer = tf.train.GradientDescentOptimizer(0.2)

  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(16, activation='relu', input_shape=(10,)))
  model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

  model.compile(loss='binary_crossentropy', optimizer=optimizer)
  model.summary()
  tf.keras.backend.set_learning_phase(True)
  keras_estimator = tf.keras.estimator.model_to_estimator(
      keras_model=model, config=config, model_dir=args[1])

  keras_estimator.train(input_fn=input_fn, steps=10)
  eval_result = keras_estimator.evaluate(input_fn=input_fn)
  print('Eval result: {}'.format(eval_result))

if __name__ == '__main__':
  tf.app.run(argv=argv)
