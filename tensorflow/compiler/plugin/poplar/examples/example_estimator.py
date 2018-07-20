# Based on the Tensorflow Estimator example:
#   https://www.tensorflow.org/get_started/custom_estimators

#  Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves.urllib.request import urlopen

import numpy as np
import tensorflow as tf

def my_model(features, labels, mode):

  vscope = tf.get_variable_scope()
  vscope.set_use_resource(True)

  with tf.device("/device:IPU:0"):

    x = tf.reshape(features["x"], [-1, 4])
    x = tf.layers.dense(inputs=x, units=10, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=3, activation=None)


    if mode == tf.estimator.ModeKeys.PREDICT:
      sm = tf.nn.softmax(x)
      pred = {
        'probabilities': sm
      }
    else:
      pred = None

    if (mode == tf.estimator.ModeKeys.TRAIN or
        mode == tf.estimator.ModeKeys.EVAL):
      loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=x, labels=labels))
    else:
      loss = None

    if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = tf.train.GradientDescentOptimizer(0.01)
      train = optimizer.minimize(loss, tf.train.get_global_step())
    else:
      train = None

  with tf.device("cpu"):
    if loss is not None:
      tf.contrib.ipu.ops.ipu_compile_summary('ipu_comp', loss)
      tf.summary.scalar('loss', loss)

  return tf.estimator.EstimatorSpec(mode=mode, predictions=pred, loss=loss,
                                    train_op=train)


def main():
  IRIS_TRAINING = "/tmp/iris_training.csv"
  IRIS_TEST = "/tmp/iris_test.csv"

  # Get the data
  if not os.path.exists(IRIS_TRAINING):
    raw = urlopen("http://download.tensorflow.org/data/iris_training.csv").read()
    with open(IRIS_TRAINING, "wb") as f:
      f.write(raw)

  if not os.path.exists(IRIS_TEST):
    raw = urlopen("http://download.tensorflow.org/data/iris_test.csv").read()
    with open(IRIS_TEST, "wb") as f:
      f.write(raw)

  # Create numpy objects for the 2 data sets
  training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TRAINING,
      target_dtype=np.int,
      features_dtype=np.float32)
  test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TEST,
      target_dtype=np.int,
      features_dtype=np.float32)


  # Create a run configuration using the IPU model device
  opts = tf.contrib.ipu.utils.create_ipu_config(profiling=True, type='IPU_MODEL')
  sess_cfg = tf.ConfigProto(ipu_options=opts)
  run_cfg = tf.estimator.RunConfig(session_config=sess_cfg)

  # Create a tf.Estimator for running the model
  classifier = tf.estimator.Estimator(model_fn=my_model,
                                      config=run_cfg,
                                      model_dir="/tmp/iris_model")

  # TRAINING

  # Define the training inputs
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(training_set.data)},
      y=np.array(training_set.target),
      num_epochs=None,
      shuffle=True,
      batch_size=4)

  # Train model.
  classifier.train(input_fn=train_input_fn, steps=1024)


  # EVALUATION

  # Define the test inputs
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(test_set.data)},
      y=np.array(test_set.target),
      num_epochs=1,
      shuffle=False,
      batch_size=4)

  # Evaluate accuracy.
  accuracy_score = classifier.evaluate(input_fn=test_input_fn)
  print (accuracy_score)


  # PREDICTION

  # Classify two new flower samples.
  new_samples = np.array(
      [[6.4, 3.2, 4.5, 1.5],
       [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
  predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": new_samples},
      num_epochs=1,
      shuffle=False)

  predictions = list(classifier.predict(input_fn=predict_input_fn))
  print(predictions)

if __name__ == "__main__":
    main()


