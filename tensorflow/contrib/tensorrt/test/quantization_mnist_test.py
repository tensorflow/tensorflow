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

import numpy as np
import os

import tensorflow as tf
from tensorflow.contrib.tensorrt.python.trt_convert import create_inference_graph
from tensorflow.core.protobuf import config_pb2 
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python import estimator as tf_estimator
from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.estimator.model_fn import ModeKeys, EstimatorSpec

INPUT_NODE_NAME = 'input'
OUTPUT_NODE_NAME = 'output'

def build_graph(x):
  def quantize(x, r):
    x = tf.fake_quant_with_min_max_args(x, -r, r)
    return x

  def dense_layer(x, num_inputs, num_outputs, quantization_range, name='dense'):
    """Equivalent to tf.layers.dense but with a quantization range between
    the MatMul and BiasAdd."""
    with tf.variable_scope(name) as scope:
      kernel = tf.get_variable('kernel', shape=[num_inputs, num_outputs],
          dtype=tf.float32, initializer=tf.keras.initializers.glorot_uniform())
      bias = tf.get_variable('bias', shape=[num_outputs,],
          dtype=tf.float32, initializer=tf.keras.initializers.zeros())
      x = tf.matmul(x, kernel)
      x = quantize(x, quantization_range)
      x = tf.nn.bias_add(x, bias)
    return x

  x = quantize(x, 1)
  # Conv + Bias + Relu6
  x = tf.layers.conv2d(x, filters=32, kernel_size=3, use_bias=True)
  x = tf.nn.relu6(x)
  # Conv + Bias + Relu6
  x = tf.layers.conv2d(x, filters=64, kernel_size=3, use_bias=True)
  x = tf.nn.relu6(x)
  x = tf.reduce_mean(x, [1, 2])
  x = quantize(x, 6)
  # FC1
  x = dense_layer(x, 64, 512, 6, name='dense')
  x = quantize(x, 6)
  x = tf.nn.relu6(x)
  # FC2
  x = dense_layer(x, 512, 10, 25, name='dense_1')
  x = quantize(x, 25)
  x = tf.identity(x, name=OUTPUT_NODE_NAME)
  return x

def preprocess_fn(x, y):
  x = tf.cast(x, tf.float32)
  x = tf.expand_dims(x, axis=2)
  x = 2.0 * (x / 255.0) - 1.0
  y = tf.cast(y, tf.int32)
  return x, y

def run(is_training, use_trt, batch_size, num_epochs, model_dir):
  """Train or evaluate the model.

  Args:
    is_training: Whether to train or evaluate the model. In training mode,
      quantization will be simulated where the fake_quant_with_min_max_args
      are placed.
    use_trt: If true, use TRT INT8 mode for evaluation, which will perform real
      quantization. Otherwise use native TensorFlow which will perform
      simulated quantization. Ignored if is_training is True.
    batch_size: Batch size.
    num_epochs: How many epochs to train. Ignored if is_training is False.
    model_dir: Where to save or load checkpoint.
  """
  # Get dataset
  train, test = mnist.load_data()
  
  def eval_input_fn():
    mnist_x, mnist_y = test
    dataset = tf.data.Dataset.from_tensor_slices((mnist_x, mnist_y))
    dataset = dataset.apply(tf.data.experimental.map_and_batch(
        map_func=preprocess_fn,
        batch_size=batch_size,
        num_parallel_calls=8))
    dataset = dataset.repeat(count=1)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels

  def train_input_fn():
    mnist_x, mnist_y = train
    dataset = tf.data.Dataset.from_tensor_slices((mnist_x, mnist_y))
    dataset = dataset.shuffle(2*len(mnist_x))
    dataset = dataset.apply(tf.data.experimental.map_and_batch(
        map_func=preprocess_fn,
        batch_size=batch_size,
        num_parallel_calls=8))
    dataset = dataset.repeat(count=num_epochs)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels

  def model_fn(features, labels, mode):
    if is_training:
      logits_out = build_graph(features)
    else:
      graph_def = get_graph_def(use_trt, batch_size, model_dir)
      logits_out = tf.import_graph_def(graph_def,
          input_map={INPUT_NODE_NAME: features},
          return_elements=[OUTPUT_NODE_NAME+':0'],
          name='')[0]
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels,
        logits=logits_out)
    tf.summary.scalar('loss', loss)
    classes_out = tf.argmax(logits_out, axis=1, name='classes_out')
    accuracy = tf.metrics.accuracy(
        labels=labels,
        predictions=classes_out,
        name='acc_op')
    tf.summary.scalar('accuracy', accuracy[1])
    if mode == ModeKeys.EVAL:
      return EstimatorSpec(
          mode,
          loss=loss,
          eval_metric_ops={'accuracy': accuracy})
    elif mode == ModeKeys.TRAIN:
      optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
      train_op = optimizer.minimize(
          loss,
          global_step=tf.train.get_global_step())
      return EstimatorSpec(
          mode,
          loss=loss,
          train_op=train_op)

  tf_config = config_pb2.ConfigProto()
  tf_config.gpu_options.allow_growth = True
  estimator = Estimator(
      model_fn=model_fn,
      model_dir=None,
      config=RunConfig(session_config=tf_config))
  if is_training:
    estimator.train(train_input_fn)
  results = estimator.evaluate(eval_input_fn)
  print('accuracy:', results['accuracy'])
  return results

def get_graph_def(use_trt, batch_size, model_dir):
  # Load graph and freeze
  with tf.Graph().as_default() as graph:
    with tf.Session() as sess:
      x = tf.placeholder(shape=(None, 28, 28, 1),
                         dtype=tf.float32,
                         name=INPUT_NODE_NAME)
      logits_out = build_graph(x)
      # Load weights
      saver = tf.train.Saver()
      checkpoint_file = tf.train.latest_checkpoint(model_dir)
      saver.restore(sess, checkpoint_file)
      # Freeze
      graph_def = tf.graph_util.convert_variables_to_constants(
          sess,
          sess.graph_def,
          output_node_names=[OUTPUT_NODE_NAME]
      )
  # Convert with TF-TRT
  if use_trt:
    print('nodes before:', len(graph_def.node))
    graph_def = create_inference_graph(graph_def,
        outputs=[OUTPUT_NODE_NAME],
        max_batch_size=batch_size,
        precision_mode='int8',
        max_workspace_size_bytes=4096 << 19,
        minimum_segment_size=2,
        use_calibration=False,
    )
    print('tftrt total nodes:', len(graph_def.node))
    print('trt only nodes',
        len([1 for n in graph_def.node if str(n.op)=='TRTEngineOp']))
  return graph_def


class QuantizationAwareTrainingMNISTTest(test_util.TensorFlowTestCase):

  def testEval(self):
    model_dir = test.test_src_dir_path(
        'contrib/tensorrt/test/quantization_mnist_test_data')
    acc_tf = run(is_training=False,
        use_trt=False,
        batch_size=128,
        num_epochs=None,
        model_dir=model_dir)['accuracy']
    acc_tftrt = run(is_training=False,
        use_trt=True,
        batch_size=128,
        num_epochs=None,
        model_dir=model_dir)['accuracy']
    self.assertAllClose(acc_tf, 0.9717)
    self.assertAllClose(acc_tftrt, 0.9744)

if __name__ == "__main__":
  test.main()
