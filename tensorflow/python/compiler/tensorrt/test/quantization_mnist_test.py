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
"""Script to test TF-TRT INT8 conversion without calibration on Mnist model."""

import os.path
import tempfile
import tensorflow_datasets as tfds
from tensorflow.compiler.tf2tensorrt._pywrap_py_utils import is_tensorrt_enabled
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.compiler.tensorrt import trt_convert
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.estimator.model_fn import EstimatorSpec
from tensorflow.python.estimator.model_fn import ModeKeys
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras.metrics import Accuracy
from tensorflow.python.layers import layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import metrics
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils as saved_model_utils
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model.load import load as saved_model_load
from tensorflow.python.summary import summary
from tensorflow.python.training import saver
from tensorflow.python.training.adam import AdamOptimizer
from tensorflow.python.training.checkpoint_management import latest_checkpoint
from tensorflow.python.training.training_util import get_global_step

INPUT_NODE_NAME = 'input'
OUTPUT_NODE_NAME = 'output'
MNIST_TEST_DIR_PATH = 'python/compiler/tensorrt/test/testdata/mnist'


def _PreprocessFn(entry):
  """Normalizes the pixel values to lay within the [-1, 1] range.

   The same normalization shall be used during training and inference.
  """
  x, y = entry['image'], entry['label']
  x = math_ops.cast(x, dtypes.float32)
  x = 2.0 * (x / 255.0) - 1.0
  y = math_ops.cast(y, dtypes.int32)
  return x, y


def _GetDataSet(batch_size):
  dataset = tfds.load('mnist', split='test')
  dataset = dataset.map(
      map_func=_PreprocessFn, num_parallel_calls=8).batch(batch_size=batch_size)
  dataset = dataset.repeat(count=1)
  return dataset


class QuantizationAwareTrainingMNISTTest(test_util.TensorFlowTestCase):
  """Testing usage of quantization ranges inserted in graph."""

  def _BuildGraph(self, x):

    def _Quantize(x, r):
      x = gen_array_ops.quantize_and_dequantize_v2(x, -r, r)
      return x

    def _DenseLayer(x, num_inputs, num_outputs, quantization_range, name):
      """Defines a dense layer with quantized outputs.

      Args:
        x: input to the dense layer
        num_inputs: number of input columns of x
        num_outputs: number of output columns
        quantization_range: the min/max range for quantization
        name: name of the variable scope

      Returns:
        The output of the layer.
      """
      with variable_scope.variable_scope(name):
        kernel = variable_scope.get_variable(
            'kernel',
            shape=[num_inputs, num_outputs],
            dtype=dtypes.float32,
            initializer=init_ops.GlorotUniform())
        bias = variable_scope.get_variable(
            'bias',
            shape=[num_outputs],
            dtype=dtypes.float32,
            initializer=init_ops.Zeros())
        x = math_ops.matmul(x, kernel)
        x = _Quantize(x, quantization_range)
        x = nn.bias_add(x, bias)
        x = _Quantize(x, quantization_range)
      return x

    x = _Quantize(x, 1)
    # Conv + Bias + Relu6
    x = layers.conv2d(x, filters=32, kernel_size=3, use_bias=True)
    x = nn.relu6(x)
    # Conv + Bias + Relu6
    x = layers.conv2d(x, filters=64, kernel_size=3, use_bias=True)
    x = nn.relu6(x)
    # Reduce
    x = math_ops.reduce_mean(x, [1, 2])
    x = _Quantize(x, 6)
    # FC1
    x = _DenseLayer(x, 64, 512, 6, name='dense')
    x = nn.relu6(x)
    # FC2
    x = _DenseLayer(x, 512, 10, 25, name='dense_1')
    x = array_ops.identity(x, name=OUTPUT_NODE_NAME)
    return x

  def _LoadWeights(self, model_dir, sess):
    mnist_saver = saver.Saver()
    checkpoint_file = latest_checkpoint(model_dir)
    if checkpoint_file is None:
      raise ValueError('latest_checkpoint returned None. check if' +
                       'model_dir={} is the right directory'.format(model_dir))
    mnist_saver.restore(sess, checkpoint_file)

  def _GetGraphDef(self, use_trt, max_batch_size, model_dir):
    """Gets the frozen mnist GraphDef.

    Args:
      use_trt: whether use TF-TRT to convert the graph.
      max_batch_size: the max batch size to apply during TF-TRT conversion.
      model_dir: the model directory to load the checkpoints.

    Returns:
      The frozen mnist GraphDef.
    """
    graph = ops.Graph()
    with self.session(graph=graph) as sess:
      with graph.device('/GPU:0'):
        x = array_ops.placeholder(
            shape=(None, 28, 28, 1), dtype=dtypes.float32, name=INPUT_NODE_NAME)
        self._BuildGraph(x)
      self._LoadWeights(model_dir, sess)
      # Freeze
      graph_def = graph_util.convert_variables_to_constants(
          sess, sess.graph_def, output_node_names=[OUTPUT_NODE_NAME])
    # Convert with TF-TRT
    if use_trt:
      logging.info('Number of nodes before TF-TRT conversion: %d',
                   len(graph_def.node))
      converter = trt_convert.TrtGraphConverter(
          input_graph_def=graph_def,
          nodes_denylist=[OUTPUT_NODE_NAME],
          max_batch_size=max_batch_size,
          precision_mode='INT8',
          # There is a 2GB GPU memory limit for each test, so we set
          # max_workspace_size_bytes to 256MB to leave enough room for TF
          # runtime to allocate GPU memory.
          max_workspace_size_bytes=1 << 28,
          minimum_segment_size=2,
          use_calibration=False)
      graph_def = converter.convert()
      logging.info('Number of nodes after TF-TRT conversion: %d',
                   len(graph_def.node))
      num_engines = len(
          [1 for n in graph_def.node if str(n.op) == 'TRTEngineOp'])
      self.assertEqual(1, num_engines)
    return graph_def

  def _Run(self, is_training, use_trt, batch_size, num_epochs, model_dir):
    """Trains or evaluates the model.

    Args:
      is_training: whether to train or evaluate the model. In training mode,
        quantization will be simulated where the quantize_and_dequantize_v2 are
        placed.
      use_trt: if true, use TRT INT8 mode for evaluation, which will perform
        real quantization. Otherwise use native TensorFlow which will perform
        simulated quantization. Ignored if is_training is True.
      batch_size: batch size.
      num_epochs: how many epochs to train. Ignored if is_training is False.
      model_dir: where to save or load checkpoint.

    Returns:
      The Estimator evaluation result.
    """

    def _EvalInputFn():
      dataset = _GetDataSet(batch_size)
      iterator = dataset_ops.make_one_shot_iterator(dataset)
      features, labels = iterator.get_next()
      return features, labels

    def _TrainInputFn():
      dataset = tfds.load('mnist', split='train')
      dataset = dataset.shuffle(60000)
      dataset = dataset.map(
          map_func=_PreprocessFn,
          num_parallel_calls=8).batch(batch_size=batch_size)
      dataset = dataset.repeat(count=num_epochs)
      iterator = dataset_ops.make_one_shot_iterator(dataset)
      features, labels = iterator.get_next()
      return features, labels

    def _ModelFn(features, labels, mode):
      if is_training:
        logits_out = self._BuildGraph(features)
      else:
        graph_def = self._GetGraphDef(use_trt, batch_size, model_dir)
        logits_out = importer.import_graph_def(
            graph_def,
            input_map={INPUT_NODE_NAME: features},
            return_elements=[OUTPUT_NODE_NAME + ':0'],
            name='')[0]

      loss = losses.sparse_softmax_cross_entropy(
          labels=labels, logits=logits_out)
      summary.scalar('loss', loss)

      classes_out = math_ops.argmax(logits_out, axis=1, name='classes_out')
      accuracy = metrics.accuracy(
          labels=labels, predictions=classes_out, name='acc_op')
      summary.scalar('accuracy', accuracy[1])

      if mode == ModeKeys.EVAL:
        return EstimatorSpec(
            mode, loss=loss, eval_metric_ops={'accuracy': accuracy})
      if mode == ModeKeys.TRAIN:
        optimizer = AdamOptimizer(learning_rate=1e-2)
        train_op = optimizer.minimize(loss, global_step=get_global_step())
        return EstimatorSpec(mode, loss=loss, train_op=train_op)

    config_proto = config_pb2.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    estimator = Estimator(
        model_fn=_ModelFn,
        model_dir=model_dir if is_training else None,
        config=RunConfig(session_config=config_proto))

    if is_training:
      estimator.train(_TrainInputFn)
    results = estimator.evaluate(_EvalInputFn)
    logging.info('accuracy: %s', str(results['accuracy']))
    return results

  # To generate the checkpoint, set a different model_dir and call self._Run()
  # by setting is_training=True and num_epochs=1000, e.g.:
  # model_dir = '/tmp/quantization_mnist'
  # self._Run(
  #     is_training=True,
  #     use_trt=False,
  #     batch_size=128,
  #     num_epochs=100,
  #     model_dir=model_dir)
  def testEval(self):

    model_dir = test.test_src_dir_path(MNIST_TEST_DIR_PATH)

    accuracy_tf_native = self._Run(
        is_training=False,
        use_trt=False,
        batch_size=128,
        num_epochs=None,
        model_dir=model_dir)['accuracy']
    logging.info('accuracy_tf_native: %f', accuracy_tf_native)
    self.assertAllClose(0.9662, accuracy_tf_native, rtol=3e-3, atol=3e-3)

    accuracy_tf_trt = self._Run(
        is_training=False,
        use_trt=True,
        batch_size=128,
        num_epochs=None,
        model_dir=model_dir)['accuracy']
    logging.info('accuracy_tf_trt: %f', accuracy_tf_trt)
    self.assertAllClose(0.9675, accuracy_tf_trt, rtol=1e-3, atol=1e-3)


class MNISTTestV2(QuantizationAwareTrainingMNISTTest):

  def _SaveModel(self, model_dir, output_dir):
    saved_model_builder = builder.SavedModelBuilder(output_dir)
    graph = ops.Graph()
    with session.Session(graph=graph) as sess:
      with graph.device('/GPU:0'):
        x = array_ops.placeholder(
            shape=(None, 28, 28, 1), dtype=dtypes.float32, name=INPUT_NODE_NAME)
        self._BuildGraph(x)
      self._LoadWeights(model_dir, sess)
      input_tensor = graph.get_tensor_by_name(INPUT_NODE_NAME + ':0')
      output = graph.get_tensor_by_name(OUTPUT_NODE_NAME + ':0')
      signature_def = signature_def_utils.build_signature_def(
          inputs={'input': saved_model_utils.build_tensor_info(input_tensor)},
          outputs={'output': saved_model_utils.build_tensor_info(output)},
          method_name=signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)
      saved_model_builder.add_meta_graph_and_variables(
          sess, [tag_constants.SERVING],
          signature_def_map={
              signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                  signature_def
          })
    saved_model_builder.save()

  def _GetFunc(self, use_trt, model_dir, use_dynamic_shape):
    """Gets the mnist function.

    Args:
      use_trt: whether use TF-TRT to convert the graph.
      model_dir: the model directory to load the checkpoints.
      use_dynamic_shape: whether to run the TF-TRT conversion in dynamic shape
        mode.

    Returns:
      The mnist model function.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
      saved_model_dir = os.path.join(tmpdir, 'mnist')
      self._SaveModel(model_dir, saved_model_dir)

      if use_trt:
        conv_params = trt_convert.TrtConversionParams(
            precision_mode='FP16',
            minimum_segment_size=2,
            max_workspace_size_bytes=1 << 28,
            maximum_cached_engines=1)
        converter = trt_convert.TrtGraphConverterV2(
            input_saved_model_dir=saved_model_dir,
            conversion_params=conv_params,
            use_dynamic_shape=use_dynamic_shape,
            dynamic_shape_profile_strategy='ImplicitBatchModeCompatible')
        converter.convert()
        func = converter._converted_func
      else:
        saved_model_loaded = saved_model_load(
            saved_model_dir, tags=[tag_constants.SERVING])
        func = saved_model_loaded.signatures[
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    return func

  def _Run(self, use_trt, batch_size, model_dir, use_dynamic_shape=False):
    """Evaluates the model.

    Args:
      use_trt: if true, use TRT INT8 mode for evaluation, which will perform
        real quantization. Otherwise use native TensorFlow which will perform
        simulated quantization. Ignored if is_training is True.
      batch_size: batch size.
      model_dir: where to save or load checkpoint.
      use_dynamic_shape: if true, then TF-TRT dynamic shape mode is enabled,
        otherwise disabled. Ignored if use_trt is false.

    Returns:
      The Estimator evaluation result.
    """
    func = self._GetFunc(use_trt, model_dir, use_dynamic_shape)
    ds = _GetDataSet(batch_size)

    m = Accuracy()
    for example in ds:
      image, label = example[0], example[1]
      pred = func(image)
      m.update_state(math_ops.argmax(pred['output'], axis=1), label)

    return m.result().numpy()

  def testEval(self):
    model_dir = test.test_src_dir_path(MNIST_TEST_DIR_PATH)

    accuracy_tf_trt = self._Run(
        use_trt=True,
        batch_size=128,
        use_dynamic_shape=False,
        model_dir=model_dir)
    logging.info('accuracy_tf_trt: %f', accuracy_tf_trt)
    self.assertAllClose(0.9675, accuracy_tf_trt, rtol=1e-3, atol=1e-3)

    accuracy_tf_trt = self._Run(
        use_trt=True,
        batch_size=128,
        use_dynamic_shape=True,
        model_dir=model_dir)
    logging.info('accuracy_tf_trt: %f', accuracy_tf_trt)
    self.assertAllClose(0.9675, accuracy_tf_trt, rtol=1e-3, atol=1e-3)

if __name__ == '__main__' and is_tensorrt_enabled():
  test.main()
