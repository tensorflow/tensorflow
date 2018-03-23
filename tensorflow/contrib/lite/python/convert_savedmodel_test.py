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
"""TF Lite SavedModel Conversion test cases.

 - test on generated saved_models from simple graphs (sanity check)
 - test mnist savedmodel generated on-the-fly

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensorflow.contrib.lite.python import convert_savedmodel
from tensorflow.python import estimator
from tensorflow.python import keras
from tensorflow.python import layers
from tensorflow.python import losses
from tensorflow.python import nn
from tensorflow.python import saved_model
from tensorflow.python import train
from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test


class ConvertSavedModelTestBasicGraph(test_util.TensorFlowTestCase):

  def _createSimpleSavedModel(self, shape):
    """Create a simple savedmodel on the fly."""
    saved_model_dir = os.path.join(self.get_temp_dir(), "simple_savedmodel")
    with session.Session() as sess:
      in_tensor = array_ops.placeholder(shape=shape, dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      inputs = {"x": in_tensor}
      outputs = {"y": out_tensor}
      saved_model.simple_save(sess, saved_model_dir, inputs, outputs)
    return saved_model_dir

  def testSimpleSavedModel(self):
    """Test a simple savedmodel created on the fly."""
    # Create a simple savedmodel
    saved_model_dir = self._createSimpleSavedModel(shape=[1, 16, 16, 3])
    # Convert to tflite
    result = convert_savedmodel.convert(saved_model_dir=saved_model_dir)
    self.assertTrue(result)

  def testSimpleSavedModelWithNoneBatchSizeInShape(self):
    """Test a simple savedmodel, with None in input tensor's shape."""
    saved_model_dir = self._createSimpleSavedModel(shape=[None, 16, 16, 3])
    result = convert_savedmodel.convert(saved_model_dir=saved_model_dir)
    self.assertTrue(result)

  def testSimpleSavedModelWithMoreNoneInShape(self):
    """Test a simple savedmodel, fail as more None in input shape."""
    saved_model_dir = self._createSimpleSavedModel(shape=[None, 16, None, 3])
    # Convert to tflite: this should raise ValueError, as 3rd dim is None.
    with self.assertRaises(ValueError):
      convert_savedmodel.convert(saved_model_dir=saved_model_dir)

  def testSimpleSavedModelWithWrongSignatureKey(self):
    """Test a simple savedmodel, fail as given signature is invalid."""
    saved_model_dir = self._createSimpleSavedModel(shape=[1, 16, 16, 3])
    # Convert to tflite: this should raise ValueError, as
    # signature_key does not exit in the saved_model.
    with self.assertRaises(ValueError):
      convert_savedmodel.convert(
          saved_model_dir=saved_model_dir, signature_key="wrong-key")

  def testSimpleSavedModelWithWrongOutputArray(self):
    """Test a simple savedmodel, fail as given output_arrays is invalid."""
    # Create a simple savedmodel
    saved_model_dir = self._createSimpleSavedModel(shape=[1, 16, 16, 3])
    # Convert to tflite: this should raise ValueError, as
    # output_arrays is not valid for the saved_model.
    with self.assertRaises(ValueError):
      convert_savedmodel.convert(
          saved_model_dir=saved_model_dir, output_arrays="wrong-output")

  def testMultipleMetaGraphDef(self):
    """Test saved model with multiple MetaGraphDef."""
    saved_model_dir = os.path.join(self.get_temp_dir(), "savedmodel_two_mgd")
    builder = saved_model.builder.SavedModelBuilder(saved_model_dir)
    with session.Session(graph=ops.Graph()) as sess:
      # MetaGraphDef 1
      in_tensor = array_ops.placeholder(shape=[1, 28, 28], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sig_input_tensor = saved_model.utils.build_tensor_info(in_tensor)
      sig_input_tensor_signature = {"x": sig_input_tensor}
      sig_output_tensor = saved_model.utils.build_tensor_info(out_tensor)
      sig_output_tensor_signature = {"y": sig_output_tensor}
      predict_signature_def = (
          saved_model.signature_def_utils.build_signature_def(
              sig_input_tensor_signature, sig_output_tensor_signature,
              saved_model.signature_constants.PREDICT_METHOD_NAME))
      signature_def_map = {
          saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
              predict_signature_def
      }
      builder.add_meta_graph_and_variables(
          sess,
          tags=[saved_model.tag_constants.SERVING, "additional_test_tag"],
          signature_def_map=signature_def_map)
      # MetaGraphDef 2
      builder.add_meta_graph(tags=["tflite"])
      builder.save(True)

    # Convert to tflite
    convert_savedmodel.convert(
        saved_model_dir=saved_model_dir,
        tag_set=set([saved_model.tag_constants.SERVING, "additional_test_tag"]))


class Model(keras.Model):
  """Model to recognize digits in the MNIST dataset.

  Train and export savedmodel, used for testOnflyTrainMnistSavedModel

  Network structure is equivalent to:
  https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/examples/tutorials/mnist/mnist_deep.py
  and
  https://github.com/tensorflow/models/blob/master/tutorials/image/mnist/convolutional.py

  But written as a ops.keras.Model using the layers API.
  """

  def __init__(self, data_format):
    """Creates a model for classifying a hand-written digit.

    Args:
      data_format: Either "channels_first" or "channels_last".
        "channels_first" is typically faster on GPUs while "channels_last" is
        typically faster on CPUs. See
        https://www.tensorflow.org/performance/performance_guide#data_formats
    """
    super(Model, self).__init__()
    self._input_shape = [-1, 28, 28, 1]

    self.conv1 = layers.Conv2D(
        32, 5, padding="same", data_format=data_format, activation=nn.relu)
    self.conv2 = layers.Conv2D(
        64, 5, padding="same", data_format=data_format, activation=nn.relu)
    self.fc1 = layers.Dense(1024, activation=nn.relu)
    self.fc2 = layers.Dense(10)
    self.dropout = layers.Dropout(0.4)
    self.max_pool2d = layers.MaxPooling2D(
        (2, 2), (2, 2), padding="same", data_format=data_format)

  def __call__(self, inputs, training):
    """Add operations to classify a batch of input images.

    Args:
      inputs: A Tensor representing a batch of input images.
      training: A boolean. Set to True to add operations required only when
        training the classifier.

    Returns:
      A logits Tensor with shape [<batch_size>, 10].
    """
    y = array_ops.reshape(inputs, self._input_shape)
    y = self.conv1(y)
    y = self.max_pool2d(y)
    y = self.conv2(y)
    y = self.max_pool2d(y)
    y = layers.flatten(y)
    y = self.fc1(y)
    y = self.dropout(y, training=training)
    return self.fc2(y)


def model_fn(features, labels, mode, params):
  """The model_fn argument for creating an Estimator."""
  model = Model(params["data_format"])
  image = features
  if isinstance(image, dict):
    image = features["image"]

  if mode == estimator.ModeKeys.PREDICT:
    logits = model(image, training=False)
    predictions = {
        "classes": math_ops.argmax(logits, axis=1),
        "probabilities": nn.softmax(logits),
    }
    return estimator.EstimatorSpec(
        mode=estimator.ModeKeys.PREDICT,
        predictions=predictions,
        export_outputs={
            "classify": estimator.export.PredictOutput(predictions)
        })

  elif mode == estimator.ModeKeys.TRAIN:
    optimizer = train.AdamOptimizer(learning_rate=1e-4)

    logits = model(image, training=True)
    loss = losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    return estimator.EstimatorSpec(
        mode=estimator.ModeKeys.TRAIN,
        loss=loss,
        train_op=optimizer.minimize(loss, train.get_or_create_global_step()))

  elif mode == estimator.ModeKeys.EVAL:
    logits = model(image, training=False)
    loss = losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    return estimator.EstimatorSpec(
        mode=estimator.ModeKeys.EVAL,
        loss=loss,
        eval_metric_ops={
            "accuracy":
                ops.metrics.accuracy(
                    labels=labels, predictions=math_ops.argmax(logits, axis=1)),
        })


def dummy_input_fn():
  image = random_ops.random_uniform([100, 784])
  labels = random_ops.random_uniform([100, 1], maxval=9, dtype=dtypes.int32)
  return image, labels


class ConvertSavedModelTestTrainGraph(test_util.TensorFlowTestCase):

  def testTrainedMnistSavedModel(self):
    """Test mnist savedmodel, trained with dummy data and small steps."""
    # Build classifier
    classifier = estimator.Estimator(
        model_fn=model_fn,
        params={
            "data_format": "channels_last"  # tflite format
        })

    # Train and pred for serving
    classifier.train(input_fn=dummy_input_fn, steps=2)
    image = array_ops.placeholder(dtypes.float32, [None, 28, 28])
    pred_input_fn = estimator.export.build_raw_serving_input_receiver_fn({
        "image": image,
    })

    # Export savedmodel
    saved_model_dir = os.path.join(self.get_temp_dir(), "mnist_savedmodel")
    classifier.export_savedmodel(saved_model_dir, pred_input_fn)

    # Convert to tflite and test output
    saved_model_name = os.listdir(saved_model_dir)[0]
    saved_model_final_dir = os.path.join(saved_model_dir, saved_model_name)
    output_tflite = os.path.join(saved_model_dir,
                                 saved_model_final_dir + ".lite")
    # TODO(zhixianyan): no need to limit output_arrays to `Softmax'
    # once b/74205001 fixed and argmax implemented in tflite.
    result = convert_savedmodel.convert(
        saved_model_dir=saved_model_final_dir,
        output_arrays="Softmax",
        output_tflite=output_tflite)

    self.assertTrue(result)


if __name__ == "__main__":
  test.main()
