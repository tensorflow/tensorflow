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
"""Functions to test TFLite models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.lite.python import convert_saved_model as _convert_saved_model
from tensorflow.contrib.lite.python import lite as _lite
from tensorflow.core.framework import graph_pb2 as _graph_pb2
from tensorflow.python import keras as _keras
from tensorflow.python.client import session as _session
from tensorflow.python.framework.importer import import_graph_def as _import_graph_def
from tensorflow.python.lib.io import file_io as _file_io
from tensorflow.python.saved_model import signature_constants as _signature_constants
from tensorflow.python.saved_model import tag_constants as _tag_constants


def _convert(converter, **kwargs):
  """Converts the model.

  Args:
    converter: TocoConverter object.
    **kwargs: Additional arguments to be passed into the converter. Supported
      flags are {"converter_mode", "post_training_quant"}.

  Returns:
    The converted TFLite model in serialized format.
  """
  if "converter_mode" in kwargs:
    converter.converter_mode = kwargs["converter_mode"]
  if "post_training_quantize" in kwargs:
    converter.post_training_quantize = kwargs["post_training_quantize"]
  return converter.convert()


def _generate_random_input_data(tflite_model, seed=None):
  """Generates input data based on the input tensors in the TFLite model.

  Args:
    tflite_model: Serialized TensorFlow Lite model.
    seed: Integer seed for the random generator. (default None)

  Returns:
    List of np.ndarray.
  """
  interpreter = _lite.Interpreter(model_content=tflite_model)
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()

  if seed:
    np.random.seed(seed=seed)
  return [
      np.array(
          np.random.random_sample(input_tensor["shape"]),
          dtype=input_tensor["dtype"]) for input_tensor in input_details
  ]


def _evaluate_tflite_model(tflite_model, input_data):
  """Returns evaluation of input data on TFLite model.

  Args:
    tflite_model: Serialized TensorFlow Lite model.
    input_data: List of np.ndarray.

  Returns:
    List of np.ndarray.
  """
  interpreter = _lite.Interpreter(model_content=tflite_model)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  for input_tensor, tensor_data in zip(input_details, input_data):
    interpreter.set_tensor(input_tensor["index"], tensor_data)

  interpreter.invoke()
  output_data = [
      interpreter.get_tensor(output_tensor["index"])
      for output_tensor in output_details
  ]
  return output_data


def evaluate_frozen_graph(filename, input_arrays, output_arrays):
  """Returns a function that evaluates the frozen graph on input data.

  Args:
    filename: Full filepath of file containing frozen GraphDef.
    input_arrays: List of input tensors to freeze graph with.
    output_arrays: List of output tensors to freeze graph with.

  Returns:
    Lambda function ([np.ndarray data] : [np.ndarray result]).
  """
  with _session.Session().as_default() as sess:
    with _file_io.FileIO(filename, "rb") as f:
      file_content = f.read()

    graph_def = _graph_pb2.GraphDef()
    graph_def.ParseFromString(file_content)
    _import_graph_def(graph_def, name="")

    inputs = _convert_saved_model.get_tensors_from_tensor_names(
        sess.graph, input_arrays)
    outputs = _convert_saved_model.get_tensors_from_tensor_names(
        sess.graph, output_arrays)

    return lambda input_data: sess.run(outputs, dict(zip(inputs, input_data)))


def evaluate_saved_model(directory, tag_set, signature_key):
  """Returns a function that evaluates the SavedModel on input data.

  Args:
    directory: SavedModel directory to convert.
    tag_set: Set of tags identifying the MetaGraphDef within the SavedModel to
      analyze. All tags in the tag set must be present.
    signature_key: Key identifying SignatureDef containing inputs and outputs.

  Returns:
    Lambda function ([np.ndarray data] : [np.ndarray result]).
  """
  with _session.Session().as_default() as sess:
    if tag_set is None:
      tag_set = set([_tag_constants.SERVING])
    if signature_key is None:
      signature_key = _signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

    meta_graph = _convert_saved_model.get_meta_graph_def(directory, tag_set)
    signature_def = _convert_saved_model.get_signature_def(
        meta_graph, signature_key)
    inputs, outputs = _convert_saved_model.get_inputs_outputs(signature_def)

    return lambda input_data: sess.run(outputs, dict(zip(inputs, input_data)))


def evaluate_keras_model(filename):
  """Returns a function that evaluates the tf.keras model on input data.

  Args:
    filename: Full filepath of HDF5 file containing the tf.keras model.

  Returns:
    Lambda function ([np.ndarray data] : [np.ndarray result]).
  """
  keras_model = _keras.models.load_model(filename)
  return lambda input_data: [keras_model.predict(input_data)]


# TODO(nupurgarg): Make this function a parameter to test_frozen_graph (and
# related functions) in order to make it easy to use different data generators.
def compare_models_random_data(tflite_model, tf_eval_func, tolerance=5):
  """Compares TensorFlow and TFLite models with random data.

  Args:
    tflite_model: Serialized TensorFlow Lite model.
    tf_eval_func: Lambda function that takes in input data and outputs the
      results of the TensorFlow model ([np.ndarray data] : [np.ndarray result]).
    tolerance: Decimal place to check accuracy to.
  """
  input_data = _generate_random_input_data(tflite_model)
  tf_results = tf_eval_func(input_data)
  tflite_results = _evaluate_tflite_model(tflite_model, input_data)
  for tf_result, tflite_result in zip(tf_results, tflite_results):
    np.testing.assert_almost_equal(tf_result, tflite_result, tolerance)


def test_frozen_graph(filename,
                      input_arrays,
                      output_arrays,
                      input_shapes=None,
                      **kwargs):
  """Validates the TensorFlow frozen graph converts to a TFLite model.

  Converts the TensorFlow frozen graph to TFLite and checks the accuracy of the
  model on random data.

  Args:
    filename: Full filepath of file containing frozen GraphDef.
    input_arrays: List of input tensors to freeze graph with.
    output_arrays: List of output tensors to freeze graph with.
    input_shapes: Dict of strings representing input tensor names to list of
      integers representing input shapes (e.g., {"foo" : [1, 16, 16, 3]}).
      Automatically determined when input shapes is None (e.g., {"foo" : None}).
        (default None)
    **kwargs: Additional arguments to be passed into the converter.
  """
  converter = _lite.TocoConverter.from_frozen_graph(filename, input_arrays,
                                                    output_arrays, input_shapes)
  tflite_model = _convert(converter, **kwargs)

  tf_eval_func = evaluate_frozen_graph(filename, input_arrays, output_arrays)
  compare_models_random_data(tflite_model, tf_eval_func)


def test_saved_model(directory, tag_set=None, signature_key=None, **kwargs):
  """Validates the TensorFlow SavedModel converts to a TFLite model.

  Converts the TensorFlow SavedModel to TFLite and checks the accuracy of the
  model on random data.

  Args:
    directory: SavedModel directory to convert.
    tag_set: Set of tags identifying the MetaGraphDef within the SavedModel to
      analyze. All tags in the tag set must be present.
    signature_key: Key identifying SignatureDef containing inputs and outputs.
    **kwargs: Additional arguments to be passed into the converter.
  """
  converter = _lite.TocoConverter.from_saved_model(directory, tag_set,
                                                   signature_key)
  tflite_model = _convert(converter, **kwargs)

  tf_eval_func = evaluate_saved_model(directory, tag_set, signature_key)
  compare_models_random_data(tflite_model, tf_eval_func)


def test_keras_model(filename, **kwargs):
  """Validates the tf.keras model converts to a TFLite model.

  Converts the tf.keras model to TFLite and checks the accuracy of the model on
  random data.

  Args:
    filename: Full filepath of HDF5 file containing the tf.keras model.
    **kwargs: Additional arguments to be passed into the converter.
  """
  converter = _lite.TocoConverter.from_keras_model_file(filename)
  tflite_model = _convert(converter, **kwargs)

  tf_eval_func = evaluate_keras_model(filename)
  compare_models_random_data(tflite_model, tf_eval_func)
