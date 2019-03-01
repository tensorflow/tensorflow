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

import os
import numpy as np

from tensorflow.core.framework import graph_pb2 as _graph_pb2
from tensorflow.lite.python import convert_saved_model as _convert_saved_model
from tensorflow.lite.python import lite as _lite
from tensorflow.python import keras as _keras
from tensorflow.python.client import session as _session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework.importer import import_graph_def as _import_graph_def
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.lib.io import file_io as _file_io
from tensorflow.python.platform import resource_loader as _resource_loader
from tensorflow.python.saved_model import load as _load
from tensorflow.python.saved_model import loader as _loader
from tensorflow.python.saved_model import signature_constants as _signature_constants
from tensorflow.python.saved_model import tag_constants as _tag_constants


def get_filepath(filename, base_dir=None):
  """Returns the full path of the filename.

  Args:
    filename: Subdirectory and name of the model file.
    base_dir: Base directory containing model file.

  Returns:
    str.
  """
  if base_dir is None:
    base_dir = "learning/brain/mobile/tflite_compat_models"
  return os.path.join(_resource_loader.get_root_dir_with_all_resources(),
                      base_dir, filename)


def get_image(size):
  """Returns an image loaded into an np.ndarray with dims [1, size, size, 3].

  Args:
    size: Size of image.

  Returns:
    np.ndarray.
  """
  img_filename = _resource_loader.get_path_to_datafile(
      "testdata/grace_hopper.jpg")
  img = image.load_img(img_filename, target_size=(size, size))
  img_array = image.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)
  return img_array


def _convert(converter, **kwargs):
  """Converts the model.

  Args:
    converter: TFLiteConverter object.
    **kwargs: Additional arguments to be passed into the converter. Supported
      flags are {"target_ops", "post_training_quantize"}.

  Returns:
    The converted TFLite model in serialized format.
  """
  if "target_ops" in kwargs:
    converter.target_ops = kwargs["target_ops"]
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

    meta_graph = _loader.load(sess, tag_set, directory)
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


def compare_models(tflite_model, tf_eval_func, input_data=None, tolerance=5):
  """Compares TensorFlow and TFLite models.

  Unless the input data is provided, the models are compared with random data.

  Args:
    tflite_model: Serialized TensorFlow Lite model.
    tf_eval_func: Lambda function that takes in input data and outputs the
      results of the TensorFlow model ([np.ndarray data] : [np.ndarray result]).
    input_data: np.ndarray to pass into models during inference. (default None)
    tolerance: Decimal place to check accuracy to. (default 5)
  """
  if input_data is None:
    input_data = _generate_random_input_data(tflite_model)
  tf_results = tf_eval_func(input_data)
  tflite_results = _evaluate_tflite_model(tflite_model, input_data)
  for tf_result, tflite_result in zip(tf_results, tflite_results):
    np.testing.assert_almost_equal(tf_result, tflite_result, tolerance)


def compare_models_v2(tflite_model, concrete_func, input_data=None,
                      tolerance=5):
  """Compares TensorFlow and TFLite models for TensorFlow 2.0.

  Unless the input data is provided, the models are compared with random data.
  Currently only 1 input and 1 output are supported by this function.

  Args:
    tflite_model: Serialized TensorFlow Lite model.
    concrete_func: TensorFlow ConcreteFunction.
    input_data: np.ndarray to pass into models during inference. (default None)
    tolerance: Decimal place to check accuracy to. (default 5)
  """
  if input_data is None:
    input_data = _generate_random_input_data(tflite_model)
  input_data_func = constant_op.constant(input_data[0])

  # Gets the TensorFlow results as a map from the output names to outputs.
  # Converts the map into a list that is equivalent to the TFLite list.
  tf_results_map = concrete_func(input_data_func)
  tf_results = [tf_results_map[tf_results_map.keys()[0]]]
  tflite_results = _evaluate_tflite_model(tflite_model, input_data)
  for tf_result, tflite_result in zip(tf_results, tflite_results):
    np.testing.assert_almost_equal(tf_result, tflite_result, tolerance)


def test_frozen_graph_quant(filename,
                            input_arrays,
                            output_arrays,
                            input_shapes=None,
                            **kwargs):
  """Sanity check to validate post quantize flag alters the graph.

  This test does not check correctness of the converted model. It converts the
  TensorFlow frozen graph to TFLite with and without the post_training_quantized
  flag. It ensures some tensors have different types between the float and
  quantized models in the case of an all TFLite model or mix-and-match model.
  It ensures tensor types do not change in the case of an all Flex model.

  Args:
    filename: Full filepath of file containing frozen GraphDef.
    input_arrays: List of input tensors to freeze graph with.
    output_arrays: List of output tensors to freeze graph with.
    input_shapes: Dict of strings representing input tensor names to list of
      integers representing input shapes (e.g., {"foo" : [1, 16, 16, 3]}).
      Automatically determined when input shapes is None (e.g., {"foo" : None}).
        (default None)
    **kwargs: Additional arguments to be passed into the converter.

  Raises:
    ValueError: post_training_quantize flag doesn't act as intended.
  """
  # Convert and load the float model.
  converter = _lite.TFLiteConverter.from_frozen_graph(
      filename, input_arrays, output_arrays, input_shapes)
  tflite_model_float = _convert(converter, **kwargs)

  interpreter_float = _lite.Interpreter(model_content=tflite_model_float)
  interpreter_float.allocate_tensors()
  float_tensors = interpreter_float.get_tensor_details()

  # Convert and load the quantized model.
  converter = _lite.TFLiteConverter.from_frozen_graph(filename, input_arrays,
                                                      output_arrays)
  tflite_model_quant = _convert(
      converter, post_training_quantize=True, **kwargs)

  interpreter_quant = _lite.Interpreter(model_content=tflite_model_quant)
  interpreter_quant.allocate_tensors()
  quant_tensors = interpreter_quant.get_tensor_details()
  quant_tensors_map = {
      tensor_detail["name"]: tensor_detail for tensor_detail in quant_tensors
  }

  # Check if weights are of different types in the float and quantized models.
  num_tensors_float = len(float_tensors)
  num_tensors_same_dtypes = sum(
      float_tensor["dtype"] == quant_tensors_map[float_tensor["name"]]["dtype"]
      for float_tensor in float_tensors)
  has_quant_tensor = num_tensors_float != num_tensors_same_dtypes

  if ("target_ops" in kwargs and
      set(kwargs["target_ops"]) == set([_lite.OpsSet.SELECT_TF_OPS])):
    if has_quant_tensor:
      raise ValueError("--post_training_quantize flag unexpectedly altered the "
                       "full Flex mode graph.")
  elif not has_quant_tensor:
    raise ValueError("--post_training_quantize flag was unable to quantize the "
                     "graph as expected in TFLite and mix-and-match mode.")


def test_frozen_graph(filename,
                      input_arrays,
                      output_arrays,
                      input_shapes=None,
                      input_data=None,
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
    input_data: np.ndarray to pass into models during inference. (default None)
    **kwargs: Additional arguments to be passed into the converter.
  """
  converter = _lite.TFLiteConverter.from_frozen_graph(
      filename, input_arrays, output_arrays, input_shapes)
  tflite_model = _convert(converter, **kwargs)

  tf_eval_func = evaluate_frozen_graph(filename, input_arrays, output_arrays)
  compare_models(tflite_model, tf_eval_func, input_data=input_data)


def test_saved_model(directory,
                     input_shapes=None,
                     tag_set=None,
                     signature_key=None,
                     input_data=None,
                     **kwargs):
  """Validates the TensorFlow SavedModel converts to a TFLite model.

  Converts the TensorFlow SavedModel to TFLite and checks the accuracy of the
  model on random data.

  Args:
    directory: SavedModel directory to convert.
    input_shapes: Dict of strings representing input tensor names to list of
      integers representing input shapes (e.g., {"foo" : [1, 16, 16, 3]}).
      Automatically determined when input shapes is None (e.g., {"foo" : None}).
        (default None)
    tag_set: Set of tags identifying the MetaGraphDef within the SavedModel to
      analyze. All tags in the tag set must be present.
    signature_key: Key identifying SignatureDef containing inputs and outputs.
    input_data: np.ndarray to pass into models during inference. (default None)
    **kwargs: Additional arguments to be passed into the converter.
  """
  converter = _lite.TFLiteConverter.from_saved_model(
      directory,
      input_shapes=input_shapes,
      tag_set=tag_set,
      signature_key=signature_key)
  tflite_model = _convert(converter, **kwargs)

  tf_eval_func = evaluate_saved_model(directory, tag_set, signature_key)
  compare_models(tflite_model, tf_eval_func, input_data=input_data)


# TODO(nupurgarg): Remove input_shape parameter after bug with shapes is fixed.
def test_saved_model_v2(directory,
                        input_shape=None,
                        tag_set=None,
                        signature_key=None,
                        input_data=None,
                        **kwargs):
  """Validates the TensorFlow SavedModel converts to a TFLite model.

  Converts the TensorFlow SavedModel to TFLite and checks the accuracy of the
  model on random data.

  Args:
    directory: SavedModel directory to convert.
    input_shape: Input shape for the single input array as a list of integers.
    tag_set: Set of tags identifying the MetaGraphDef within the SavedModel to
      analyze. All tags in the tag set must be present.
    signature_key: Key identifying SignatureDef containing inputs and outputs.
    input_data: np.ndarray to pass into models during inference. (default None)
    **kwargs: Additional arguments to be passed into the converter.
  """
  model = _load.load(directory, tags=tag_set)
  if not signature_key:
    signature_key = _signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
  concrete_func = model.signatures[signature_key]
  concrete_func.inputs[0].set_shape(input_shape)

  converter = _lite.TFLiteConverterV2.from_concrete_function(concrete_func)
  tflite_model = _convert(converter, **kwargs)

  compare_models_v2(tflite_model, concrete_func, input_data=input_data)


def test_keras_model(filename,
                     input_arrays=None,
                     input_shapes=None,
                     input_data=None,
                     **kwargs):
  """Validates the tf.keras model converts to a TFLite model.

  Converts the tf.keras model to TFLite and checks the accuracy of the model on
  random data.

  Args:
    filename: Full filepath of HDF5 file containing the tf.keras model.
    input_arrays: List of input tensors to freeze graph with.
    input_shapes: Dict of strings representing input tensor names to list of
      integers representing input shapes (e.g., {"foo" : [1, 16, 16, 3]}).
      Automatically determined when input shapes is None (e.g., {"foo" : None}).
        (default None)
    input_data: np.ndarray to pass into models during inference. (default None)
    **kwargs: Additional arguments to be passed into the converter.
  """
  converter = _lite.TFLiteConverter.from_keras_model_file(
      filename, input_arrays=input_arrays, input_shapes=input_shapes)
  tflite_model = _convert(converter, **kwargs)

  tf_eval_func = evaluate_keras_model(filename)
  compare_models(tflite_model, tf_eval_func, input_data=input_data)
