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
from six import PY2
from tensorflow import keras

from google.protobuf import text_format as _text_format
from google.protobuf.message import DecodeError
from tensorflow.core.framework import graph_pb2 as _graph_pb2
from tensorflow.lite.python import convert_saved_model as _convert_saved_model
from tensorflow.lite.python import interpreter as _interpreter
from tensorflow.lite.python import lite as _lite
from tensorflow.lite.python import util as _util
from tensorflow.python.client import session as _session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework.importer import import_graph_def as _import_graph_def
from tensorflow.python.lib.io import file_io as _file_io
from tensorflow.python.platform import resource_loader as _resource_loader
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import load as _load
from tensorflow.python.saved_model import loader as _loader
from tensorflow.python.saved_model import signature_constants as _signature_constants
from tensorflow.python.saved_model import tag_constants as _tag_constants


_GOLDENS_UPDATE_WARNING = """
  Golden file update requested!
  This test is now going to write new golden files.

  Make sure to package the updates together with your CL.
"""


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


def get_golden_filepath(name):
  """Returns the full path to a golden values file.

  Args:
    name: the name of the golden data, usually same as the test name.
  """
  goldens_directory = os.path.join(_resource_loader.get_data_files_path(),
                                   "testdata", "golden")
  return os.path.join(goldens_directory, "%s.npy.golden" % name)


def get_image(size):
  """Returns an image loaded into an np.ndarray with dims [1, size, size, 3].

  Args:
    size: Size of image.

  Returns:
    np.ndarray.
  """
  img_filename = _resource_loader.get_path_to_datafile(
      "testdata/grace_hopper.jpg")
  img = keras.preprocessing.image.load_img(
      img_filename, target_size=(size, size))
  img_array = keras.preprocessing.image.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)
  return img_array


def _get_calib_data_func(input_size):
  """Returns a function to generate a representative data set.

  Args:
    input_size: 3D shape of the representative data.
  """
  def representative_data_gen():
    num_calibration = 20
    for _ in range(num_calibration):
      yield [
          np.random.rand(
              1,
              input_size[0],
              input_size[1],
              input_size[2],
          ).astype(np.float32)
      ]

  return representative_data_gen


def _convert(converter, **kwargs):
  """Converts the model.

  Args:
    converter: TFLiteConverter object.
    **kwargs: Additional arguments to be passed into the converter. Supported
      flags are {"target_ops", "post_training_quantize", "quantize_to_float16",
      "post_training_quantize_int8", "post_training_quantize_16x8",
      "model_input_size"}.

  Returns:
    The converted TFLite model in serialized format.

  Raises:
    ValueError: Invalid version number.
  """

  if "target_ops" in kwargs:
    converter.target_spec.supported_ops = kwargs["target_ops"]
  if "post_training_quantize" in kwargs:
    converter.optimizations = [_lite.Optimize.DEFAULT]
  if kwargs.get("quantize_to_float16", False):
    converter.target_spec.supported_types = [dtypes.float16]
  if kwargs.get("post_training_quantize_int8", False):
    input_size = kwargs.get("model_input_size")
    converter.optimizations = [_lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [_lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.representative_dataset = _get_calib_data_func(input_size)
    # Note that the full integer quantization is by the mlir quantizer
    converter._experimental_new_quantizer = True  # pylint: disable=protected-access
  if kwargs.get("post_training_quantize_16x8", False):
    input_size = kwargs.get("model_input_size")
    converter.optimizations = [_lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = \
      [_lite.OpsSet.\
        EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
    converter.representative_dataset = _get_calib_data_func(input_size)
  return converter.convert()


def _check_model_quantized_to_16x8(tflite_model):
  """Checks that the activations are quantized into int16.

  Args:
    tflite_model: Serialized TensorFlow Lite model.

  Raises:
    ValueError: Activations with int16 type are not found.
  """
  interpreter = _get_tflite_interpreter(tflite_model)
  interpreter.allocate_tensors()
  all_tensor_details = interpreter.get_tensor_details()

  found_input = False
  for tensor in all_tensor_details:
    if "_int16" in tensor["name"]:
      found_input = True
      if tensor["dtype"] is not np.int16:
        raise ValueError("Activations should be int16.")

  # Check that we found activations in the correct type: int16
  if not found_input:
    raise ValueError("Could not find int16 activations.")


def _get_tflite_interpreter(tflite_model,
                            input_shapes_resize=None,
                            custom_op_registerers=None):
  """Creates a TFLite interpreter with resized input tensors.

  Args:
    tflite_model: Serialized TensorFlow Lite model.
    input_shapes_resize: A map where the key is the input tensor name and the
      value is the shape of the input tensor. This resize happens after model
      conversion, prior to calling allocate tensors. (default None)
    custom_op_registerers: Op registerers for custom ops.

  Returns:
    lite.Interpreter
  """
  if custom_op_registerers is None:
    custom_op_registerers = []
  interpreter = _interpreter.InterpreterWithCustomOps(
      model_content=tflite_model, custom_op_registerers=custom_op_registerers)
  if input_shapes_resize:
    input_details = interpreter.get_input_details()
    input_details_map = {
        detail["name"]: detail["index"] for detail in input_details
    }
    for name, shape in input_shapes_resize.items():
      idx = input_details_map[name]
      interpreter.resize_tensor_input(idx, shape)
  return interpreter


def _get_input_data_map(tflite_model, input_data, custom_op_registerers=None):
  """Generates a map of input data based on the TFLite model.

  Args:
    tflite_model: Serialized TensorFlow Lite model.
    input_data: List of np.ndarray.
    custom_op_registerers: Op registerers for custom ops.

  Returns:
    {str: [np.ndarray]}.
  """
  interpreter = _get_tflite_interpreter(
      tflite_model, custom_op_registerers=custom_op_registerers)
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  return {
      input_tensor["name"]: data
      for input_tensor, data in zip(input_details, input_data)
  }


def _generate_random_input_data(tflite_model,
                                seed=None,
                                input_data_range=None,
                                input_shapes_resize=None,
                                custom_op_registerers=None):
  """Generates input data based on the input tensors in the TFLite model.

  Args:
    tflite_model: Serialized TensorFlow Lite model.
    seed: Integer seed for the random generator. (default None)
    input_data_range: A map where the key is the input tensor name and
      the value is a tuple (min_val, max_val) which specifies the value range of
      the corresponding input tensor. For example, '{'input1': (1, 5)}' means to
      generate a random value for tensor `input1` within range [1.0, 5.0)
      (half-inclusive). (default None)
    input_shapes_resize: A map where the key is the input tensor name and the
      value is the shape of the input tensor. This resize happens after model
      conversion, prior to calling allocate tensors. (default None)
    custom_op_registerers: Op registerers for custom ops.

  Returns:
    ([np.ndarray], {str : [np.ndarray]}).
  """
  interpreter = _get_tflite_interpreter(
      tflite_model,
      input_shapes_resize,
      custom_op_registerers=custom_op_registerers)
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()

  if seed:
    np.random.seed(seed=seed)

  # Generate random input data. If a tensor's value range is specified, say
  # [a, b), then the generated value will be (b - a) * Unif[0.0, 1.0) + a,
  # otherwise it's Unif[0.0, 1.0).
  input_data = []
  for input_tensor in input_details:
    val = np.random.random_sample(input_tensor["shape"])
    if (input_data_range is not None and
        input_tensor["name"] in input_data_range):
      val = (input_data_range[input_tensor["name"]][1] -
             input_data_range[input_tensor["name"]][0]
            ) * val + input_data_range[input_tensor["name"]][0]
    input_data.append(np.array(val, dtype=input_tensor["dtype"]))

  input_data_map = _get_input_data_map(
      tflite_model, input_data, custom_op_registerers=custom_op_registerers)
  return input_data, input_data_map


def _evaluate_tflite_model(tflite_model,
                           input_data,
                           input_shapes_resize=None,
                           custom_op_registerers=None):
  """Returns evaluation of input data on TFLite model.

  Args:
    tflite_model: Serialized TensorFlow Lite model.
    input_data: List of np.ndarray.
    input_shapes_resize: A map where the key is the input tensor name and the
      value is the shape of the input tensor. This resize happens after model
      conversion, prior to calling allocate tensors. (default None)
    custom_op_registerers: Op registerers for custom ops.

  Returns:
    List of np.ndarray.
  """
  interpreter = _get_tflite_interpreter(
      tflite_model,
      input_shapes_resize,
      custom_op_registerers=custom_op_registerers)
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
  output_labels = [output_tensor["name"] for output_tensor in output_details]
  return output_data, output_labels


def evaluate_frozen_graph(filename, input_arrays, output_arrays):
  """Returns a function that evaluates the frozen graph on input data.

  Args:
    filename: Full filepath of file containing frozen GraphDef.
    input_arrays: List of input tensors to freeze graph with.
    output_arrays: List of output tensors to freeze graph with.

  Returns:
    Lambda function ([np.ndarray data] : [np.ndarray result]).
  """
  with _file_io.FileIO(filename, "rb") as f:
    file_content = f.read()

  graph_def = _graph_pb2.GraphDef()
  try:
    graph_def.ParseFromString(file_content)
  except (_text_format.ParseError, DecodeError):
    if not isinstance(file_content, str):
      if PY2:
        file_content = file_content.encode("utf-8")
      else:
        file_content = file_content.decode("utf-8")
    _text_format.Merge(file_content, graph_def)

  graph = ops.Graph()
  with graph.as_default():
    _import_graph_def(graph_def, name="")
  inputs = _util.get_tensors_from_tensor_names(graph, input_arrays)
  outputs = _util.get_tensors_from_tensor_names(graph, output_arrays)

  def run_session(input_data):
    with _session.Session(graph=graph) as sess:
      return sess.run(outputs, dict(zip(inputs, input_data)))

  return run_session


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
  keras_model = keras.models.load_model(filename)
  return lambda input_data: [keras_model.predict(input_data)]


def compare_models(tflite_model,
                   tf_eval_func,
                   input_shapes_resize=None,
                   input_data=None,
                   input_data_range=None,
                   tolerance=5):
  """Compares TensorFlow and TFLite models.

  Unless the input data is provided, the models are compared with random data.

  Args:
    tflite_model: Serialized TensorFlow Lite model.
    tf_eval_func: Lambda function that takes in input data and outputs the
      results of the TensorFlow model ([np.ndarray data] : [np.ndarray result]).
    input_shapes_resize: A map where the key is the input tensor name and the
      value is the shape of the input tensor. This resize happens after model
      conversion, prior to calling allocate tensors. (default None)
    input_data: np.ndarray to pass into models during inference. (default None)
    input_data_range: A map where the key is the input tensor name and
      the value is a tuple (min_val, max_val) which specifies the value range of
      the corresponding input tensor. For example, '{'input1': (1, 5)}' means to
      generate a random value for tensor `input1` within range [1.0, 5.0)
      (half-inclusive). (default None)
    tolerance: Decimal place to check accuracy to. (default 5).
  """
  if input_data is None:
    input_data, _ = _generate_random_input_data(
        tflite_model=tflite_model,
        input_data_range=input_data_range,
        input_shapes_resize=input_shapes_resize)
  tf_results = tf_eval_func(input_data)
  tflite_results, _ = _evaluate_tflite_model(
      tflite_model, input_data, input_shapes_resize=input_shapes_resize)
  for tf_result, tflite_result in zip(tf_results, tflite_results):
    np.testing.assert_almost_equal(tf_result, tflite_result, tolerance)


def _compare_tf_tflite_results(tf_results,
                               tflite_results,
                               tflite_labels,
                               tolerance=5):
  """Compare the result of TF and TFLite model.

  Args:
    tf_results: results returned by the TF model.
    tflite_results: results returned by the TFLite model.
    tflite_labels: names of the output tensors in the TFlite model.
    tolerance: Decimal place to check accuracy to. (default 5).
  """
  # Convert the output TensorFlow results into an ordered list.
  if isinstance(tf_results, dict):
    if len(tf_results) == 1:
      tf_results = [tf_results[list(tf_results.keys())[0]]]
    else:
      tf_results = [tf_results[tflite_label] for tflite_label in tflite_labels]
  else:
    tf_results = [tf_results]

  for tf_result, tflite_result in zip(tf_results, tflite_results):
    np.testing.assert_almost_equal(tf_result, tflite_result, tolerance)


def compare_models_v2(tflite_model,
                      tf_eval_func,
                      input_data=None,
                      input_data_range=None,
                      tolerance=5):
  """Compares TensorFlow and TFLite models for TensorFlow 2.0.

  Unless the input data is provided, the models are compared with random data.
  Currently only 1 input and 1 output are supported by this function.

  Args:
    tflite_model: Serialized TensorFlow Lite model.
    tf_eval_func: Function to evaluate TensorFlow model. Either a lambda
      function that takes in input data and outputs the results or a TensorFlow
      ConcreteFunction.
    input_data: np.ndarray to pass into models during inference. (default None).
    input_data_range: A map where the key is the input tensor name and
      the value is a tuple (min_val, max_val) which specifies the value range of
      the corresponding input tensor. For example, '{'input1': (1, 5)}' means to
      generate a random value for tensor `input1` within range [1.0, 5.0)
      (half-inclusive). (default None)
    tolerance: Decimal place to check accuracy to. (default 5)
  """
  # Convert the input data into a map.
  if input_data is None:
    input_data, input_data_map = _generate_random_input_data(
        tflite_model=tflite_model, input_data_range=input_data_range)
  else:
    input_data_map = _get_input_data_map(tflite_model, input_data)
  input_data_func_map = {
      input_name: constant_op.constant(input_data)
      for input_name, input_data in input_data_map.items()
  }

  if len(input_data) > 1:
    tf_results = tf_eval_func(**input_data_func_map)
  else:
    tf_results = tf_eval_func(constant_op.constant(input_data[0]))
  tflite_results, tflite_labels = _evaluate_tflite_model(
      tflite_model, input_data)

  _compare_tf_tflite_results(tf_results, tflite_results, tflite_labels,
                             tolerance)


def compare_tflite_keras_models_v2(tflite_model,
                                   keras_model,
                                   input_data=None,
                                   input_data_range=None,
                                   tolerance=5,
                                   custom_op_registerers=None):
  """Similar to compare_models_v2 but accept Keras model.

  Unless the input data is provided, the models are compared with random data.
  Currently only 1 input and 1 output are supported by this function.

  Args:
    tflite_model: Serialized TensorFlow Lite model.
    keras_model: Keras model to evaluate.
    input_data: np.ndarray to pass into models during inference. (default None).
    input_data_range: A map where the key is the input tensor name and the value
      is a tuple (min_val, max_val) which specifies the value range of
      the corresponding input tensor. For example, '{'input1': (1, 5)}' means to
      generate a random value for tensor `input1` within range [1.0, 5.0)
      (half-inclusive). (default None)
    tolerance: Decimal place to check accuracy to. (default 5)
    custom_op_registerers: Op registerers for custom ops.
  """
  # Generate random input data if not provided.
  if input_data is None:
    input_data, _ = _generate_random_input_data(
        tflite_model=tflite_model,
        input_data_range=input_data_range,
        custom_op_registerers=custom_op_registerers)

  if len(input_data) > 1:
    tf_results = keras_model.predict(input_data)
  else:
    tf_results = keras_model.predict(input_data[0])
  tflite_results, tflite_labels = _evaluate_tflite_model(
      tflite_model, input_data, custom_op_registerers=custom_op_registerers)

  _compare_tf_tflite_results(tf_results, tflite_results, tflite_labels,
                             tolerance)


def compare_model_golden(tflite_model,
                         input_data,
                         golden_name,
                         update_golden=False,
                         tolerance=5):
  """Compares the output of a TFLite model against pre-existing golden values.

  Args:
    tflite_model: Serialized TensorFlow Lite model.
    input_data: np.ndarray to pass into models during inference.
    golden_name: Name of the file containing the (expected) golden values.
    update_golden: Whether to update the golden values with the model output
      instead of comparing against them. This should only be done when a change
      in TFLite warrants it.
    tolerance: Decimal place to check accuracy to. (default 5).
  """
  tflite_results, _ = _evaluate_tflite_model(tflite_model, input_data)
  golden_file = get_golden_filepath(golden_name)
  if update_golden:
    logging.warning(_GOLDENS_UPDATE_WARNING)
    logging.warning("Updating golden values in file %s.", golden_file)
    if not os.path.exists(golden_file):
      golden_relative_path = os.path.relpath(
          golden_file, _resource_loader.get_root_dir_with_all_resources())
      logging.warning(
          "Golden file not found. Manually create it first:\ntouch %r",
          golden_relative_path)

    with open(golden_file, "wb") as f:
      np.save(f, tflite_results, allow_pickle=False)
  else:
    golden_data = np.load(golden_file, allow_pickle=False)
    np.testing.assert_almost_equal(golden_data, tflite_results, tolerance)


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

  interpreter_float = _get_tflite_interpreter(tflite_model_float)
  interpreter_float.allocate_tensors()
  float_tensors = interpreter_float.get_tensor_details()

  # Convert and load the quantized model.
  converter = _lite.TFLiteConverter.from_frozen_graph(filename, input_arrays,
                                                      output_arrays,
                                                      input_shapes)
  tflite_model_quant = _convert(
      converter, post_training_quantize=True, **kwargs)

  interpreter_quant = _get_tflite_interpreter(tflite_model_quant)
  interpreter_quant.allocate_tensors()
  quant_tensors = interpreter_quant.get_tensor_details()
  quant_tensors_map = {
      tensor_detail["name"]: tensor_detail for tensor_detail in quant_tensors
  }
  quantized_tensors = {
      tensor_detail["name"]: tensor_detail
      for tensor_detail in quant_tensors
      if tensor_detail["quantization_parameters"]
  }

  # Check if weights are of different types in the float and quantized models.
  num_tensors_float = len(float_tensors)
  num_tensors_same_dtypes = sum(
      float_tensor["dtype"] == quant_tensors_map[float_tensor["name"]]["dtype"]
      for float_tensor in float_tensors)
  has_quant_tensor = num_tensors_float != num_tensors_same_dtypes

  # For the "flex" case, post_training_quantize should not alter the graph,
  # unless we are quantizing to float16.
  if ("target_ops" in kwargs and
      not kwargs.get("quantize_to_float16", False) and
      not kwargs.get("post_training_quantize_int8", False) and
      not kwargs.get("post_training_quantize_16x8", False) and
      set(kwargs["target_ops"]) == set([_lite.OpsSet.SELECT_TF_OPS])):
    if has_quant_tensor:
      raise ValueError("--post_training_quantize flag unexpectedly altered the "
                       "full Flex mode graph.")
  elif kwargs.get("post_training_quantize_int8", False):
    # Instead of using tensor names, we use the number of tensors which have
    # quantization parameters to verify the model is quantized.
    if not quantized_tensors:
      raise ValueError("--post_training_quantize flag was unable to quantize "
                       "the graph as expected in TFLite.")
  elif not has_quant_tensor:
    raise ValueError("--post_training_quantize flag was unable to quantize the "
                     "graph as expected in TFLite and mix-and-match mode.")


def test_frozen_graph(filename,
                      input_arrays,
                      output_arrays,
                      input_shapes=None,
                      input_shapes_resize=None,
                      input_data=None,
                      input_data_range=None,
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
    input_shapes_resize: A map where the key is the input tensor name and the
      value is the shape of the input tensor. This resize happens after model
      conversion, prior to calling allocate tensors. (default None)
    input_data: np.ndarray to pass into models during inference. (default None).
    input_data_range: A map where the key is the input tensor name and
      the value is a tuple (min_val, max_val) which specifies the value range of
      the corresponding input tensor. For example, '{'input1': (1, 5)}' means to
      generate a random value for tensor `input1` within range [1.0, 5.0)
      (half-inclusive). (default None)
    **kwargs: Additional arguments to be passed into the converter.
  """
  converter = _lite.TFLiteConverter.from_frozen_graph(
      filename, input_arrays, output_arrays, input_shapes)
  tflite_model = _convert(converter, **kwargs)

  tf_eval_func = evaluate_frozen_graph(filename, input_arrays, output_arrays)
  compare_models(
      tflite_model,
      tf_eval_func,
      input_shapes_resize=input_shapes_resize,
      input_data=input_data,
      input_data_range=input_data_range)


def test_saved_model(directory,
                     input_shapes=None,
                     tag_set=None,
                     signature_key=None,
                     input_data=None,
                     input_data_range=None,
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
    input_data: np.ndarray to pass into models during inference. (default None).
    input_data_range: A map where the key is the input tensor name and
      the value is a tuple (min_val, max_val) which specifies the value range of
      the corresponding input tensor. For example, '{'input1': (1, 5)}' means to
      generate a random value for tensor `input1` within range [1.0, 5.0)
      (half-inclusive). (default None)
    **kwargs: Additional arguments to be passed into the converter.
  """
  converter = _lite.TFLiteConverter.from_saved_model(
      directory,
      input_shapes=input_shapes,
      tag_set=tag_set,
      signature_key=signature_key)
  tflite_model = _convert(converter, **kwargs)

  # 5 decimal places by default
  tolerance = 5
  if kwargs.get("post_training_quantize_16x8", False):
    _check_model_quantized_to_16x8(tflite_model)
    # only 2 decimal places for full quantization
    tolerance = 2

  tf_eval_func = evaluate_saved_model(directory, tag_set, signature_key)
  compare_models(
      tflite_model,
      tf_eval_func,
      input_data=input_data,
      input_data_range=input_data_range,
      tolerance=tolerance)


def test_saved_model_v2(directory,
                        tag_set=None,
                        signature_key=None,
                        input_data=None,
                        input_data_range=None,
                        **kwargs):
  """Validates the TensorFlow SavedModel converts to a TFLite model.

  Converts the TensorFlow SavedModel to TFLite and checks the accuracy of the
  model on random data.

  Args:
    directory: SavedModel directory to convert.
    tag_set: Set of tags identifying the MetaGraphDef within the SavedModel to
      analyze. All tags in the tag set must be present.
    signature_key: Key identifying SignatureDef containing inputs and outputs.
    input_data: np.ndarray to pass into models during inference. (default None).
    input_data_range: A map where the key is the input tensor name and
      the value is a tuple (min_val, max_val) which specifies the value range of
      the corresponding input tensor. For example, '{'input1': (1, 5)}' means to
      generate a random value for tensor `input1` within range [1.0, 5.0)
      (half-inclusive). (default None)
    **kwargs: Additional arguments to be passed into the converter.
  """
  model = _load.load(directory, tags=tag_set)
  if not signature_key:
    signature_key = _signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
  concrete_func = model.signatures[signature_key]

  converter = _lite.TFLiteConverterV2.from_concrete_functions([concrete_func])
  tflite_model = _convert(converter, **kwargs)

  compare_models_v2(
      tflite_model,
      concrete_func,
      input_data=input_data,
      input_data_range=input_data_range)


def _test_conversion_quant_float16(converter,
                                   input_data,
                                   golden_name=None,
                                   update_golden=False,
                                   **kwargs):
  """Validates conversion with float16 quantization.

  Args:
    converter: TFLite converter instance for the model to convert.
    input_data: np.ndarray to pass into models during inference.
    golden_name: Optional golden values to compare the output of the model
      against.
    update_golden: Whether to update the golden values with the model output
      instead of comparing against them.
    **kwargs: Additional arguments to be passed into the converter.
  """
  tflite_model_float = _convert(converter, version=2, **kwargs)

  interpreter_float = _get_tflite_interpreter(tflite_model_float)
  interpreter_float.allocate_tensors()
  float_tensors = interpreter_float.get_tensor_details()

  tflite_model_quant = _convert(
      converter,
      version=2,
      post_training_quantize=True,
      quantize_to_float16=True,
      **kwargs)

  interpreter_quant = _get_tflite_interpreter(tflite_model_quant)
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

  if not has_quant_tensor:
    raise ValueError("--post_training_quantize flag was unable to quantize the "
                     "graph as expected.")

  if golden_name:
    compare_model_golden(tflite_model_quant, input_data, golden_name,
                         update_golden)


def test_saved_model_v2_quant_float16(directory,
                                      input_data,
                                      golden_name=None,
                                      update_golden=False,
                                      **kwargs):
  """Validates conversion of a saved model to TFLite with float16 quantization.

  Args:
    directory: SavedModel directory to convert.
    input_data: np.ndarray to pass into models during inference.
    golden_name: Optional golden values to compare the output of the model
      against.
    update_golden: Whether to update the golden values with the model output
      instead of comparing against them.
    **kwargs: Additional arguments to be passed into the converter.
  """
  converter = _lite.TFLiteConverterV2.from_saved_model(directory)
  _test_conversion_quant_float16(converter, input_data, golden_name,
                                 update_golden, **kwargs)


def test_frozen_graph_quant_float16(filename,
                                    input_arrays,
                                    output_arrays,
                                    input_data,
                                    input_shapes=None,
                                    golden_name=None,
                                    update_golden=False,
                                    **kwargs):
  """Validates conversion of a frozen graph to TFLite with float16 quantization.

  Args:
    filename: Full filepath of file containing frozen GraphDef.
    input_arrays: List of input tensors to freeze graph with.
    output_arrays: List of output tensors to freeze graph with.
    input_data: np.ndarray to pass into models during inference.
    input_shapes: Dict of strings representing input tensor names to list of
      integers representing input shapes (e.g., {"foo" : [1, 16, 16, 3]}).
      Automatically determined when input shapes is None (e.g., {"foo" : None}).
        (default None)
    golden_name: Optional golden values to compare the output of the model
      against.
    update_golden: Whether to update the golden values with the model output
      instead of comparing against them.
    **kwargs: Additional arguments to be passed into the converter.
  """
  converter = _lite.TFLiteConverter.from_frozen_graph(filename, input_arrays,
                                                      output_arrays,
                                                      input_shapes)
  _test_conversion_quant_float16(converter, input_data,
                                 golden_name, update_golden, **kwargs)


def test_keras_model(filename,
                     input_arrays=None,
                     input_shapes=None,
                     input_data=None,
                     input_data_range=None,
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
    input_data: np.ndarray to pass into models during inference. (default None).
    input_data_range: A map where the key is the input tensor name and
      the value is a tuple (min_val, max_val) which specifies the value range of
      the corresponding input tensor. For example, '{'input1': (1, 5)}' means to
      generate a random value for tensor `input1` within range [1.0, 5.0)
      (half-inclusive). (default None)
    **kwargs: Additional arguments to be passed into the converter.
  """
  converter = _lite.TFLiteConverter.from_keras_model_file(
      filename, input_arrays=input_arrays, input_shapes=input_shapes)
  tflite_model = _convert(converter, **kwargs)

  tf_eval_func = evaluate_keras_model(filename)
  compare_models(
      tflite_model,
      tf_eval_func,
      input_data=input_data,
      input_data_range=input_data_range)


def test_keras_model_v2(filename,
                        input_shapes=None,
                        input_data=None,
                        input_data_range=None,
                        **kwargs):
  """Validates the tf.keras model converts to a TFLite model.

  Converts the tf.keras model to TFLite and checks the accuracy of the model on
  random data.

  Args:
    filename: Full filepath of HDF5 file containing the tf.keras model.
    input_shapes: List of list of integers representing input shapes in the
      order of the tf.keras model's .input attribute (e.g., [[1, 16, 16, 3]]).
      (default None)
    input_data: np.ndarray to pass into models during inference. (default None).
    input_data_range: A map where the key is the input tensor name and
      the value is a tuple (min_val, max_val) which specifies the value range of
      the corresponding input tensor. For example, '{'input1': (1, 5)}' means to
      generate a random value for tensor `input1` within range [1.0, 5.0)
      (half-inclusive). (default None)
    **kwargs: Additional arguments to be passed into the converter.
  """
  keras_model = keras.models.load_model(filename)
  if input_shapes:
    for tensor, shape in zip(keras_model.inputs, input_shapes):
      tensor.set_shape(shape)

  converter = _lite.TFLiteConverterV2.from_keras_model(keras_model)
  tflite_model = _convert(converter, **kwargs)

  tf_eval_func = evaluate_keras_model(filename)
  compare_models_v2(
      tflite_model,
      tf_eval_func,
      input_data=input_data,
      input_data_range=input_data_range)
