# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""TensorFlow Lite tooling helper functionality.

EXPERIMENTAL: APIs here are unstable and likely to change without notice.

@@TocoConverter
@@toco_convert
@@toco_convert_protos
@@Interpreter
@@OpHint
@@convert_op_hints_to_stubs
@@build_toco_convert_protos

@@FLOAT
@@QUANTIZED_UINT8
@@TFLITE
@@GRAPHVIZ_DOT

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six import PY3

from google.protobuf import text_format as _text_format
from google.protobuf.message import DecodeError
from tensorflow.contrib.lite.python import lite_constants as constants
from tensorflow.contrib.lite.python.convert import build_toco_convert_protos  # pylint: disable=unused-import
from tensorflow.contrib.lite.python.convert import tensor_name as _tensor_name
from tensorflow.contrib.lite.python.convert import toco_convert  # pylint: disable=unused-import
from tensorflow.contrib.lite.python.convert import toco_convert_graph_def as _toco_convert_graph_def
from tensorflow.contrib.lite.python.convert import toco_convert_impl as _toco_convert_impl
from tensorflow.contrib.lite.python.convert import toco_convert_protos  # pylint: disable=unused-import
from tensorflow.contrib.lite.python.convert_saved_model import freeze_saved_model as _freeze_saved_model
from tensorflow.contrib.lite.python.convert_saved_model import get_tensors_from_tensor_names as _get_tensors_from_tensor_names
from tensorflow.contrib.lite.python.convert_saved_model import set_tensor_shapes as _set_tensor_shapes
from tensorflow.contrib.lite.python.interpreter import Interpreter  # pylint: disable=unused-import
from tensorflow.contrib.lite.python.op_hint import convert_op_hints_to_stubs  # pylint: disable=unused-import
from tensorflow.contrib.lite.python.op_hint import OpHint  # pylint: disable=unused-import
from tensorflow.core.framework import graph_pb2 as _graph_pb2
from tensorflow.python import keras as _keras
from tensorflow.python.client import session as _session
from tensorflow.python.framework import graph_util as _tf_graph_util
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework.errors_impl import NotFoundError as _NotFoundError
from tensorflow.python.framework.importer import import_graph_def as _import_graph_def
from tensorflow.python.lib.io import file_io as _file_io
from tensorflow.python.saved_model import signature_constants as _signature_constants
from tensorflow.python.saved_model import tag_constants as _tag_constants


class TocoConverter(object):
  """Convert a TensorFlow model into `output_format` using TOCO.

  This is used to convert from a TensorFlow GraphDef or SavedModel into either a
  TFLite FlatBuffer or graph visualization.

  Attributes:

    inference_type: Target data type of real-number arrays in the output file.
      Must be `{FLOAT, QUANTIZED_UINT8}`.  (default FLOAT)
    inference_input_type: Target data type of real-number input arrays. Allows
      for a different type for input arrays in the case of quantization.
      Must be `{FLOAT, QUANTIZED_UINT8}`. (default `inference_type`)
    output_format: Output file format. Currently must be `{TFLITE,
      GRAPHVIZ_DOT}`. (default TFLITE)
    quantized_input_stats: Dict of strings representing input tensor names
      mapped to tuple of floats representing the mean and standard deviation
      of the training data (e.g., {"foo" : (0., 1.)}). Only need if
      `inference_input_type` is `QUANTIZED_UINT8`.
      real_input_value = (quantized_input_value - mean_value) / std_dev_value.
      (default {})
    default_ranges_stats: Tuple of integers representing (min, max) range values
      for all arrays without a specified range. Intended for experimenting with
      quantization via "dummy quantization". (default None)
    drop_control_dependency: Boolean indicating whether to drop control
      dependencies silently. This is due to TFLite not supporting control
      dependencies. (default True)
    reorder_across_fake_quant: Boolean indicating whether to reorder FakeQuant
      nodes in unexpected locations. Used when the location of the FakeQuant
      nodes is preventing graph transformations necessary to convert the graph.
      Results in a graph that differs from the quantized training graph,
      potentially causing differing arithmetic behavior. (default False)
    change_concat_input_ranges: Boolean to change behavior of min/max ranges for
      inputs and outputs of the concat operator for quantized models. Changes
      the ranges of concat operator overlap when true. (default False)
    allow_custom_ops: Boolean indicating whether to allow custom operations.
      When false any unknown operation is an error. When true, custom ops are
      created for any op that is unknown. The developer will need to provide
      these to the TensorFlow Lite runtime with a custom resolver.
      (default False)
    post_training_quantize: Boolean indicating whether to quantize the weights
      of the converted float model. Model size will be reduced and there will be
      latency improvements (at the cost of accuracy).
      (default False)
    dump_graphviz_dir: Full filepath of folder to dump the graphs at various
      stages of processing GraphViz .dot files. Preferred over
      --output_format=GRAPHVIZ_DOT in order to keep the requirements of the
      output file. (default None)
    dump_graphviz_video: Boolean indicating whether to dump the graph after
      every graph transformation. (default False)

  Example usage:

    ```python
    # Converting a GraphDef from session.
    converter = lite.TocoConverter.from_session(sess, in_tensors, out_tensors)
    tflite_model = converter.convert()
    open("converted_model.tflite", "wb").write(tflite_model)

    # Converting a GraphDef from file.
    converter = lite.TocoConverter.from_frozen_graph(
      graph_def_file, input_arrays, output_arrays)
    tflite_model = converter.convert()
    open("converted_model.tflite", "wb").write(tflite_model)

    # Converting a SavedModel.
    converter = lite.TocoConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()

    # Converting a tf.keras model.
    converter = lite.TocoConverter.from_keras_model_file(keras_model)
    tflite_model = converter.convert()
    ```
  """

  def __init__(self,
               graph_def,
               input_tensors,
               output_tensors,
               input_arrays_with_shape=None,
               output_arrays=None):
    """Constructor for TocoConverter.

    Args:

      graph_def: Frozen TensorFlow GraphDef.
      input_tensors: List of input tensors. Type and shape are computed using
        `foo.get_shape()` and `foo.dtype`.
      output_tensors: List of output tensors (only .name is used from this).
      input_arrays_with_shape: Tuple of strings representing input tensor names
        and list of integers representing input shapes
        (e.g., [("foo" : [1, 16, 16, 3])]). Use only when graph cannot be loaded
        into TensorFlow and when `input_tensors` and `output_tensors` are None.
        (default None)
      output_arrays: List of output tensors to freeze graph with. Use only when
        graph cannot be loaded into TensorFlow and when `input_tensors` and
        `output_tensors` are None. (default None)

    Raises:
      ValueError: Invalid arguments.
    """
    self._graph_def = graph_def
    self._input_tensors = input_tensors
    self._output_tensors = output_tensors
    self.inference_type = constants.FLOAT
    self.inference_input_type = None
    self.output_format = constants.TFLITE
    self.quantized_input_stats = {}
    self.default_ranges_stats = None
    self.drop_control_dependency = True
    self.reorder_across_fake_quant = False
    self.change_concat_input_ranges = False
    self.allow_custom_ops = False
    self.post_training_quantize = False
    self.dump_graphviz_dir = None
    self.dump_graphviz_video = False

    # Attributes are used by models that cannot be loaded into TensorFlow.
    if not self._has_valid_tensors():
      if not input_arrays_with_shape or not output_arrays:
        raise ValueError(
            "If input_tensors and output_tensors are None, both "
            "input_arrays_with_shape and output_arrays must be defined.")
      self._input_arrays_with_shape = input_arrays_with_shape
      self._output_arrays = output_arrays

  @classmethod
  def from_session(cls, sess, input_tensors, output_tensors):
    """Creates a TocoConverter class from a TensorFlow Session.

    Args:
      sess: TensorFlow Session.
      input_tensors: List of input tensors. Type and shape are computed using
        `foo.get_shape()` and `foo.dtype`.
      output_tensors: List of output tensors (only .name is used from this).

    Returns:
      TocoConverter class.
    """
    graph_def = _freeze_graph(sess, output_tensors)
    return cls(graph_def, input_tensors, output_tensors)

  @classmethod
  def from_frozen_graph(cls,
                        graph_def_file,
                        input_arrays,
                        output_arrays,
                        input_shapes=None):
    """Creates a TocoConverter class from a file containing a frozen GraphDef.

    Args:
      graph_def_file: Full filepath of file containing frozen GraphDef.
      input_arrays: List of input tensors to freeze graph with.
      output_arrays: List of output tensors to freeze graph with.
      input_shapes: Dict of strings representing input tensor names to list of
        integers representing input shapes (e.g., {"foo" : [1, 16, 16, 3]}).
        Automatically determined when input shapes is None (e.g., {"foo" :
        None}). (default None)

    Returns:
      TocoConverter class.

    Raises:
      IOError:
        File not found.
        Unable to parse input file.
      ValueError:
        The graph is not frozen.
        input_arrays or output_arrays contains an invalid tensor name.
        input_shapes is not correctly defined when required
    """
    with _ops.Graph().as_default():
      with _session.Session() as sess:
        # Read GraphDef from file.
        if not _file_io.file_exists(graph_def_file):
          raise IOError("File '{0}' does not exist.".format(graph_def_file))
        with _file_io.FileIO(graph_def_file, "rb") as f:
          file_content = f.read()

        try:
          graph_def = _graph_pb2.GraphDef()
          graph_def.ParseFromString(file_content)
        except (_text_format.ParseError, DecodeError):
          try:
            print("Ignore 'tcmalloc: large alloc' warnings.")

            if not isinstance(file_content, str):
              if PY3:
                file_content = file_content.decode("utf-8")
              else:
                file_content = file_content.encode("utf-8")
            graph_def = _graph_pb2.GraphDef()
            _text_format.Merge(file_content, graph_def)
          except (_text_format.ParseError, DecodeError):
            raise IOError(
                "Unable to parse input file '{}'.".format(graph_def_file))

        # Handles models with custom TFLite ops that cannot be resolved in
        # TensorFlow.
        load_model_in_session = True
        try:
          _import_graph_def(graph_def, name="")
        except _NotFoundError:
          load_model_in_session = False

        if load_model_in_session:
          # Check if graph is frozen.
          if not _is_frozen_graph(sess):
            raise ValueError("Please freeze the graph using freeze_graph.py.")

          # Get input and output tensors.
          input_tensors = _get_tensors_from_tensor_names(
              sess.graph, input_arrays)
          output_tensors = _get_tensors_from_tensor_names(
              sess.graph, output_arrays)
          _set_tensor_shapes(input_tensors, input_shapes)

          return cls(sess.graph_def, input_tensors, output_tensors)
        else:
          if not input_shapes:
            raise ValueError("input_shapes must be defined for this model.")
          if set(input_arrays) != set(input_shapes.keys()):
            raise ValueError("input_shapes must contain a value for each item "
                             "in input_array.")

          input_arrays_with_shape = [
              (name, input_shapes[name]) for name in input_arrays
          ]
          return cls(
              graph_def,
              input_tensors=None,
              output_tensors=None,
              input_arrays_with_shape=input_arrays_with_shape,
              output_arrays=output_arrays)

  @classmethod
  def from_saved_model(cls,
                       saved_model_dir,
                       input_arrays=None,
                       input_shapes=None,
                       output_arrays=None,
                       tag_set=None,
                       signature_key=None):
    """Creates a TocoConverter class from a SavedModel.

    Args:
      saved_model_dir: SavedModel directory to convert.
      input_arrays: List of input tensors to freeze graph with. Uses input
        arrays from SignatureDef when none are provided. (default None)
      input_shapes: Dict of strings representing input tensor names to list of
        integers representing input shapes (e.g., {"foo" : [1, 16, 16, 3]}).
        Automatically determined when input shapes is None (e.g., {"foo" :
        None}). (default None)
      output_arrays: List of output tensors to freeze graph with. Uses output
        arrays from SignatureDef when none are provided. (default None)
      tag_set: Set of tags identifying the MetaGraphDef within the SavedModel to
        analyze. All tags in the tag set must be present. (default set("serve"))
      signature_key: Key identifying SignatureDef containing inputs and outputs.
        (default DEFAULT_SERVING_SIGNATURE_DEF_KEY)

    Returns:
      TocoConverter class.
    """
    if tag_set is None:
      tag_set = set([_tag_constants.SERVING])
    if signature_key is None:
      signature_key = _signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

    result = _freeze_saved_model(saved_model_dir, input_arrays, input_shapes,
                                 output_arrays, tag_set, signature_key)
    return cls(
        graph_def=result[0], input_tensors=result[1], output_tensors=result[2])

  @classmethod
  def from_keras_model_file(cls,
                            model_file,
                            input_arrays=None,
                            input_shapes=None,
                            output_arrays=None):
    """Creates a TocoConverter class from a tf.keras model file.

    Args:
      model_file: Full filepath of HDF5 file containing the tf.keras model.
      input_arrays: List of input tensors to freeze graph with. Uses input
        arrays from SignatureDef when none are provided. (default None)
      input_shapes: Dict of strings representing input tensor names to list of
        integers representing input shapes (e.g., {"foo" : [1, 16, 16, 3]}).
        Automatically determined when input shapes is None (e.g., {"foo" :
        None}). (default None)
      output_arrays: List of output tensors to freeze graph with. Uses output
        arrays from SignatureDef when none are provided. (default None)

    Returns:
      TocoConverter class.
    """
    _keras.backend.clear_session()
    _keras.backend.set_learning_phase(False)
    keras_model = _keras.models.load_model(model_file)
    sess = _keras.backend.get_session()

    # Get input and output tensors.
    if input_arrays:
      input_tensors = _get_tensors_from_tensor_names(sess.graph, input_arrays)
    else:
      input_tensors = keras_model.inputs

    if output_arrays:
      output_tensors = _get_tensors_from_tensor_names(sess.graph, output_arrays)
    else:
      output_tensors = keras_model.outputs
    _set_tensor_shapes(input_tensors, input_shapes)

    graph_def = _freeze_graph(sess, output_tensors)
    return cls(graph_def, input_tensors, output_tensors)

  def convert(self):
    """Converts a TensorFlow GraphDef based on instance variables.

    Returns:
      The converted data in serialized format. Either a TFLite Flatbuffer or a
      Graphviz graph depending on value in `output_format`.

    Raises:
      ValueError:
        Input shape is not specified.
        None value for dimension in input_tensor.
    """
    # Checks dimensions in input tensor.
    if self._has_valid_tensors():
      for tensor in self._input_tensors:
        if not tensor.get_shape():
          raise ValueError("Provide an input shape for input array "
                           "'{0}'.".format(_tensor_name(tensor)))
        shape = tensor.get_shape().as_list()
        if None in shape[1:]:
          raise ValueError(
              "None is only supported in the 1st dimension. Tensor '{0}' has "
              "invalid shape '{1}'.".format(_tensor_name(tensor), shape))
        elif shape[0] is None:
          self._set_batch_size(batch_size=1)

    # Get quantization stats. Ensures there is one stat per name if the stats
    # are specified.
    if self.quantized_input_stats:
      quantized_stats = []
      invalid_stats = []
      for name in self.get_input_arrays():
        if name in self.quantized_input_stats:
          quantized_stats.append(self.quantized_input_stats[name])
        else:
          invalid_stats.append(name)

      if invalid_stats:
        raise ValueError("Quantization input stats are not available for input "
                         "tensors '{0}'.".format(",".join(invalid_stats)))
    else:
      quantized_stats = None

    converter_kwargs = {
        "inference_type": self.inference_type,
        "inference_input_type": self.inference_input_type,
        "input_format": constants.TENSORFLOW_GRAPHDEF,
        "output_format": self.output_format,
        "quantized_input_stats": quantized_stats,
        "default_ranges_stats": self.default_ranges_stats,
        "drop_control_dependency": self.drop_control_dependency,
        "reorder_across_fake_quant": self.reorder_across_fake_quant,
        "change_concat_input_ranges": self.change_concat_input_ranges,
        "allow_custom_ops": self.allow_custom_ops,
        "post_training_quantize": self.post_training_quantize,
        "dump_graphviz_dir": self.dump_graphviz_dir,
        "dump_graphviz_video": self.dump_graphviz_video
    }

    # Converts model.
    if self._has_valid_tensors():
      result = _toco_convert_impl(
          input_data=self._graph_def,
          input_tensors=self._input_tensors,
          output_tensors=self._output_tensors,
          **converter_kwargs)
    else:
      result = _toco_convert_graph_def(
          input_data=self._graph_def,
          input_arrays_with_shape=self._input_arrays_with_shape,
          output_arrays=self._output_arrays,
          **converter_kwargs)
    return result

  def get_input_arrays(self):
    """Returns a list of the names of the input tensors.

    Returns:
      List of strings.
    """
    if self._has_valid_tensors():
      return [_tensor_name(tensor) for tensor in self._input_tensors]
    else:
      return [name for name, _ in self._input_arrays_with_shape]

  def _has_valid_tensors(self):
    """Checks if the input and output tensors have been initialized.

    Returns:
      Bool.
    """
    return self._input_tensors and self._output_tensors

  def _set_batch_size(self, batch_size):
    """Sets the first dimension of the input tensor to `batch_size`.

    Args:
      batch_size: Batch size for the model. Replaces the first dimension of an
        input size array if undefined. (default 1)

    Raises:
      ValueError: input_tensor is not defined.
    """
    if not self._has_valid_tensors():
      raise ValueError("The batch size cannot be set for this model. Please "
                       "use input_shapes parameter.")

    for tensor in self._input_tensors:
      shape = tensor.get_shape().as_list()
      shape[0] = batch_size
      tensor.set_shape(shape)


def _is_frozen_graph(sess):
  """Determines if the graph is frozen.

  Determines if a graph has previously been frozen by checking for any
  operations of type Variable*. If variables are found, the graph is not frozen.

  Args:
    sess: TensorFlow Session.

  Returns:
    Bool.
  """
  for op in sess.graph.get_operations():
    if op.type.startswith("Variable") or op.type.endswith("VariableOp"):
      return False
  return True


def _freeze_graph(sess, output_tensors):
  """Returns a frozen GraphDef.

  Freezes a graph with Variables in it. Otherwise the existing GraphDef is
  returned.

  Args:
    sess: TensorFlow Session.
    output_tensors: List of output tensors (only .name is used from this).

  Returns:
    Frozen GraphDef.
  """
  if not _is_frozen_graph(sess):
    output_arrays = [_tensor_name(tensor) for tensor in output_tensors]
    return _tf_graph_util.convert_variables_to_constants(
        sess, sess.graph_def, output_arrays)
  else:
    return sess.graph_def
