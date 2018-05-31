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

@@FLOAT
@@QUANTIZED_UINT8
@@TFLITE
@@GRAPHVIZ_DOT

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.lite.python import lite_constants as constants
from tensorflow.contrib.lite.python.convert import tensor_name
from tensorflow.contrib.lite.python.convert import toco_convert
from tensorflow.contrib.lite.python.convert import toco_convert_protos  # pylint: disable=unused-import
from tensorflow.contrib.lite.python.convert_saved_model import freeze_saved_model
from tensorflow.contrib.lite.python.interpreter import Interpreter  # pylint: disable=unused-import
from tensorflow.contrib.lite.python.op_hint import convert_op_hints_to_stubs  # pylint: disable=unused-import
from tensorflow.contrib.lite.python.op_hint import OpHint  # pylint: disable=unused-import
from tensorflow.python.framework import graph_util as tf_graph_util
from tensorflow.python.ops.variables import global_variables_initializer
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants


class TocoConverter(object):
  """Convert a TensorFlow model into `output_format` using TOCO.

  This is used to convert from a TensorFlow GraphDef or SavedModel into either a
  TFLite FlatBuffer or graph visualization.

  Attributes:

    inference_type: Currently must be `{FLOAT, QUANTIZED_UINT8}`.
      (default FLOAT)
    output_format: Type of data to write (currently must be TFLITE or
      GRAPHVIZ_DOT). (default TFLITE)
    quantized_input_stats: The mean and std deviation of training data for each
      input tensor. Only needed if `inference_type` is `QUANTIZED_UINT8`.
      (default None)
    drop_control_dependency: Boolean indicating whether to drop control
      dependencies silently. This is due to TFLite not supporting control
      dependencies. (default True)
    allow_custom_ops: Boolean indicating whether to allow custom operations.
      (default False)

  Example usage:

    # Converting a frozen graph.
    converter = lite.TocoConverter.from_session(sess, in_tensors, out_tensors)
    tflite_model = converter.convert()
    open("converted_model.tflite", "wb").write(tflite_model)

    # Converting a SavedModel.
    converter = lite.TocoConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
  """

  def __init__(self, graph_def, input_tensors, output_tensors):
    """Constructor for TocoConverter.

    Args:

      graph_def: TensorFlow GraphDef.
      input_tensors: List of input tensors. Type and shape are computed using
        `foo.get_shape()` and `foo.dtype`.
      output_tensors: List of output tensors (only .name is used from this).
    """
    self._graph_def = graph_def
    self._input_tensors = input_tensors
    self._output_tensors = output_tensors
    self.inference_type = constants.FLOAT
    self.output_format = constants.TFLITE
    self.quantized_input_stats = None
    self.drop_control_dependency = True
    self.allow_custom_ops = False

  @classmethod
  def from_session(cls,
                   sess,
                   input_tensors,
                   output_tensors,
                   freeze_variables=False):
    """Creates a TocoConverter class from a TensorFlow Session.

    Args:
      sess: TensorFlow Session.
      input_tensors: List of input tensors. Type and shape are computed using
        `foo.get_shape()` and `foo.dtype`.
      output_tensors: List of output tensors (only .name is used from this).
      freeze_variables: Boolean indicating whether the variables need to be
        converted into constants via the freeze_graph.py script.
        (default False)

    Returns:
      TocoConverter class.
    """

    # Get GraphDef.
    if freeze_variables:
      sess.run(global_variables_initializer())
      output_arrays = [tensor_name(tensor) for tensor in output_tensors]
      graph_def = tf_graph_util.convert_variables_to_constants(
          sess, sess.graph_def, output_arrays)
    else:
      graph_def = sess.graph_def

    # Create TocoConverter class.
    return cls(graph_def, input_tensors, output_tensors)

  @classmethod
  def from_saved_model(
      cls,
      saved_model_dir,
      input_arrays=None,
      input_shapes=None,
      output_arrays=None,
      tag_set=None,
      signature_key=signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY):
    """Creates a TocoConverter class from a SavedModel.

    Args:
      saved_model_dir: SavedModel directory to convert.
      input_arrays: List of input tensors to freeze graph with. Uses input
        arrays from SignatureDef when none are provided. (default None)
      input_shapes: Map of strings representing input tensor names to list of
        integers representing input shapes (e.g., {"foo": : [1, 16, 16, 3]}).
        Automatically determined when input shapes is None (e.g., {"foo" :
        None}). (default None)
      output_arrays: List of output tensors to freeze graph with. Uses output
        arrays from SignatureDef when none are provided. (default None)
      tag_set: Set of tags identifying the MetaGraphDef within the SavedModel to
        analyze. All tags in the tag set must be present. (default "serve")
      signature_key: Key identifying SignatureDef containing inputs and outputs.

    Returns:
      TocoConverter class.
    """
    if tag_set is None:
      tag_set = set([tag_constants.SERVING])

    result = freeze_saved_model(saved_model_dir, input_arrays, input_shapes,
                                output_arrays, tag_set, signature_key)
    return cls(
        graph_def=result[0], input_tensors=result[1], output_tensors=result[2])

  def convert(self):
    """Converts a TensorFlow GraphDef based on instance variables.

    Returns:
      The converted data in serialized format. Either a TFLite Flatbuffer or a
      Graphviz graph depending on value in `output_format`.

    Raises:
      ValueError:
        None value for dimension in input_tensor.
    """
    # Checks dimensions in input tensor.
    for tensor in self._input_tensors:
      shape = tensor.get_shape().as_list()
      if None in shape[1:]:
        raise ValueError(
            "None is only supported in the 1st dimension. Tensor '{0}' has "
            "invalid shape '{1}'.".format(tensor.name, shape))
      elif shape[0] is None:
        self._set_batch_size(batch_size=1)

    # Converts model.
    result = toco_convert(
        input_data=self._graph_def,
        input_tensors=self._input_tensors,
        output_tensors=self._output_tensors,
        inference_type=self.inference_type,
        input_format=constants.TENSORFLOW_GRAPHDEF,
        output_format=self.output_format,
        quantized_input_stats=self.quantized_input_stats,
        drop_control_dependency=self.drop_control_dependency)
    return result

  def _set_batch_size(self, batch_size):
    """Sets the first dimension of the input tensor to `batch_size`.

    Args:
      batch_size: Batch size for the model. Replaces the first dimension of an
        input size array if undefined. (default 1)
    """
    for tensor in self._input_tensors:
      shape = tensor.get_shape().as_list()
      shape[0] = batch_size
      tensor.set_shape(shape)
