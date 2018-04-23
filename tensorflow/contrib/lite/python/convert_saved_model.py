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
r"""TensorFlow Lite flatbuffer generation from saved_models.

Example:

bazel run third_party/tensorflow/contrib/lite/python:convert_saved_model -- \
  --saved_model_dir=/tmp/test_saved_model/1519865537 \
  --output_tflite=/tmp/test.lite

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.lite.python import lite
from tensorflow.contrib.saved_model.python.saved_model import reader
from tensorflow.contrib.saved_model.python.saved_model import signature_def_utils
from tensorflow.core.framework import types_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import graph_util as tf_graph_util
from tensorflow.python.framework import ops
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

flags.DEFINE_string("saved_model_dir", "", "Saved model directory to convert.")
flags.DEFINE_string("output_tflite", None, "File path to write flatbuffer.")
flags.DEFINE_string("output_arrays", None,
                    "List of output tensor names, the default value is None, "
                    "which means the conversion will keep all outputs.")
flags.DEFINE_integer("batch_size", 1,
                     "If input tensor shape has None at first dimension, "
                     "e.g. (None,224,224,3), replace None with batch_size.")
flags.DEFINE_string("tag_set", tag_constants.SERVING,
                    "Group of tag(s) of the MetaGraphDef in the saved_model, "
                    "in string format, separated by ','. For tag-set contains "
                    "multiple tags, all tags must be passed in.")
flags.DEFINE_string("signature_key",
                    signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
                    "This is signature key to extract inputs, outputs.")


def log_tensor_details(tensor_info):
  """Log tensor details: name, shape, and type."""
  for key in tensor_info:
    val = tensor_info[key]
    dtype = types_pb2.DataType.Name(val.dtype)
    if val.tensor_shape.unknown_rank:
      shape = "unknown_rank"
    else:
      dims = [str(dim.size) for dim in val.tensor_shape.dim]
      shape = "({})".format(", ".join(dims))

    logging.info("Tensor's key in saved_model's tensor_map: %s", key)
    logging.info(" tensor name: %s, shape: %s, type: %s", val.name, shape,
                 dtype)


def get_meta_graph_def(saved_model_dir, tag_set):
  """Validate saved_model and extract MetaGraphDef.

  Args:
    saved_model_dir: saved_model path to convert.
    tag_set: Set of tag(s) of the MetaGraphDef to load.

  Returns:
    The meta_graph_def used for tflite conversion.

  Raises:
    ValueError: No valid MetaGraphDef for given tag_set.
  """
  saved_model = reader.read_saved_model(saved_model_dir)
  tag_sets = []
  result_meta_graph_def = None
  for meta_graph_def in saved_model.meta_graphs:
    meta_graph_tag_set = set(meta_graph_def.meta_info_def.tags)
    tag_sets.append(meta_graph_tag_set)
    if meta_graph_tag_set == tag_set:
      result_meta_graph_def = meta_graph_def
  logging.info("The given saved_model contains the following tags: %s",
               tag_sets)
  if result_meta_graph_def is not None:
    return result_meta_graph_def
  else:
    raise ValueError("No valid MetaGraphDef for this tag_set '{}'. Possible "
                     "values are '{}'. ".format(tag_set, tag_sets))


def get_signature_def(meta_graph, signature_key):
  """Get the signature def from meta_graph with given signature_key.

  Args:
    meta_graph: meta_graph_def.
    signature_key: signature_def in the meta_graph_def.

  Returns:
    The signature_def used for tflite conversion.

  Raises:
    ValueError: Given signature_key is not valid for this meta_graph.
  """
  signature_def_map = meta_graph.signature_def
  signature_def_keys = set(signature_def_map.keys())
  logging.info(
      "The given saved_model MetaGraphDef contains SignatureDefs with the "
      "following keys: %s", signature_def_keys)
  if signature_key not in signature_def_keys:
    raise ValueError("No '{}' in the saved_model\'s SignatureDefs. Possible "
                     "values are '{}'. ".format(signature_key,
                                                signature_def_keys))
  signature_def = signature_def_utils.get_signature_def_by_key(
      meta_graph, signature_key)
  return signature_def


def get_inputs_outputs(signature_def):
  """Get inputs and outputs from signature def.

  Args:
    signature_def: signatuer def in the meta_graph_def for conversion.

  Returns:
    The inputs and outputs in the graph for conversion.
  """
  inputs_tensor_info = signature_def.inputs
  outputs_tensor_info = signature_def.outputs
  logging.info("input tensors info: ")
  log_tensor_details(inputs_tensor_info)
  logging.info("output tensors info: ")
  log_tensor_details(outputs_tensor_info)

  def gather_names(tensor_info):
    return [tensor_info[key].name for key in tensor_info]

  inputs = gather_names(inputs_tensor_info)
  outputs = gather_names(outputs_tensor_info)
  return inputs, outputs


def convert(saved_model_dir,
            output_tflite=None,
            output_arrays=None,
            tag_set=None,
            signature_key=signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
            batch_size=1):
  """Convert a saved_model to tflite flatbuffer.

  Args:
    saved_model_dir: Saved model directory to convert.
    output_tflite: File path to write result flatbuffer.
    output_arrays: List of output tensor names, the default value is None, which
      means conversion keeps all output tensors. This is also used to filter
      tensors that are from Op currently not supported in tflite, e.g., Argmax).
    tag_set: This is the set of tags to get meta_graph_def in saved_model.
    signature_key: This is the signature key to extract inputs, outputs.
    batch_size: If input tensor shape has None at first dimension,
      e.g. (None,224,224,3), replace None with batch_size.

  Returns:
    The converted data. For example if tflite was the destination, then
    this will be a tflite flatbuffer in a bytes array.

  Raises:
    ValueError: If tag_set does not indicate any meta_graph_def in saved_model,
      or signature_key is not in relevant meta_graph_def,
      or input shape has None beyond 1st dimension, e.g., (1,None, None, 3),
      or given output_arrays are not valid causing empty outputs.
  """
  if tag_set is None:
    tag_set = set([tag_constants.SERVING])

  meta_graph = get_meta_graph_def(saved_model_dir, tag_set)
  signature_def = get_signature_def(meta_graph, signature_key)
  inputs, outputs = get_inputs_outputs(signature_def)

  graph = ops.Graph()
  with session.Session(graph=graph) as sess:

    loader.load(sess, meta_graph.meta_info_def.tags, saved_model_dir)

    in_tensors = [graph.get_tensor_by_name(input_) for input_ in inputs]

    # Users can use output_arrays to filter output tensors for conversion.
    # If output_arrays is None, we keep all output tensors. In future, we may
    # use tflite supported Op list and check whether op is custom Op to
    # automatically filter output arrays.
    # TODO(zhixianyan): Use tflite supported Op list to filter outputs.
    if output_arrays is not None:
      output_arrays = output_arrays.split(",")
      out_tensors = [
          graph.get_tensor_by_name(output)
          for output in outputs
          if output.split(":")[0] in output_arrays
      ]
    else:
      out_tensors = [graph.get_tensor_by_name(output) for output in outputs]

    output_names = [node.split(":")[0] for node in outputs]

    if not out_tensors:
      raise ValueError(
          "No valid output tensors for '{}', possible values are '{}'".format(
              output_arrays, output_names))

    frozen_graph_def = tf_graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), output_names)

    # Toco requires fully defined tensor shape, for input tensor with None in
    # their shape, e.g., (None, 224, 224, 3), we need to replace first None with
    # a given batch size. For shape with more None, e.g. (None, None, None, 3),
    # still be able to replace and convert, but require further investigation.
    # TODO(zhixianyan): Add supports for input tensor with more None in shape.
    for i in range(len(in_tensors)):
      shape = in_tensors[i].get_shape().as_list()
      if shape[0] is None:
        shape[0] = batch_size
      if None in shape[1:]:
        raise ValueError(
            "Only support None shape at 1st dim as batch_size. But tensor "
            "'{}' 's shape '{}' has None at other dimension. ".format(
                inputs[i], shape))
      in_tensors[i].set_shape(shape)

    result = lite.toco_convert(frozen_graph_def, in_tensors, out_tensors)

    if output_tflite is not None:
      with gfile.Open(output_tflite, "wb") as f:
        f.write(result)
      logging.info("Successfully converted to: %s", output_tflite)

    return result


def main(_):
  convert(
      saved_model_dir=flags.FLAGS.saved_model_dir,
      output_tflite=flags.FLAGS.output_tflite,
      output_arrays=flags.FLAGS.output_arrays,
      batch_size=flags.FLAGS.batch_size,
      tag_set=set(flags.FLAGS.tag_set.split(",")),
      signature_key=flags.FLAGS.signature_key)


if __name__ == "__main__":
  app.run(main)
