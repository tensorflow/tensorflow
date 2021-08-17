# pylint: disable=g-bad-file-header
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
r"""Removes parts of a graph that are only needed for training.

There are several common transformations that can be applied to GraphDefs
created to train a model, that help reduce the amount of computation needed when
the network is used only for inference. These include:

 - Removing training-only operations like checkpoint saving.

 - Stripping out parts of the graph that are never reached.

 - Removing debug operations like CheckNumerics.

 - Fusing a group of primitive ops for batch normalization to FusedBatchNorm op.

 - Folding batch normalization ops into the pre-calculated weights.

 - Fusing common operations into unified versions.

This script takes a frozen GraphDef file (where the weight variables have been
converted into constants by the freeze_graph script) and outputs a new GraphDef
with the optimizations applied.

An example of command-line usage is:

bazel build tensorflow/python/tools:optimize_for_inference && \
bazel-bin/tensorflow/python/tools/optimize_for_inference \
--input_graph=some_graph_def.pb \
--output_graph=/tmp/optimized_graph.pb \
--input_names=Mul \
--output_names=softmax

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import re

import numpy as np

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import flags as flags_lib
from tensorflow.python.platform import tf_logging
from tensorflow.python.tools import strip_unused_lib

flags = flags_lib
FLAGS = flags.FLAGS


# Support folding two types of batch norm ops:
# BatchNormWithGlobalNormalization and FusedBatchNorm.  The two types only
# differ in input order and attribute names, so we've collected their
# differences up front.
INPUT_ORDER = {
    # Order of inputs for BatchNormWithGlobalNormalization.
    "BatchNormWithGlobalNormalization": [
        "conv_op", "mean_op", "var_op", "beta_op", "gamma_op"
    ],
    # Order of inputs for FusedBatchNorm.
    "FusedBatchNorm": ["conv_op", "gamma_op", "beta_op", "mean_op", "var_op"],
    # Order of inputs for FusedBatchNormV3.
    "FusedBatchNormV3": ["conv_op", "gamma_op", "beta_op", "mean_op", "var_op"]
}
# Name of the attribute epsilon value is stored in.
EPSILON_ATTR = {
    "BatchNormWithGlobalNormalization": "variance_epsilon",
    "FusedBatchNorm": "epsilon",
    "FusedBatchNormV3": "epsilon"
}


def optimize_for_inference(input_graph_def, input_node_names, output_node_names,
                           placeholder_type_enum, toco_compatible=False):
  """Applies a series of inference optimizations on the input graph.

  Args:
    input_graph_def: A GraphDef containing a training model.
    input_node_names: A list of names of the nodes that are fed inputs during
      inference.
    output_node_names: A list of names of the nodes that produce the final
      results.
    placeholder_type_enum: The AttrValue enum for the placeholder data type, or
        a list that specifies one value per input node name.
    toco_compatible: Boolean, if True, only runs optimizations that result in
      TOCO compatible graph operations (default=False).

  Returns:
    An optimized version of the input graph.
  """
  ensure_graph_is_valid(input_graph_def)
  optimized_graph_def = input_graph_def
  optimized_graph_def = strip_unused_lib.strip_unused(
      optimized_graph_def, input_node_names, output_node_names,
      placeholder_type_enum)
  optimized_graph_def = graph_util.remove_training_nodes(
      optimized_graph_def, output_node_names)
  optimized_graph_def = fuse_decomposed_batch_norm(optimized_graph_def)
  optimized_graph_def = fold_batch_norms(optimized_graph_def)
  if not toco_compatible:
    optimized_graph_def = fuse_resize_and_conv(optimized_graph_def,
                                               output_node_names)
  ensure_graph_is_valid(optimized_graph_def)
  return optimized_graph_def


def ensure_graph_is_valid(graph_def):
  """Makes sure that the graph is internally consistent.

  Checks basic properties of the graph def and raises an exception if there are
  input references to missing nodes, duplicated names, or other logic errors.

  Args:
    graph_def: Definition of a graph to be checked.

  Raises:
    ValueError: If the graph is incorrectly constructed.
  """
  node_map = {}
  for node in graph_def.node:
    if node.name not in node_map:
      node_map[node.name] = node
    else:
      raise ValueError("Duplicate node names detected for ", node.name)
  for node in graph_def.node:
    for input_name in node.input:
      input_node_name = node_name_from_input(input_name)
      if input_node_name not in node_map:
        raise ValueError("Input for ", node.name, " not found: ", input_name)


def node_name_from_input(node_name):
  """Strips off ports and other decorations to get the underlying node name."""
  if node_name.startswith("^"):
    node_name = node_name[1:]
  m = re.search(r"(.*):\d+$", node_name)
  if m:
    node_name = m.group(1)
  return node_name


def node_from_map(node_map, name):
  """Pulls a node def from a dictionary for a given name.

  Args:
    node_map: Dictionary containing an entry indexed by name for every node.
    name: Identifies the node we want to find.

  Returns:
    NodeDef of the node with the given name.

  Raises:
    ValueError: If the node isn't present in the dictionary.
  """
  stripped_name = node_name_from_input(name)
  if stripped_name not in node_map:
    raise ValueError("No node named '%s' found in map." % name)
  return node_map[stripped_name]


def values_from_const(node_def):
  """Extracts the values from a const NodeDef as a numpy ndarray.

  Args:
    node_def: Const NodeDef that has the values we want to access.

  Returns:
    Numpy ndarray containing the values.

  Raises:
    ValueError: If the node isn't a Const.
  """
  if node_def.op != "Const":
    raise ValueError(
        "Can not extract constant value from a node that is not Const. Got:\n"
        f"{node_def}")
  input_tensor = node_def.attr["value"].tensor
  tensor_value = tensor_util.MakeNdarray(input_tensor)
  return tensor_value


# Whether to scale by gamma after normalization.
def scale_after_normalization(node):
  if node.op == "BatchNormWithGlobalNormalization":
    return node.attr["scale_after_normalization"].b
  return True


def fold_batch_norms(input_graph_def):
  """Removes batch normalization ops by folding them into convolutions.

  Batch normalization during training has multiple dynamic parameters that are
  updated, but once the graph is finalized these become constants. That means
  there's an opportunity to reduce the computations down to a scale and
  addition, rather than the more expensive multiple ops, and even bake the
  scaling into the convolution weights. This function identifies the typical
  pattern of batch normalization subgraphs, and performs the transformation to
  fold the computations down into a simpler form. It currently only supports
  batch normalization that's performed by the BatchNormWithGlobalNormalization
  FusedBatchNorm and FusedBatchNormV3 ops, and will need to be extended in the
  future to handle the newer style.

  Args:
    input_graph_def: A GraphDef containing a model.

  Returns:
    Modified graph with BN ops removed, and modified weights.

  Raises:
    ValueError: If the graph is badly formed with duplicate node names.
  """
  input_node_map = {}
  for node in input_graph_def.node:
    if node.name not in input_node_map:
      input_node_map[node.name] = node
    else:
      raise ValueError("Duplicate node names detected for ", node.name)

  nodes_to_skip = {}
  new_ops = []
  for node in input_graph_def.node:
    if (node.op not in ("BatchNormWithGlobalNormalization", "FusedBatchNorm",
                        "FusedBatchNormV3")):
      continue

    bias = None
    conv_op = node_from_map(input_node_map,
                            node.input[INPUT_ORDER[node.op].index("conv_op")])
    # There might be an Add/BiasAdd op between the conv and the batchnorm,
    # which we can fold into the mean param of the batchnorm.
    if conv_op.op in ["BiasAdd", "Add", "AddV2"]:
      add_op = conv_op
      # Follow the first input of the add to get to the conv.
      conv_op = node_from_map(input_node_map, add_op.input[0])
      bias = node_from_map(input_node_map, add_op.input[1])
      if conv_op.op not in ["Conv2D", "DepthwiseConv2dNative"]:
        # Follow the second input of the add to get to the conv.
        conv_op = node_from_map(input_node_map, add_op.input[1])
        bias = node_from_map(input_node_map, add_op.input[0])
    if bias and bias.op != "Const":
      tf_logging.warning("The bias %s after the conv %s was not a constant. "
                         "Maybe because freeze_graph wasn't "
                         "run first?" % (bias.name, conv_op.name))
      continue
    if conv_op.op not in ["Conv2D", "DepthwiseConv2dNative"]:
      tf_logging.warning("Didn't find expected Conv2D or DepthwiseConv2dNative"
                         " input to '%s'" % node.name)
      continue

    weights_op = node_from_map(input_node_map, conv_op.input[1])
    if weights_op.op != "Const":
      tf_logging.warning("Didn't find expected conv Constant input to '%s',"
                         " found %s instead. Maybe because freeze_graph wasn't"
                         " run first?" % (conv_op.name, weights_op))
      continue
    weights = values_from_const(weights_op)
    if conv_op.op == "Conv2D":
      channel_count = weights.shape[3]
    elif conv_op.op == "DepthwiseConv2dNative":
      channel_count = weights.shape[2] * weights.shape[3]

    mean_op = node_from_map(input_node_map,
                            node.input[INPUT_ORDER[node.op].index("mean_op")])
    if mean_op.op != "Const":
      tf_logging.warning("Didn't find expected mean Constant input to '%s',"
                         " found %s instead. Maybe because freeze_graph wasn't"
                         " run first?" % (node.name, mean_op))
      continue
    mean_value = values_from_const(mean_op)
    if bias is not None:
      # Adjust the mean of the batchnorm based on the add op in-between the conv
      # and the batchnorm.
      mean_value = mean_value - values_from_const(bias)
    if mean_value.shape != (channel_count,):
      tf_logging.warning("Incorrect shape for mean, found %s, expected %s,"
                         " for node %s" % (str(mean_value.shape), str(
                             (channel_count,)), node.name))
      continue

    var_op = node_from_map(input_node_map,
                           node.input[INPUT_ORDER[node.op].index("var_op")])
    if var_op.op != "Const":
      tf_logging.warning("Didn't find expected var Constant input to '%s',"
                         " found %s instead. Maybe because freeze_graph wasn't"
                         " run first?" % (node.name, var_op))
      continue
    var_value = values_from_const(var_op)
    if var_value.shape != (channel_count,):
      tf_logging.warning("Incorrect shape for var, found %s, expected %s,"
                         " for node %s" % (str(var_value.shape), str(
                             (channel_count,)), node.name))
      continue

    beta_op = node_from_map(input_node_map,
                            node.input[INPUT_ORDER[node.op].index("beta_op")])
    if beta_op.op != "Const":
      tf_logging.warning("Didn't find expected beta Constant input to '%s',"
                         " found %s instead. Maybe because freeze_graph wasn't"
                         " run first?" % (node.name, beta_op))
      continue
    beta_value = values_from_const(beta_op)
    if beta_value.shape != (channel_count,):
      tf_logging.warning("Incorrect shape for beta, found %s, expected %s,"
                         " for node %s" % (str(beta_value.shape), str(
                             (channel_count,)), node.name))
      continue

    gamma_op = node_from_map(input_node_map,
                             node.input[INPUT_ORDER[node.op].index("gamma_op")])
    if gamma_op.op != "Const":
      tf_logging.warning("Didn't find expected gamma Constant input to '%s',"
                         " found %s instead. Maybe because freeze_graph wasn't"
                         " run first?" % (node.name, gamma_op))
      continue
    gamma_value = values_from_const(gamma_op)
    if gamma_value.shape != (channel_count,):
      tf_logging.warning("Incorrect shape for gamma, found %s, expected %s,"
                         " for node %s" % (str(gamma_value.shape), str(
                             (channel_count,)), node.name))
      continue

    variance_epsilon_value = node.attr[EPSILON_ATTR[node.op]].f
    nodes_to_skip[node.name] = True
    nodes_to_skip[weights_op.name] = True
    nodes_to_skip[conv_op.name] = True
    if bias is not None:
      nodes_to_skip[add_op.name] = True

    if scale_after_normalization(node):
      scale_value = (
          (1.0 / np.vectorize(math.sqrt)(var_value + variance_epsilon_value)) *
          gamma_value)
    else:
      scale_value = (
          1.0 / np.vectorize(math.sqrt)(var_value + variance_epsilon_value))
    offset_value = (-mean_value * scale_value) + beta_value
    scaled_weights = np.copy(weights)
    it = np.nditer(
        scaled_weights, flags=["multi_index"], op_flags=["readwrite"])
    if conv_op.op == "Conv2D":
      while not it.finished:
        current_scale = scale_value[it.multi_index[3]]
        it[0] *= current_scale
        it.iternext()
    elif conv_op.op == "DepthwiseConv2dNative":
      channel_multiplier = weights.shape[3]
      while not it.finished:
        current_scale = scale_value[it.multi_index[2] * channel_multiplier +
                                    it.multi_index[3]]
        it[0] *= current_scale
        it.iternext()
    scaled_weights_op = node_def_pb2.NodeDef()
    scaled_weights_op.op = "Const"
    scaled_weights_op.name = conv_op.name + "_weights"
    scaled_weights_op.attr["dtype"].CopyFrom(weights_op.attr["dtype"])
    scaled_weights_op.attr["value"].CopyFrom(
        attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
            scaled_weights, weights.dtype.type, weights.shape)))
    # Replace the weights node with scaled weights node
    for i, weights_node in enumerate(conv_op.input):
      if weights_node == weights_op.name:
        conv_op.input[i] = scaled_weights_op.name

    new_conv_op = node_def_pb2.NodeDef()
    new_conv_op.CopyFrom(conv_op)
    offset_op = node_def_pb2.NodeDef()
    offset_op.op = "Const"
    offset_op.name = conv_op.name + "_bn_offset"
    offset_op.attr["dtype"].CopyFrom(mean_op.attr["dtype"])
    offset_op.attr["value"].CopyFrom(
        attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
            offset_value, mean_value.dtype.type, offset_value.shape)))
    bias_add_op = node_def_pb2.NodeDef()
    bias_add_op.op = "BiasAdd"
    bias_add_op.name = node.name
    bias_add_op.attr["T"].CopyFrom(conv_op.attr["T"])
    bias_add_op.attr["data_format"].CopyFrom(conv_op.attr["data_format"])
    bias_add_op.input.extend([new_conv_op.name, offset_op.name])
    new_ops.extend([scaled_weights_op, new_conv_op, offset_op, bias_add_op])

  result_graph_def = graph_pb2.GraphDef()
  for node in input_graph_def.node:
    if node.name in nodes_to_skip:
      continue
    new_node = node_def_pb2.NodeDef()
    new_node.CopyFrom(node)
    retained_input = []
    for input_node in new_node.input:
      if not input_node.startswith("^") or input_node[1:] not in nodes_to_skip:
        retained_input.append(input_node)
    new_node.input[:] = retained_input

    result_graph_def.node.extend([new_node])

  result_graph_def.node.extend(new_ops)
  result_graph_def.versions.CopyFrom(input_graph_def.versions)
  return result_graph_def


def fuse_resize_and_conv(input_graph_def, output_node_names):
  """Merges preceding resize and mirror pad ops into a specialized convolution.

  There's a common pattern of enlarging the input to a convolution using a
  resize operation, and also using MirrorPad to extend the boundaries to that
  zero edge pixels don't bleed inwards when convolving. This routine looks for
  that pattern of operations, and fuses them together into a Conv2DWithResizeOp.

  Args:
    input_graph_def: A GraphDef containing a model.
    output_node_names: A list of names of the nodes that produce the final
      results.

  Returns:
    Modified graph with resize and pad ops merged.

  Raises:
    ValueError: If the graph is badly formed with duplicate node names.
  """

  input_node_map = {}
  for node in input_graph_def.node:
    if node.name not in input_node_map:
      input_node_map[node.name] = node
    else:
      raise ValueError("Duplicate node names detected for ", node.name)

  node_reference_count = collections.defaultdict(int)
  for node in input_graph_def.node:
    for input_name in node.input:
      stripped_name = node_name_from_input(input_name)
      node_reference_count[stripped_name] += 1
  for output_name in output_node_names:
    node_reference_count[output_name] += 1

  new_ops = []
  for node in input_graph_def.node:

    if node.op != "Conv2D":
      continue
    conv_op = node

    input_op = node_from_map(input_node_map, conv_op.input[0])
    if input_op.op == "MirrorPad":
      mirror_pad_op = input_op
      resize_op = node_from_map(input_node_map, mirror_pad_op.input[0])
      if resize_op.op != "ResizeBilinear":
        resize_op = None
    else:
      mirror_pad_op = None
      if input_op.op == "ResizeBilinear":
        resize_op = input_op
      else:
        resize_op = None

    # There are no ops to be fused into the conv, so skip replacing this one.
    if not mirror_pad_op and not resize_op:
      continue

    # We're replacing this node, so make sure the old one is removed.
    node_reference_count[conv_op.name] = 0
    if mirror_pad_op:
      node_reference_count[mirror_pad_op.name] -= 1
    if resize_op:
      node_reference_count[resize_op.name] -= 1

    fused_conv_op = node_def_pb2.NodeDef()
    if resize_op:
      fused_conv_op.op = "FusedResizeAndPadConv2D"
    else:
      fused_conv_op.op = "FusedPadConv2D"
    fused_conv_op.name = conv_op.name
    if mirror_pad_op:
      mirror_paddings_name = mirror_pad_op.input[1]
      mirror_paddings_mode = mirror_pad_op.attr["mode"]
    else:
      # If there was no MirrorPad op, then create settings that make the padding
      # stage of the fused operation a no-op.
      paddings_op = node_def_pb2.NodeDef()
      paddings_op.op = "Const"
      paddings_op.name = conv_op.name + "_dummy_paddings"
      paddings_op.attr["dtype"].CopyFrom(
          attr_value_pb2.AttrValue(type=dtypes.int32.as_datatype_enum))
      paddings_op.attr["value"].CopyFrom(
          attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
              [0, 0, 0, 0, 0, 0, 0, 0], dtypes.int32, [4, 2])))
      new_ops.extend([paddings_op])
      mirror_paddings_name = paddings_op.name
      mirror_paddings_mode = attr_value_pb2.AttrValue(s=b"REFLECT")
    if resize_op:
      fused_conv_op.input.extend([
          resize_op.input[0], resize_op.input[1], mirror_paddings_name,
          conv_op.input[1]
      ])
      fused_conv_op.attr["resize_align_corners"].CopyFrom(
          resize_op.attr["align_corners"])
    else:
      fused_conv_op.input.extend(
          [mirror_pad_op.input[0], mirror_paddings_name, conv_op.input[1]])
    fused_conv_op.attr["T"].CopyFrom(conv_op.attr["T"])
    fused_conv_op.attr["mode"].CopyFrom(mirror_paddings_mode)
    fused_conv_op.attr["strides"].CopyFrom(conv_op.attr["strides"])
    fused_conv_op.attr["padding"].CopyFrom(conv_op.attr["padding"])
    new_ops.extend([fused_conv_op])

  result_graph_def = graph_pb2.GraphDef()
  for node in input_graph_def.node:
    if node_reference_count[node.name] < 1:
      continue
    new_node = node_def_pb2.NodeDef()
    new_node.CopyFrom(node)
    result_graph_def.node.extend([new_node])

  result_graph_def.node.extend(new_ops)
  return result_graph_def

def get_const_dim_count(node_def):
  """Get the number of dimensions for a Const node.

  Args:
    node_def: Const NodeDef.

  Returns:
    Number of dimensions for the Const node.
  """
  const_value = values_from_const(node_def)
  return const_value.ndim

def fuse_decomposed_batch_norm(input_graph_def):
  """Fuse individual ops in batch normalization to FusedBatchNorm.

  In some models, the batch normalizatin is performed via a group of individual
  ops instead of using single FusedBatchNorm op. This function identifies a
  pattern of batch normalization subgraph which is made of multiple ops and
  transforms the graph by replacing those individual ops with FusedBatchNorm op.
  This will provide the opportunity to further fold the FusedBatchNorm with
  convolution ops to reduce the computation steps during inference.
  This function currently recognizes batch normalization patterns described
  below, this could be extended if newer patterns are seen. Also, the fusion
  is only attempted if the input graph is in NHWC format or has no format set.

  Computation function:
    (X * multiplier) + (Beta - Mean * multiplier)
      where multiplier = rsqrt (Variance + Epsilon) * Gamma
                    OR = rsqrt (Variance + Epsilon) when Gamma is 1

  Subgraph:
  {"Add"
      {{"Mul"  // mul_0
          {{"*"},  // input to apply batchnorm
           {"Mul"  // mul_1, same op is used inside the Sub block
              {{"Rsqrt"
                  {"Add"
                      {{"Const"},  // Variance
                       {"Const"}  // Epsilon
                      }
                  }
                },  // end - Rsqrt
                {"Const"}  // Gamma
              }
            }  // end - mul_1
          }
       },  // end - mul_0
       {"Sub"
          {{"Const"},  // Beta
           {"Mul"  // mul_3
              {{"Const"},  // Mean
               {"Mul"  // same mul_1 op as in previous block
                  {{"Rsqrt"
                      {"Add"
                          {{"Const"},  // Variance
                           {"Const"}  // Epsilon
                          }
                      }
                   },  // end - Rsqrt
                   {"Const"}  // Gamma
                  }
                }  // end - mul_1
              }
            }  // end - mul_3
          }
        }  // end - Sub
      }
  }  // end - Add

  Subgraph pattern when gamma value is 1 and the gamma scaling Mul is skipped
  {"Add"
      {{"Mul"  // mul_0
          {{"*"},  // input to apply batchnorma
           {"Rsqrt"  // same Rsqrt op used in Sub block
              {"Add"
                 {{"Const"},  // Variance
                  {"Const"}  // Epsilon
                 }
              }
            }  // end - Rsqrt
          }
        },  // end - mul_0
        {"Sub"
          {{"Const"},  // Beta
           {"Mul"  // mul_1
              {{"Const"},  // Mean
               {"Rsqrt"  // same Rsqrt op as in previous mul_0 block
                  {"Add"
                    {{"Const"},  // Variance
                     {"Const"}  // Epsilon
                    }
                  }
                }  // end - Rsqrt
              }
           }  // end - mul_1
          }
        }  // end - Sub
      }
  }  // end - Add

  Args:
    input_graph_def: A GraphDef containing a model.

  Returns:
    Modified graph with individual ops that made up of batch normalization
    fused to FusedBatchNorm.

  Raises:
    ValueError: If the graph is badly formed with duplicate node names.
  """
  input_node_map = {}
  for node in input_graph_def.node:
    if node.name not in input_node_map:
      input_node_map[node.name] = node
    else:
      raise ValueError("Duplicate node names detected for ", node.name)

  # Check format and only proceed if graph is in NHWC or has no format set.
  data_format = None
  for node in input_graph_def.node:
    if "data_format" in node.attr.keys():
      data_format = node.attr["data_format"]
      if data_format is not None and data_format.s != b"NHWC":
        tf_logging.warn("%s in %s format, not candidate for batchnorm fusion."
                        % (node.name, data_format.s))
        return input_graph_def
    else:
      continue

  nodes_to_skip = {}
  new_ops = []
  for node in input_graph_def.node:
    if node.op != "Add":
      continue

    # Add (Mul, Sub) or Add (Sub, Mul)
    input0_op = node_from_map(input_node_map, node.input[0])
    input1_op = node_from_map(input_node_map, node.input[1])

    if input0_op.op == "Mul" and input1_op.op == "Sub":
      data_scale_mul_op = input0_op
      bias_mean_sub_op = input1_op
    elif input0_op.op == "Sub" and input1_op.op == "Mul":
      bias_mean_sub_op = input0_op
      data_scale_mul_op = input1_op
    else:
      continue

    # Mul (input, Mul)
    input_data_op = node_from_map(input_node_map, data_scale_mul_op.input[0])
    scale_op = node_from_map(input_node_map, data_scale_mul_op.input[1])

    if scale_op.op == "Rsqrt":
      gamma_op = None
      rsqrt_op = scale_op
    elif scale_op.op == "Mul":
      # Mul (Rsqrt, Constant_gamma)
      rsqrt_op = node_from_map(input_node_map, scale_op.input[0])
      gamma_op = node_from_map(input_node_map, scale_op.input[1])
      if rsqrt_op.op != "Rsqrt":
        continue
      if gamma_op.op != "Const" or get_const_dim_count(gamma_op) != 1:
        continue
    else:
      continue

    # Sub (Constant_beta, Mul)
    beta_op = node_from_map(input_node_map, bias_mean_sub_op.input[0])
    mean_scale_mul_op = node_from_map(input_node_map, bias_mean_sub_op.input[1])
    if mean_scale_mul_op.op != "Mul":
      continue
    if beta_op.op != "Const" or get_const_dim_count(beta_op) != 1:
      continue

    # Common scale applies to both input and running mean
    if scale_op != node_from_map(input_node_map,
                                 mean_scale_mul_op.input[1]):
      continue

    mean_op = node_from_map(input_node_map, mean_scale_mul_op.input[0])
    if mean_op.op != "Const" or get_const_dim_count(mean_op) != 1:
      continue

    # Add (Constant_variance, Constant_epsilon)
    variance_epsilon_add_op = node_from_map(input_node_map, rsqrt_op.input[0])
    if variance_epsilon_add_op.op != "Add":
      continue

    variance_op = node_from_map(input_node_map,
                                variance_epsilon_add_op.input[0])
    epsilon_op = node_from_map(input_node_map,
                               variance_epsilon_add_op.input[1])
    if epsilon_op.op != "Const" or get_const_dim_count(epsilon_op) != 0:
      continue
    if variance_op.op != "Const" or get_const_dim_count(variance_op) != 1:
      continue

    epsilon = values_from_const(epsilon_op)

    nodes_to_skip[node.name] = True
    nodes_to_skip[data_scale_mul_op.name] = True
    nodes_to_skip[bias_mean_sub_op.name] = True
    nodes_to_skip[mean_scale_mul_op.name] = True
    nodes_to_skip[scale_op.name] = True
    if scale_op.op != "Rsqrt":
      nodes_to_skip[rsqrt_op.name] = True
    nodes_to_skip[variance_epsilon_add_op.name] = True

    if gamma_op is None:
      gamma_op = node_def_pb2.NodeDef()
      gamma_op.op = "Const"
      # Assign name with same root of Rsqrt op's name plus "gamma"
      m = re.search(r"(.*)/(.*)", scale_op.name)
      if m:
        gamma_op.name = m.group(1) + "/gamma"
      else:
        gamma_op.name = scale_op.name + "/gamma"
      gamma_op.attr["dtype"].CopyFrom(beta_op.attr["dtype"])
      beta_value = values_from_const(beta_op)
      gamma_op.attr["value"].CopyFrom(
          attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
              1, beta_value.dtype.type, beta_value.shape,
              allow_broadcast=True)))
      new_ops.append(gamma_op)

    new_fused_batchnorm_op = node_def_pb2.NodeDef()
    new_fused_batchnorm_op.op = "FusedBatchNorm"
    new_fused_batchnorm_op.name = node.name
    new_fused_batchnorm_op.attr["T"].CopyFrom(node.attr["T"])
    new_fused_batchnorm_op.attr["is_training"].CopyFrom(
        attr_value_pb2.AttrValue(b=False))
    new_fused_batchnorm_op.attr["epsilon"].CopyFrom(
        attr_value_pb2.AttrValue(f=epsilon.tolist()))
    if data_format is not None:
      new_fused_batchnorm_op.attr["data_format"].CopyFrom(data_format)
    new_fused_batchnorm_op.input.extend([input_data_op.name, gamma_op.name,
                                         beta_op.name, mean_op.name,
                                         variance_op.name])

    new_ops.append(new_fused_batchnorm_op)

  result_graph_def = graph_pb2.GraphDef()
  for node in input_graph_def.node:
    if node.name in nodes_to_skip:
      continue
    new_node = node_def_pb2.NodeDef()
    new_node.CopyFrom(node)
    retained_input = []
    for input_node in new_node.input:
      if not input_node.startswith("^") or input_node[1:] not in nodes_to_skip:
        retained_input.append(input_node)
    new_node.input[:] = retained_input
    result_graph_def.node.append(new_node)

  result_graph_def.node.extend(new_ops)
  result_graph_def.versions.CopyFrom(input_graph_def.versions)
  return result_graph_def
