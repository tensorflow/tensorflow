# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Summary Operations."""
# pylint: disable=protected-access
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_logging_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_logging_ops import *

# pylint: enable=wildcard-import


def _Collect(val, collections, default_collections):
  if collections is None:
    collections = default_collections
  for key in collections:
    ops.add_to_collection(key, val)


def tensor_summary(display_name,  # pylint: disable=invalid-name
                   tensor,
                   description="",
                   labels=None,
                   collections=None,
                   name=None):
  # pylint: disable=line-too-long
  """Outputs a `Summary` protocol buffer with a serialized tensor.proto.

  The generated
  [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
  has one summary value containing the input tensor.

  Args:
    display_name: A name to associate with the data series. Will be used to
      organize output data and as a name in visualizers.
    tensor: A tensor of any type and shape to serialize.
    description: An optional long description of the data being output.
    labels: a list of strings used to specify how the data can be interpreted,
      for example:
      * `'encoding:image/jpg'` for a string tensor containing jpg images
      * `'encoding:proto/X/Y/foo.proto'` for a string tensor containing Foos
      * `'group:$groupName/$roleInGroup'` for a tensor that is related to
         other tensors that are all in a group. (e.g. bounding boxes and images)
    collections: Optional list of graph collections keys. The new summary op is
      added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
    name: An optional name for the generated node (optional).

  Returns:
    A scalar `Tensor` of type `string`. The serialized `Summary` protocol
    buffer.
  """
  # pylint: enable=line-too-long

  with ops.name_scope(name, "TensorSummary", [tensor]) as scope:
    val = gen_logging_ops._tensor_summary(
        display_name=display_name,
        tensor=tensor,
        description=description,
        labels=labels,
        name=scope)
    _Collect(val, collections, [ops.GraphKeys.SUMMARIES])
  return val


ops.NotDifferentiable("TensorSummary")


ops.RegisterShape("TensorSummary")(common_shapes.call_cpp_shape_fn)
