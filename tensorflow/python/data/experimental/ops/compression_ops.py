# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Ops for compressing and uncompressing dataset elements."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.util import structure
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops


def compress(element):
  """Compress a dataset element.

  Args:
    element: A nested structure of types supported by Tensorflow.

  Returns:
    A variant tensor representing the compressed element. This variant can be
    passed to `uncompress` to get back the original element.
  """
  element_spec = structure.type_spec_from_value(element)
  tensor_list = structure.to_tensor_list(element_spec, element)
  return ged_ops.compress_element(tensor_list)


def uncompress(element, output_spec):
  """Uncompress a compressed dataset element.

  Args:
    element: A scalar variant tensor to uncompress. The element should have been
      created by calling `compress`.
    output_spec: A nested structure of `tf.TypeSpec` representing the type(s) of
      the uncompressed element.

  Returns:
    The uncompressed element.
  """
  flat_types = structure.get_flat_tensor_types(output_spec)
  flat_shapes = structure.get_flat_tensor_shapes(output_spec)
  tensor_list = ged_ops.uncompress_element(
      element, output_types=flat_types, output_shapes=flat_shapes)
  return structure.from_tensor_list(output_spec, tensor_list)
