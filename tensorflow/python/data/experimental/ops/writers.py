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
"""Python wrappers for tf.data writers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import convert
from tensorflow.python.data.util import structure
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_experimental_dataset_ops
from tensorflow.python.util.tf_export import tf_export


@tf_export("data.experimental.TFRecordWriter")
class TFRecordWriter(object):
  """Writes data to a TFRecord file."""

  def __init__(self, filename, compression_type=None):
    self._filename = ops.convert_to_tensor(
        filename, dtypes.string, name="filename")
    self._compression_type = convert.optional_param_to_tensor(
        "compression_type",
        compression_type,
        argument_default="",
        argument_dtype=dtypes.string)

  def write(self, dataset):
    """Returns a `tf.Operation` to write a dataset to a file.

    Args:
      dataset: a `tf.data.Dataset` whose elements are to be written to a file

    Returns:
      A `tf.Operation` that, when run, writes contents of `dataset` to a file.
    """
    if not isinstance(dataset, dataset_ops.DatasetV2):
      raise TypeError("`dataset` must be a `tf.data.Dataset` object.")
    if not dataset_ops.get_structure(dataset).is_compatible_with(
        structure.TensorStructure(dtypes.string, [])):
      raise TypeError(
          "`dataset` must produce scalar `DT_STRING` tensors whereas it "
          "produces shape {0} and types {1}".format(
              dataset_ops.get_legacy_output_shapes(dataset),
              dataset_ops.get_legacy_output_types(dataset)))
    return gen_experimental_dataset_ops.experimental_dataset_to_tf_record(
        dataset._variant_tensor, self._filename, self._compression_type)  # pylint: disable=protected-access
