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
"""SequenceFile Dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.hadoop.python.ops import gen_dataset_ops
from tensorflow.contrib.hadoop.python.ops import hadoop_op_loader  # pylint: disable=unused-import
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import structure
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.util import deprecation


class SequenceFileDataset(dataset_ops.DatasetSource):
  """A Sequence File Dataset that reads the sequence file."""

  @deprecation.deprecated(
      None,
      "tf.contrib.hadoop will be removed in 2.0, the support for Apache Hadoop "
      "will continue to be provided through the tensorflow/io GitHub project.")
  def __init__(self, filenames):
    """Create a `SequenceFileDataset`.

    `SequenceFileDataset` allows a user to read data from a hadoop sequence
    file. A sequence file consists of (key value) pairs sequentially. At
    the moment, `org.apache.hadoop.io.Text` is the only serialization type
    being supported, and there is no compression support.

    For example:

    ```python
    tf.enable_eager_execution()

    dataset = tf.contrib.hadoop.SequenceFileDataset("/foo/bar.seq")
    # Prints the (key, value) pairs inside a hadoop sequence file.
    for key, value in dataset:
      print(key, value)
    ```

    Args:
      filenames: A `tf.string` tensor containing one or more filenames.
    """
    self._filenames = ops.convert_to_tensor(
        filenames, dtype=dtypes.string, name="filenames")
    variant_tensor = gen_dataset_ops.sequence_file_dataset(
        self._filenames, self._element_structure._flat_types)  # pylint: disable=protected-access
    super(SequenceFileDataset, self).__init__(variant_tensor)

  @property
  def _element_structure(self):
    return structure.NestedStructure(
        (structure.TensorStructure(dtypes.string, []),
         structure.TensorStructure(dtypes.string, [])))
