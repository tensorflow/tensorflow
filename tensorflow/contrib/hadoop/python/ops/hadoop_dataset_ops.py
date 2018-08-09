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

from tensorflow.contrib.hadoop.python.ops import hadoop_op_loader  # pylint: disable=unused-import
from tensorflow.contrib.hadoop.python.ops import gen_dataset_ops
from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape


class SequenceFileDataset(Dataset):
  """A Sequence File Dataset that reads the sequence file."""

  def __init__(self, filenames):
    """Create a `SequenceFileDataset`.

    `SequenceFileDataset` allows a user to read data from a hadoop sequence
    file. A sequence file consists of (key value) pairs sequentially. At
    the moment, `org.apache.hadoop.io.Text` is the only serialization type
    being supported, and there is no compression support.

    For example:

    ```python
    dataset = tf.contrib.hadoop.SequenceFileDataset("/foo/bar.seq")
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    # Prints the (key, value) pairs inside a hadoop sequence file.
    while True:
      try:
        print(sess.run(next_element))
      except tf.errors.OutOfRangeError:
        break
    ```

    Args:
      filenames: A `tf.string` tensor containing one or more filenames.
    """
    super(SequenceFileDataset, self).__init__()
    self._filenames = ops.convert_to_tensor(
        filenames, dtype=dtypes.string, name="filenames")

  def _as_variant_tensor(self):
    return gen_dataset_ops.sequence_file_dataset(
        self._filenames, nest.flatten(self.output_types))

  @property
  def output_classes(self):
    return ops.Tensor, ops.Tensor

  @property
  def output_shapes(self):
    return (tensor_shape.TensorShape([]), tensor_shape.TensorShape([]))

  @property
  def output_types(self):
    return dtypes.string, dtypes.string
