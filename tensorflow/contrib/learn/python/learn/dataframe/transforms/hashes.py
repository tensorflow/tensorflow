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

"""Masks one `Series` based on the content of another `Series`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.learn.python.learn.dataframe import transform
from tensorflow.python.ops import string_ops


class HashFast(transform.Transform):
  """Perform a fast hash of a `Series`."""

  def __init__(self, num_buckets):
    """Initialize `CSVParser`.

    Args:
      num_buckets: The number of hash buckets to use.
    """
    # TODO(soergel): allow seed?
    super(HashFast, self).__init__()
    self._num_buckets = num_buckets

  @property
  def name(self):
    return "HashFast"

  @property
  def input_valency(self):
    return 1

  @property
  def _output_names(self):
    return "output",

  def _apply_transform(self, input_tensors, **kwargs):
    """Applies the transformation to the `transform_input`.

    Args:
      input_tensors: a list of Tensors representing the input to
        the Transform.
      **kwargs: additional keyword arguments, unused here.

    Returns:
        A namedtuple of Tensors representing the transformed output.
    """
    result = string_ops.string_to_hash_bucket_fast(input_tensors[0],
                                                   self._num_buckets,
                                                   name=None)
    # pylint: disable=not-callable
    return self.return_type(result)


