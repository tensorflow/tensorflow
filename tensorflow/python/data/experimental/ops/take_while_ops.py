# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""take-while dataset transformation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


@deprecation.deprecated(None, "Use `tf.data.Dataset.take_while(...)")
@tf_export("data.experimental.take_while")
def take_while(predicate):
  """A transformation that stops dataset iteration based on a `predicate`.

  Args:
    predicate: A function that maps a nested structure of tensors (having shapes
      and types defined by `self.output_shapes` and `self.output_types`) to a
      scalar `tf.bool` tensor.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """

  def _apply_fn(dataset):
    return dataset.take_while(predicate=predicate)

  return _apply_fn
