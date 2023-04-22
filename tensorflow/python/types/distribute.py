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
"""Types specific to tf.distribute."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# TODO(mdan, anjalisridhar): Decide the location of this file.


class Iterable(object):
  """Interface for distributed objects that admit iteration/reduction."""

  def __iter__(self):
    pass

  # TODO(mdan): Describe this contract.
  def reduce(self, initial_state, reduce_func):
    """Reduces this iterable object to a single element.

    The transformation calls `reduce_func` successively on each element.
    The `initial_state` argument is used for the initial state and the final
    state is returned as the result.

    Args:
      initial_state: An element representing the initial state of the
        reduction.
      reduce_func: A function that maps `(old_state, input_element)` to
        `new_state`. The structure of `new_state` must match the structure of
        `old_state`. For the first element, `old_state` is `initial_state`.

    Returns:
      The final state of the transformation.
    """


class Iterator(object):
  """Interface for distributed iterators."""

  def get_next(self):
    """Unlike __next__, this may use a non-raising mechanism."""

  def __next__(self):
    pass

  def __iter__(self):
    pass
