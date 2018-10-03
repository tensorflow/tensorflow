# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Experimental shuffle ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.experimental.ops import shuffle_ops
from tensorflow.python.util import deprecation


@deprecation.deprecated(None,
                        "Use `tf.data.experimental.shuffle_and_repeat(...)`.")
def shuffle_and_repeat(buffer_size, count=None, seed=None):
  """Shuffles and repeats a Dataset returning a new permutation for each epoch.

  `dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size, count))`

  is equivalent to

  `dataset.shuffle(buffer_size, reshuffle_each_iteration=True).repeat(count)`

  The difference is that the latter dataset is not serializable. So,
  if you need to checkpoint an input pipeline with reshuffling you must use
  this implementation.

  Args:
    buffer_size: A `tf.int64` scalar `tf.Tensor`, representing the
      maximum number elements that will be buffered when prefetching.
    count: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the
      number of times the dataset should be repeated. The default behavior
      (if `count` is `None` or `-1`) is for the dataset be repeated
      indefinitely.
    seed: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the
      random seed that will be used to create the distribution. See
      `tf.set_random_seed` for behavior.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """
  return shuffle_ops.shuffle_and_repeat(buffer_size, count, seed)
