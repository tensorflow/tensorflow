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
"""Helper for choosing which op to run next in a distributed setting."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import ops as tf_ops


class OpQueue(object):
  """Class for choosing which Op to run next.

  Constructs an infinitely repeating sequence of Ops in shuffled order.

  In K-FAC, this can be used to distribute inverse update operations among
  workers.
  """

  def __init__(self, ops, seed=None):
    """Initializes an OpQueue.

    Args:
      ops: list of TensorFlow Ops. Ops to be selected from. All workers must
        initialize with the same set of ops.
      seed: int or None. Random seed used when shuffling order of ops.
    """
    self._ops_by_name = {op.name: op for op in ops}

    # Construct a (shuffled) Dataset with Op names.
    op_names = tf_ops.convert_to_tensor(list(sorted(op.name for op in ops)))
    op_names_dataset = (dataset_ops.Dataset.from_tensor_slices(op_names)
                        .shuffle(len(ops), seed=seed).repeat())
    self._next_op_name = op_names_dataset.make_one_shot_iterator().get_next()

  @property
  def ops(self):
    """Ops this OpQueue can return in next_op()."""
    return self._ops_by_name.values()

  def next_op(self, sess):
    """Chooses which op to run next.

    Note: This call will make a call to sess.run().

    Args:
      sess: tf.Session.

    Returns:
      Next Op chosen from 'ops'.
    """
    # In Python 3, type(next_op_name) == bytes. Calling bytes.decode('ascii')
    # returns a str.
    next_op_name = sess.run(self._next_op_name).decode('ascii')
    return self._ops_by_name[next_op_name]
