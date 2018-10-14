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
# =============================================================================

"""Helper library for functions used during TPU compilation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib

from tensorflow.python.util import tf_inspect


class TpuContext(object):
  """A context object holding state about the TPU computation being built."""

  def __init__(self):
    """Creates a new TpuContext."""
    self._number_of_shards = None

  @property
  def number_of_shards(self):
    return self._number_of_shards

  def set_number_of_shards(self, number_of_shards):
    self._number_of_shards = number_of_shards


# The Tpu context holds the number of shards when a sharded computation is
# being built, or None if no computation is being built.
_current_tpu_context = TpuContext()


@contextlib.contextmanager
def tpu_shard_context(number_of_shards):
  if _current_tpu_context.number_of_shards is not None:
    raise NotImplementedError("tpu_shard_context cannot be nested.")
  try:
    _current_tpu_context.set_number_of_shards(number_of_shards)
    yield
  finally:
    _current_tpu_context.set_number_of_shards(None)


def get_tpu_context():
  return _current_tpu_context


def check_function_argument_count(func, input_arity, infeed_queue):
  """Validate the number of input arguments to a tpu function.

  Args:
    func: the Python function that will be called to generate the body of an XLA
      computation graph.
    input_arity: the number of explicit arguments supplied by the caller.
    infeed_queue: if not None, the infeed queue that will supply
      additional arguments to the function.

  Returns:
    None if function can be called with the supplied number of
      arguments, or an error string if it cannot.
  """
  def format_error(complaint, quantity):
    return "%s %d argument%s" % (complaint, quantity, ""
                                 if quantity == 1 else "s")

  number_of_arguments_needed = input_arity
  if infeed_queue is not None:
    number_of_arguments_needed += infeed_queue.number_of_tuple_elements
  arg_spec = tf_inspect.getargspec(func)
  number_of_args = len(arg_spec.args)
  if arg_spec.defaults is None:
    number_of_defaults = 0
  else:
    number_of_defaults = len(arg_spec.defaults)
  min_required_arguments = number_of_args - number_of_defaults
  if number_of_arguments_needed < min_required_arguments:
    # The required number of arguments is not enough to call the function.
    if number_of_defaults == 0 and arg_spec.varargs is None:
      return format_error("exactly", number_of_args)
    else:
      return format_error("at least", min_required_arguments)
  if arg_spec.varargs is None and number_of_arguments_needed > number_of_args:
    # The required number of arguments is too many to call the function.
    if number_of_defaults == 0:
      return format_error("exactly", number_of_args)
    else:
      return format_error("at most", number_of_args)
  # Since there are varargs, func can accept any number of arguments
  # greater than the minimum.
  return None
