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
