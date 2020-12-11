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
import threading


class TpuContext(threading.local):
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
  """A context manager setting current number of shards."""
  if _current_tpu_context.number_of_shards is not None:
    raise NotImplementedError(
        "tpu_shard_context cannot be nested."
        "If you're using TPUEstimator with inference_on_tpu, "
        "make sure you have set "
        "export_saved_model_api_version=ExportSavedModelApiVersion.V2 in "
        "the creation of TPUEstimator.")
  try:
    _current_tpu_context.set_number_of_shards(number_of_shards)
    yield
  finally:
    _current_tpu_context.set_number_of_shards(None)


def get_tpu_context():
  return _current_tpu_context


# Decorator function for tpu computation func that was passed to tpu.rewrite()
# if there is an embedded training loop in this func, trace tools will generate
# step markers for each iteration.
def on_device_training_loop(func):
  # Value for this attribute is from xla.DebugOptions.StepMarkerLocation.
  setattr(func, "step_marker_location", "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP")
  return func
