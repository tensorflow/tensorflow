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
"""Methods to allow dict of numpy arrays."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.estimator.inputs.numpy_io import numpy_input_fn as core_numpy_input_fn


def numpy_input_fn(x,
                   y=None,
                   batch_size=128,
                   num_epochs=1,
                   shuffle=True,
                   queue_capacity=1000,
                   num_threads=1):
  """This input_fn diffs from the core version with default `shuffle`."""
  return core_numpy_input_fn(x=x,
                             y=y,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_epochs=num_epochs,
                             queue_capacity=queue_capacity,
                             num_threads=num_threads)
