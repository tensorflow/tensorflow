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
"""The implementation of `tf.data.Dataset.counter`."""

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import ops


def _counter(start, step, dtype, name=None):
  with ops.name_scope("counter"):
    start = ops.convert_to_tensor(start, dtype=dtype, name="start")
    step = ops.convert_to_tensor(step, dtype=dtype, name="step")
    return (dataset_ops.Dataset.from_tensors(0, name=name).repeat(None).scan(
        start, lambda state, _: (state + step, state)))
